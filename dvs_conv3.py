#!/usr/bin/env python3
"""
DVSGesture 5-layer SNN (4x Conv→LIF + Linear→LIF) using Tonic + snnTorch.
Saves the BEST checkpoint by validation accuracy.

Usage (CLI):
  python dvs_conv5.py --save-to ./data --model-path ./dvs_conv5.pth \
    --num-steps 25 --batch-size 8 --eval-samples 64 --resize 64 \
    --train-epochs 1 --train-samples 512 --lr 1e-3 --grad-clip 1.0
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import tonic
from tonic.transforms import ToFrame, Compose

import snntorch as snn
import torch.nn.functional as F

NUM_CLASSES = 11

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvSNN5(nn.Module):
    """
    5 neuron layers total:
      1) Conv1 → LIF1
      2) Conv2 → LIF2
      3) Conv3 → LIF3
      4) Conv4 → LIF4
      5) Linear head → LIF5 (readout)

    All LIF membrane states are initialized with torch.ones_like(current) each sequence.
    """
    def __init__(
        self,
        in_channels=2,
        img_size=64,
        beta=0.9,
        hidden_channels=(16, 32, 64, 128),  # 4 conv stages
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        c1, c2, c3, c4 = hidden_channels

        # Four downsampling convs (stride=2) → spatial /16
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2)   # /2
        self.lif1  = snn.Leaky(beta=beta)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)            # /4
        self.lif2  = snn.Leaky(beta=beta)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)            # /8
        self.lif3  = snn.Leaky(beta=beta)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)            # /16
        self.lif4  = snn.Leaky(beta=beta)

        feat_hw = img_size // 16
        self.head = nn.Linear(c4 * feat_hw * feat_hw, num_classes)
        # 5th neuron layer (readout)
        self.readout = snn.Leaky(beta=torch.ones(num_classes), learn_beta=True)

    def forward(self, x, num_steps: int):
        """
        x: [B, T, C, H, W]
        returns: spk_rec [T,B,num_classes], mem_rec [T,B,num_classes]
        """
        B, T, C, H, W = x.shape
        spk_rec, mem_rec = [], []

        # Membrane states, initialized with ones_like(...) at first step
        mem1 = mem2 = mem3 = mem4 = mem5 = None

        for t in range(num_steps):
            xt = x[:, t]  # [B, C, H, W]

            cur1 = self.conv1(xt)
            if mem1 is None: mem1 = torch.ones_like(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.conv2(spk1)
            if mem2 is None: mem2 = torch.ones_like(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.conv3(spk2)
            if mem3 is None: mem3 = torch.ones_like(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.conv4(spk3)
            if mem4 is None: mem4 = torch.ones_like(cur4)
            spk4, mem4 = self.lif4(cur4, mem4)

            flat = spk4.reshape(B, -1)
            cur5 = self.head(flat)
            if mem5 is None: mem5 = torch.ones_like(cur5)
            spk5, mem5 = self.readout(cur5, mem5)

            spk_rec.append(spk5)
            mem_rec.append(mem5)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

class ResizeFrames:
    """Resize framed events [T,H,W,C] to (size,size) using bilinear on torch."""
    def __init__(self, size): self.size = size
    def __call__(self, frames):
        import numpy as np
        if isinstance(frames, np.ndarray):
            x = torch.from_numpy(frames).float()  # [T,H,W,C]
        else:
            x = frames.float()
        if x.ndim == 4 and x.shape[-1] in (1,2):
            x = x.permute(0,3,1,2).contiguous()   # [T,C,H,W]
        m = x.max()
        if torch.isfinite(m) and m > 1:
            x = x / m
        x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
        x = x.permute(0,2,3,1).contiguous().cpu().numpy()  # [T,H,W,C]
        return x

def make_transforms(num_steps, resize):
    tfs = [ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, n_time_bins=num_steps)]
    if resize and resize > 0:
        tfs.append(ResizeFrames(resize))
    return Compose(tfs)

def load_dvs(save_to: str, num_steps: int, split: str = "test", resize: int = 64):
    assert split in {"train", "test"}
    transform = make_transforms(num_steps, resize)
    return tonic.datasets.DVSGesture(save_to=save_to, train=(split=="train"), transform=transform)

def ensure_shape(frames: torch.Tensor):
    # [T,H,W,C] → [T,C,H,W], float32 in [0,1]
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    if frames.ndim == 4 and frames.shape[-1] in (1, 2):
        frames = frames.permute(0, 3, 1, 2).contiguous()
    frames = frames.to(dtype=torch.float32)
    maxv = frames.max()
    if torch.isfinite(maxv) and maxv > 1:
        frames = frames / maxv
    return frames

def collate_fn(batch):
    xs, ys = [], []
    for frames, label in batch:
        xs.append(ensure_shape(frames))
        ys.append(label)
    x = torch.stack(xs, dim=0)  # [B, T, C, H, W]
    y = torch.tensor(ys, dtype=torch.long)
    return x, y

@torch.no_grad()
def evaluate(model, loader, device, num_steps):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        spk, mem = model(x, num_steps=num_steps)
        logits = mem.sum(dim=0)  # [B, 11]
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def train(model, train_loader, test_loader, device, num_steps,
          epochs=1, lr=1e-3, grad_clip=1.0, best_path:str=None):
    """Train with BPTT; save the *best* checkpoint by validation accuracy."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0

    for ep in range(1, epochs+1):
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            spk, mem = model(x, num_steps=num_steps)
            logits = mem.sum(dim=0)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item()*y.size(0)

        train_loss = running / max(1, len(train_loader.dataset))
        val_acc = evaluate(model, test_loader, device, num_steps)
        print(f"Epoch {ep}: train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if best_path and val_acc > best_acc:
            best_acc = val_acc
            try:
                torch.save(model.state_dict(), best_path)
                print(f"  ↑ New best ({best_acc:.4f}) saved to {best_path}")
            except Exception as e:
                print(f"Warning: could not save best model: {e}")

    return best_acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-to", type=str, default="./data", help="Where to store/download DVSGesture")
    ap.add_argument("--model-path", type=str, default="./dvs_conv5.pth", help="Path to load a pretrained model")
    ap.add_argument("--best-model-path", type=str, default="", help="Where to save the BEST model (defaults to --model-path)")
    ap.add_argument("--num-steps", type=int, default=25, help="Number of time bins / simulation steps")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-samples", type=int, default=64, help="Limit eval set for quick run; -1 = full test")
    ap.add_argument("--resize", type=int, default=64, help="Resize frames to H=W=resize; 0 disables")
    # Training controls
    ap.add_argument("--train-epochs", type=int, default=0, help="If >0, force training even if model exists")
    ap.add_argument("--train-samples", type=int, default=512, help="Subsample train for a quick demo; -1 = full")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Test data
    testset = load_dvs(args.save_to, args.num_steps, split="test", resize=args.resize)
    if args.eval_samples != -1:
        test_indices = list(range(min(len(testset), args.eval_samples)))
        testset = torch.utils.data.Subset(testset, test_indices)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    img_size = args.resize if args.resize and args.resize > 0 else tonic.datasets.DVSGesture.sensor_size[0]
    net = ConvSNN5(in_channels=2, img_size=img_size).to(device)

    # Load or train
    need_train = args.train_epochs > 0 or (not os.path.exists(args.model_path))
    if not need_train:
        try:
            state = torch.load(args.model_path, map_location=device)
            net.load_state_dict(state)
            print(f"Loaded pretrained weights from {args.model_path}")
        except Exception as e:
            print(f"Failed to load model ({e}). Will train instead.")
            need_train = True

    if need_train:
        print("Starting training... (saving BEST checkpoint only; DVSGesture can be heavy)")
        trainset = load_dvs(args.save_to, args.num_steps, split="train", resize=args.resize)
        if args.train_samples != -1:
            train_indices = list(range(min(len(trainset), args.train_samples)))
            trainset = torch.utils.data.Subset(trainset, train_indices)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        best_path = args.best_model_path if args.best_model_path else args.model_path
        best_acc = train(
            net, trainloader, testloader, device, num_steps=args.num_steps,
            epochs=max(200, args.train_epochs), lr=args.lr, grad_clip=args.grad_clip,
            best_path=best_path
        )

        # Reload the best weights before final eval
        try:
            net.load_state_dict(torch.load(best_path, map_location=device))
            print(f"Reloaded best model from {best_path} (val_acc={best_acc:.4f})")
        except Exception as e:
            print(f"Warning: could not reload best model ({e}). Continuing with current weights.")

    # Final evaluation
    acc = evaluate(net, testloader, device, num_steps=args.num_steps)
    print(f"Test accuracy (mem-sum logits): {acc:.4f}")

if __name__ == "__main__":
    main()
