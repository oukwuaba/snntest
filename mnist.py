#!/usr/bin/env python3
"""
Simple Leaky SNN on MNIST (MLP-style) with fallback training.

- Uses your original Net (784 → 300 → 10 with LIF layers).
- Repeats the static image across time steps inside Net.forward (no encoder).
- Trains if a checkpoint is missing (or if --epochs > 0).
- Classifies using membrane-sum logits over time.

Usage:
  python snn_mnist.py --model-path snnmodel2.pth --epochs 2 --limit-train 2048 --limit-test 512
"""

# ==================== Imports ====================
import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tvt
import snntorch as snn

# ==================== Globals ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_STEPS = 100  # Number of simulation steps (time bins) for the SNN


# ==================== Model ====================
class Net(nn.Module):
    """Your original 784-300-10 MLP with Leaky neurons."""

    def __init__(self):
        super().__init__()
        num_inputs = 784
        num_hidden = 300
        num_outputs = 10

        beta1 = 0.9
        beta2 = torch.rand((num_outputs,), dtype=torch.float)

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta1)                   # fixed beta
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2, learn_beta=True)  # learnable beta

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 1, 28, 28]
        Returns:
            spk2_rec: [T, B, 10]
            mem2_rec: [T, B, 10]
        """
        # Initialize states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec, mem2_rec = [], []

        # Repeat the static input for NUM_STEPS
        B = x.size(0)
        x_flat = x.view(B, -1)  # [B, 784]

        for _ in range(NUM_STEPS):
            cur1 = self.fc1(x_flat)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)


def load_model(model_path: str = "snnmodel2.pth") -> Net:
    """Load a pretrained SNN if available; otherwise return randomly-initialized net."""
    net = Net().to(DEVICE)
    try:
        net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Using randomly initialized model.")
        print("For proper testing, please ensure you have a trained model.")
    net.eval()
    return net


# ==================== Data ====================
def get_loaders(batch_size: int = 64, limit_train: int = -1, limit_test: int = -1):
    """
    Build MNIST train/test DataLoaders.
    limit_* = -1 uses full split; otherwise subsample N items for quick runs.
    """
    transform = tvt.Compose([tvt.ToTensor()])  # [0,1], float32

    train_set = torchvision.datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_set  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    if limit_train != -1:
        train_set = torch.utils.data.Subset(train_set, list(range(min(limit_train, len(train_set)))))
    if limit_test != -1:
        test_set  = torch.utils.data.Subset(test_set,  list(range(min(limit_test,  len(test_set)))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ==================== Train / Eval ====================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate with membrane-sum logits over time.
    Returns accuracy in [0,1].
    """
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, mem = model(x)         # [T, B, 10]
        logits = mem.sum(dim=0)   # [B, 10]
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return correct / max(1, total)


def train(model: nn.Module,
          train_loader: DataLoader,
          test_loader: DataLoader,
          epochs: int = 1,
          lr: float = 1e-3,
          grad_clip: float = 1.0) -> None:
    """
    Simple BPTT: CE(membrane-sum logits, label).
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            opt.zero_grad()
            _, mem = model(x)          # [T, B, 10]
            logits = mem.sum(dim=0)    # [B, 10]
            loss = criterion(logits, y)
            loss.backward()

            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            running += loss.item() * y.size(0)

        train_loss = running / max(1, len(train_loader.dataset))
        test_acc = evaluate(model, test_loader)
        print(f"Epoch {ep}: train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")


# ==================== Utilities ====================
def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== Main (CLI) ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="snnmodel2.pth")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=0, help="If 0: eval only (train if missing). If >0: train.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--limit-train", type=int, default=-1, help="-1 full; otherwise N samples for quick training")
    ap.add_argument("--limit-test", type=int, default=-1, help="-1 full; otherwise N samples for quick eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    # Data
    train_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )

    # Model
    net = Net().to(DEVICE)

    # Load or train
    need_train = (args.epochs > 0) or (not os.path.exists(args.model_path))
    if not need_train:
        try:
            net.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
            print(f"Model loaded successfully from {args.model_path}")
        except Exception as e:
            print(f"Failed to load model ({e}). Will train instead.")
            need_train = True

    if need_train:
        if args.epochs == 0:
            print("Checkpoint missing; running a quick 1-epoch training...")
        epochs = max(1, args.epochs)
        train(net, train_loader, test_loader, epochs=epochs, lr=args.lr, grad_clip=args.grad_clip)

        # Save
        try:
            torch.save(net.state_dict(), args.model_path)
            print(f"Saved model to {args.model_path}")
        except Exception as e:
            print(f"Warning: could not save model: {e}")

    # Final evaluation
    acc = evaluate(net, test_loader)
    print(f"Test accuracy (membrane-sum logits): {acc:.4f}")


if __name__ == "__main__":
    main()
