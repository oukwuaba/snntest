#!/usr/bin/env python3
"""
Tutorial 7 (Neuromorphic Datasets with Tonic + snnTorch) — CLI version
- Loads a trained model if it exists; otherwise trains and saves the BEST checkpoint
- Uses N-MNIST via Tonic, with optional disk/memory caching
- Matches the Tutorial 7 architecture and training recipe

Quick start:
  python -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install snntorch tonic torch torchvision
  python tutorial7_cli.py --save-to ./data --cache ./cache \
      --model-path ./nmnist_csnntorch.pth --epochs 1 --num-iters 50
"""

import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils as snn_utils

import tonic
import tonic.transforms as ttransforms
from tonic import DiskCachedDataset, MemoryCachedDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/evaluate a CSNN on N-MNIST (Tutorial 7) with Tonic + snnTorch"
    )
    p.add_argument("--save-to", type=str, default="./data",
                   help="Where to store/download datasets")
    p.add_argument("--cache", type=str, default="./cache",
                   help="Root directory for cached frames")
    p.add_argument("--memory-cache", action="store_true",
                   help="Use MemoryCachedDataset instead of DiskCachedDataset (needs RAM)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--num-iters", type=int, default=50,
                   help="Max training iterations per epoch (early break)")
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Preferred device; 'auto' tries CUDA then MPS then CPU")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable RandomRotation on training frames")
    p.add_argument("--model-path", type=str, default="./nmnist_csnntorch.pth",
                   help="Checkpoint path for saving/loading the model")
    p.add_argument("--force-train", action="store_true",
                   help="Ignore existing checkpoint and train anyway")
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    if choice == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] MPS requested but not available; falling back to CPU")
        return torch.device("cpu")
    if choice == "cpu":
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_datasets(save_to: str) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    # As in Tutorial 7: Denoise + ToFrame(sensor_size, time_window=1000us)
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = ttransforms.Compose([
        ttransforms.Denoise(filter_time=10000),
        ttransforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])
    trainset = tonic.datasets.NMNIST(save_to=save_to, transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to=save_to, transform=frame_transform, train=False)
    return trainset, testset


def build_cached_loaders(trainset, testset, cache_root: str, batch_size: int,
                         use_mem_cache: bool, augment: bool):
    os.makedirs(cache_root, exist_ok=True)
    train_cache = os.path.join(cache_root, "nmnist", "train")
    test_cache = os.path.join(cache_root, "nmnist", "test")
    os.makedirs(train_cache, exist_ok=True)
    os.makedirs(test_cache, exist_ok=True)

    # Apply same rotation to the tensor sample (frames tensor). If this ever trips you up,
    # run with --no-augment for a clean baseline.
    if augment:
        train_aug = ttransforms.Compose([
            torch.from_numpy,
            torchvision.transforms.RandomRotation([-10, 10]),
        ])
    else:
        train_aug = ttransforms.Compose([torch.from_numpy])

    if use_mem_cache:
        cached_train = MemoryCachedDataset(trainset, transform=train_aug)
        cached_test = MemoryCachedDataset(testset)  # no augmentation on test
    else:
        cached_train = DiskCachedDataset(trainset, transform=train_aug, cache_path=train_cache)
        cached_test = DiskCachedDataset(testset, cache_path=test_cache)

    # Collate to [T, B, C, H, W]
    train_loader = DataLoader(
        cached_train,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=True,
        num_workers=os.cpu_count() if os.name != "nt" else 0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        cached_test,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=False,
        num_workers=os.cpu_count() if os.name != "nt" else 0,
        pin_memory=False,
    )
    return train_loader, test_loader


def build_net(device: torch.device) -> nn.Module:
    # 12C5 - MP2 - LIF - 32C5 - MP2 - LIF - Flatten - FC(800->10) - LIF(output)
    spike_grad = surrogate.atan()
    beta = 0.5
    net = nn.Sequential(
        nn.Conv2d(2, 12, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Conv2d(12, 32, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, 10),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    ).to(device)
    return net


def forward_pass(net: nn.Module, data: torch.Tensor) -> torch.Tensor:
    """
    Forward over time (autograd ENABLED for training).
    data: [T, B, C, H, W] -> returns [T, B, num_classes]
    """
    spk_rec = []
    snn_utils.reset(net)  # reset hidden states of all LIF layers each sequence
    for t in range(data.size(0)):
        spk_out, _ = net(data[t])
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)


@torch.no_grad()
def evaluate(net: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = None) -> float:
    """Mean accuracy over loader; optionally limit to first max_batches for speed."""
    net.eval()
    n = 0
    acc_sum = 0.0
    for i, (data, targets) in enumerate(loader):
        data = data.to(device)
        targets = targets.to(device)
        spk_rec = forward_pass(net, data)
        acc_sum += float(SF.accuracy_rate(spk_rec, targets))
        n += 1
        if max_batches is not None and (i + 1) >= max_batches:
            break
    return acc_sum / max(n, 1)


def train_and_maybe_save(args, net, train_loader, test_loader, device) -> None:
    torch.manual_seed(args.seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    best_val = -1.0
    os.makedirs(os.path.dirname(os.path.abspath(args.model_path)) or ".", exist_ok=True)
    num_print = max(1, args.num_iters // 10)

    print("[info] Starting training ...")
    t0 = time.time()
    for epoch in range(args.train_epochs):
        iters = 0
        net.train()
        for i, (data, targets) in enumerate(train_loader):
            data = data.to(device, dtype=torch.float32)
            targets = targets.to(device)

            spk_rec = forward_pass(net, data)           # requires grad
            loss_val = loss_fn(spk_rec, targets)

            optimizer.zero_grad(set_to_none=True)
            loss_val.backward()
            optimizer.step()

            if i % num_print == 0:
                # quick running val check on a few test batches to keep it snappy
                val_acc = evaluate(net, test_loader, device, max_batches=5)
                print(f"[epoch {epoch}] iter {i:04d} | loss {loss_val.item():6.3f} | val_acc {val_acc*100:5.2f}%")

                # Save best
                if val_acc > best_val:
                    best_val = val_acc
                    torch.save(
                        {
                            "model_state": net.state_dict(),
                            "val_acc": best_val,
                            "epoch": epoch,
                            "iter": i,
                            "args": vars(args),
                        },
                        args.model_path,
                    )
                    print(f"[checkpoint] Saved best model to {args.model_path} (val_acc={best_val*100:.2f}%)")

            iters += 1
            if iters >= args.num_iters:
                break

    print(f"[info] Training finished in {time.time()-t0:.1f}s")
    # Final, full evaluation
    final_val = evaluate(net, test_loader, device, max_batches=None)
    print(f"[info] Full validation accuracy (post-train): {final_val*100:.2f}%")

    # Ensure something is saved even if best-save never triggered
    torch.save(
        {
            "model_state": net.state_dict(),
            "val_acc": final_val,
            "epoch": args.train_epochs - 1,
            "args": vars(args),
        },
        args.model_path,
    )
    print(f"[checkpoint] Saved model to {args.model_path} (val_acc={final_val*100:.2f}%)")
    # if best_val < 0:
    #     torch.save(
    #         {
    #             "model_state": net.state_dict(),
    #             "val_acc": final_val,
    #             "epoch": args.train_epochs - 1,
    #             "args": vars(args),
    #         },
    #         args.model_path,
    #     )
    #     print(f"[checkpoint] Saved model to {args.model_path} (val_acc={final_val*100:.2f}%)")


def main():
    args = parse_args()
    os.makedirs(args.save_to, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[info] Using device: {device}")

    print("[info] Preparing datasets ...")
    trainset, testset = build_datasets(args.save_to)

    print("[info] Building cached dataloaders ...")
    train_loader, test_loader = build_cached_loaders(
        trainset,
        testset,
        cache_root=args.cache,
        batch_size=args.batch_size,
        use_mem_cache=args.memory_cache,
        augment=not args.no_augment,
    )

    net = build_net(device)

    # ----- Load-if-exists, else train-and-save-best -----
    if os.path.isfile(args.model_path) and not args.force_train:
        print(f"[info] Found checkpoint at {args.model_path}. Loading ...")
        ckpt = torch.load(args.model_path, map_location=device)
        net.load_state_dict(ckpt["model_state"])
        net.to(device)
        val = evaluate(net, test_loader, device)
        print(f"[info] Loaded model | validation accuracy: {val*100:.2f}%")
        return

    if not os.path.isfile(args.model_path):
        print(f"[info] No checkpoint found at {args.model_path}. Training from scratch.")
    else:
        print(f"[info] --force-train specified. Training from scratch and will overwrite {args.model_path} if improved.")

    train_and_maybe_save(args, net, train_loader, test_loader, device)


if __name__ == "__main__":
    main()
