# Spiking Neural Network Demos (snnTorch + Tonic)

This repo includes three runnable examples for event-based and frame-based SNNs:

1. **MNIST MLP SNN** — classic 784→300→10 network with Leaky IF neurons  
   File: `snn_mnist.py` (your cleaned script)

2. **NMNIST MLP SNN (Tonic)** — events → framed tensors → MLP-style SNN  
   File: `nmnist_mlp.py`

3. **DVSGesture 3-Layer Conv SNN (Tonic)** — events → framed tensors → Conv SNN  
   File: `dvs_conv3.py`

A companion notebook is included for quick, interactive runs:

- `SNN_tonic_snntorch_demo.ipynb`

---

## 1) Environment & Requirements

- Python 3.9–3.12
- PyTorch
- snnTorch
- Tonic (for NMNIST/DVSGesture)
- torchvision, numpy (and tqdm if you like progress bars)

```bash
# CPU-only example
pip install torch torchvision

# or follow PyTorch's site for CUDA builds, then:
pip install snntorch tonic numpy
```

> The first run will auto-download datasets to `./data/` (or a path you choose via `--save-to` for Tonic scripts).

---

## 2) File Overview

```
.
├── snn_mnist.py                 # Frame-based MNIST SNN (784→300→10, Leaky units)
├── nmnist_mlp.py                # NMNIST via Tonic + MLP SNN, frame-accumulated
├── dvs_conv3.py                 # DVSGesture via Tonic + 3-layer Conv SNN
└── SNN_tonic_snntorch_demo.ipynb# Notebook covering NMNIST + DVSGesture
```

---

## 3) Common CLI Flags (training & eval)

All scripts share similar flags:

- `--model-path` : where to load/save a checkpoint (default varies per script)
- `--batch-size` : batch size
- `--epochs`     : `0` = eval only (will **train for 1 epoch** if checkpoint missing); `>0` = train for N epochs
- `--lr`         : learning rate (Adam)
- `--grad-clip`  : gradient clipping norm (e.g., `1.0`)
- Subsampling flags for quick runs (names vary slightly by script):
  - MNIST: `--limit-train`, `--limit-test`
  - NMNIST/DVSGesture: `--train-samples`, `--eval-samples`
- Tonic-specific:
  - `--save-to`   : dataset root for Tonic datasets (default `./data`)
  - `--num-steps` : number of time bins for framing/simulation (e.g., `25` or `50`)
  - DVSGesture: `--resize` (downsample frames spatially; helps on CPU/low-VRAM)

---

## 4) Model Details & Commands

### A. MNIST MLP SNN — `snn_mnist.py`

- Architecture: `784 → 300 → 10` with `snn.Leaky` layers  
- Temporal handling: repeats the static image across `NUM_STEPS` in `forward`  
- Readout: **membrane-sum logits** over time (`mem.sum(dim=0) → CrossEntropy`)

**Train (quick demo)**
```bash
python mnist.py --model-path snnmodel2.pth   --epochs 2 --limit-train 2048 --limit-test 512   --batch-size 64 --lr 1e-3 --grad-clip 1.0
```

**Evaluate (loads checkpoint if present; otherwise 1 quick epoch)**
```bash
python mnist.py --model-path snnmodel2.pth --epochs 0
```

---

### B. NMNIST MLP SNN (Tonic) — `nmnist_mlp.py`

- Dataset: **NMNIST** (events) via `tonic.datasets.NMNIST`
- Transform: `ToFrame(..., n_time_bins=NUM_STEPS)` → frames shaped `[B,T,C,H,W]` with `C=2`
- Model: MLP per time step (flattened) + `snn.Leaky` layers
- Readout: **spike count** or **membrane-sum** (script uses mem-sum in training/eval)

**Train (if missing) + Evaluate**
```bash
python nmnist.py --save-to ./data --model-path ./nmnist_mlp.pth   --num-steps 25 --batch-size 64 --eval-samples 256   --train-epochs 2 --train-samples 2048 --lr 1e-3 --grad-clip 1.0
```

**Evaluate only**
```bash
python nmnist.py --save-to ./data --model-path ./nmnist_mlp.pth   --num-steps 25 --batch-size 64 --eval-samples 256
```

---

### C. DVSGesture Conv SNN (Tonic) — `dvs_conv3.py`

- Dataset: **DVSGesture** (events) via `tonic.datasets.DVSGesture`
- Transform: `ToFrame(..., n_time_bins=NUM_STEPS)` → optional **resize**  
  (A local `ResizeFrames` is provided for compatibility with older Tonic versions)
- Model: `Conv→LIF → Conv→LIF → Conv→LIF → Linear→LIF (readout)`
- Readout: **membrane-sum logits** over time

**Train (if missing) + Evaluate**
```bash
python dvs_conv3.py --save-to ./data --model-path ./dvs_conv3.pth   --num-steps 25 --batch-size 8 --eval-samples 64 --resize 64   --train-epochs 1 --train-samples 512 --lr 1e-3 --grad-clip 1.0
```

**Evaluate only**
```bash
python dvs_conv3.py --save-to ./data --model-path ./dvs_conv3.pth   --num-steps 25 --batch-size 8 --eval-samples 64 --resize 64
```

---

### Quick Start (Copy/Paste)

```bash
# MNIST (quick demo)
python mnist.py --model-path snnmodel2.pth --epochs 2 --limit-train 2048 --limit-test 512

# NMNIST (Tonic)
python nmnist.py --save-to ./data --model-path ./nmnist_mlp.pth   --num-steps 25 --batch-size 64 --eval-samples 256   --train-epochs 2 --train-samples 2048

# DVSGesture (Tonic, with resize)
python dvs_conv3.py --save-to ./data --model-path ./dvs_conv3.pth   --num-steps 25 --batch-size 8 --eval-samples 64 --resize 64   --train-epochs 1 --train-samples 512
```
