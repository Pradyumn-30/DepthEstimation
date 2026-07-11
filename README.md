# Self-Supervised Monocular Depth Estimation

A Monodepth2-style self-supervised monocular depth estimation pipeline built from scratch in PyTorch. Predicts dense depth maps from single images using only monocular video sequences for training — no ground truth depth required.

## Architecture

| Component | Details |
|-----------|---------|
| Depth Encoder | ResNet18 (ImageNet pretrained), 5 multi-scale feature maps |
| Depth Decoder | U-Net with skip connections, 4-scale disparity output |
| Pose Network | Separate ResNet18 encoder, 6-DOF relative pose prediction |
| Training Signal | Photometric consistency (L1 + SSIM) via view synthesis |
| Key Innovations | Auto-masking, minimum reprojection, edge-aware smoothness |

## Dataset

Trained on a **curated subset of the KITTI Eigen split** (10 drives, ~2,400 frames) covering city, residential, and road scenes. The curated drives are strictly disjoint across train/val/test and are all drawn from the Eigen training set — evaluation against the full 697-image Eigen test set remains valid.

> **Honest scope note**: This is a curated subset, not the full ~22,600-image Eigen training split. Metrics should be compared against models trained on similarly-sized data, not full-split baselines.

## Quick Start

### 1. Setup
```bash
# Create virtual environment (using uv)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Verify MPS (Apple Silicon only)
```bash
python scripts/verify_mps.py
```

### 3. Download KITTI Data
```bash
python scripts/download_kitti.py --output data/kitti_raw --split eigen_curated
```

### 4. Train (Phase 1: local M4, reduced resolution)
```bash
python scripts/train.py --config configs/local_debug.yaml
```

### 5. Evaluate
```bash
python scripts/evaluate.py --checkpoint checkpoints/epoch_04.pt --config configs/local_debug.yaml
```

### 6. Export to CoreML
```bash
python depth/export/coreml_export.py --checkpoint checkpoints/epoch_04.pt
```

### 7. Run Tests
```bash
pytest tests/ -v
```

## Project Structure

```
├── configs/               # YAML training configs
│   ├── local_debug.yaml   # M4 local (128×416, batch 4, 5 epochs)
│   └── default.yaml       # AWS A10G (192×640, batch 12, 20 epochs)
├── data/splits/           # Train/val/test file lists
├── depth/
│   ├── datasets/          # KITTI dataloader + augmentations
│   ├── models/            # ResNet18 encoder, depth decoder, pose decoder
│   ├── losses/            # Photometric, smoothness, auto-masking
│   ├── geometry/          # 3D projection, grid_sample warping
│   ├── export/            # CoreML export + validation
│   └── utils/             # Device selection
├── scripts/               # Training, evaluation, MPS verification, download
└── tests/                 # Unit tests (geometry, dataloader, forward pass)
```

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Local pipeline (M4) | **In progress** | Verified pipeline at reduced res |
| 2. Full training (AWS) | Planned | A10G, 192×640, 20 epochs |
| 3. CI/CD | Planned | GitHub Actions: unit tests, smoke test, regression |
| 4. C++ deployment | Planned | CoreML/TensorRT inference in C++ |
| 5. Multi-modal (optional) | Planned | Sparse LiDAR auxiliary loss |

## License

MIT
