# rgen — Running Generative AI Experiments (CIFAR-10 Diffusion Baseline)

A minimal, reproducible testbed for **class-conditional diffusion on CIFAR-10** with a clean path to inject **URC / region-aware loss**. The goal is to reproduce SOTA-style baselines first (ε-MSE, cosine schedule, EMA, CFG), then swap in new loss terms with everything else kept fixed.

## Features
- CIFAR-10 exporter → class-folder PNGs
- 32×32 U-Net (ADM-style) with label + time embeddings
- Gaussian diffusion (cosine schedule, ε-loss)
- AMP + EMA + JSONL logging
- Click-based CLI (`rgen`)
- Ready for classifier-free guidance (CFG) and URC hooks (next phases)

## Quickstart

```bash
# 1) Export CIFAR-10
uv run rgen prep-data cifar10-export --root data

# 2) Train baseline (no CFG yet)
uv run rgen train \
  --data-dir data/cifar10/train \
  --out-dir runs/cifar10_unet_eps \
  --batch-size 256 --lr 2e-4 --timesteps 4000 \
  --iters 100000 --ema 0.9999 --amp --seed 0
