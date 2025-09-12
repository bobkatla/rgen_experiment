#!/usr/bin/env bash
set -euo pipefail

# 1) Sample 50k with DDIM + CFG (w=3)
CKPT=output/cifar10_unet_eps/ckpt_ema_0001000.pt  # <- pick an EMA checkpoint
uv run rgen sample --ckpt "$CKPT" --out-dir samples/cfg_w3_250 --steps 250 --guidance --w 3.0 --n-samples 50000

# 2) Evaluate FID against CIFAR-10 test
uv run rgen eval-fid \
  --gen-dir samples/cfg_w3_250 \
  --reference folder \
  --ref-dir data/cifar10/test