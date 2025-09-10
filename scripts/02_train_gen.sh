#!/usr/bin/env bash
set -euo pipefail
# Train (baseline; no CFG yet)
uv run rgen train \
  --data-dir data/cifar10/train \
  --out-dir output/cifar10_unet_eps \
  --batch-size 256 --lr 2e-4 --timesteps 4000 --num-workers 4 \
  --iters 100000 --ema 0.9999 --amp --seed 0 --save-interval 10000
