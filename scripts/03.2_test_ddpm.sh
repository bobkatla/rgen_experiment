#!/usr/bin/env bash
set -euo pipefail

CKPT=output/cifar10_unet_eps/ckpt_ema_0100000.pt 

uv run rgen sample --ckpt "$CKPT" \
  --out-dir samples/base_ddpm4000 --steps 4000 --no-guidance --n-samples 256 --sample-type ddpm

# Evaluate FID against CIFAR-10 test
uv run rgen eval-fid \
  --gen-dir samples/base_ddpm4000 \
  --reference folder \
  --ref-dir data/cifar10/test