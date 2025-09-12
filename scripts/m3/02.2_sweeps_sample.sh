#!/bin/bash
#SBATCH --job-name=cifar10_sweeps_sample
#SBATCH --output=cifar10_sweeps_sample.out
#SBATCH --error=cifar10_sweeps_sample.err
#SBATCH --account=tx89
#SBATCH --time=90:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

module load miniforge3
mamba activate urc
# 1) Sample 50k with DDIM + CFG (w=3)
CKPT=output/cifar10_unet_eps/ckpt_ema_0090000.pt  # <- pick an EMA checkpoint
# 10k samples, steps=100, sweep w
uv run rgen sweep-cfg \
  --ckpt "$CKPT" \
  --out-root sweeps/cifar10_cfg_fast \
  --weights 1,2,3,4 \
  --steps 100 \
  --n-samples 10000
