#!/bin/bash
#SBATCH --job-name=cifar10_sweeps_cfg
#SBATCH --output=cifar10_sweeps_cfg.out
#SBATCH --error=cifar10_sweeps_cfg.err
#SBATCH --account=tx89
#SBATCH --time=90:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load miniforge3
mamba activate urc
CKPT=output/cifar10_unet_eps_cfg/ckpt_ema_0100000.pt  # <- pick an EMA checkpoint
# 10k samples, steps=100, sweep w
rgen sweep-cfg \
  --ckpt "$CKPT" \
  --out-root sweeps/cifar10_cfg_fast \
  --weights 1,2,3,4 \
  --steps 100 \
  --n-samples 10000

