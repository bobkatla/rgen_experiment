#!/bin/bash
#SBATCH --job-name=cifar10_base_sample
#SBATCH --output=cifar10_base_sample.out
#SBATCH --error=cifar10_base_sample.err
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
rgen sample --ckpt "$CKPT" --out-dir samples/cfg_w3_250 --steps 250 --guidance --w 3.0 --n-samples 50000

# 2) Evaluate FID against CIFAR-10 test
rgen eval-fid \
  --gen-dir samples/cfg_w3_250 \
  --reference folder \
  --ref-dir data/cifar10/test
