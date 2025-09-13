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
CKPT=output/cifar10_unet_eps/ckpt_ema_0100000.pt  # <- pick an EMA checkpoint

# A) Re-sample with the fixed embedding (no retrain): sometimes enough to decode better
rgen sample --ckpt "$CKPT" \
  --out-dir samples/base_ddim250_noguid_fixpos --steps 250 --no-guidance --n-samples 1000

# Evaluate FID against CIFAR-10 test
rgen eval-fid \
  --gen-dir samples/base_ddim250_noguid_fixpos \
  --reference folder \
  --ref-dir data/cifar10/test

# B) DDPM-1000 fallback
rgen sample --ckpt "$CKPT" \
  --out-dir samples/base_ddpm250 --steps 250 --no-guidance --n-samples 1000 --sample-type ddpm

# Evaluate FID against CIFAR-10 test
rgen eval-fid \
  --gen-dir samples/base_ddpm250 \
  --reference folder \
  --ref-dir data/cifar10/test
