#!/bin/bash
#SBATCH --job-name=cifar10_cfg_sample
#SBATCH --output=cifar10_cfg_sample.out
#SBATCH --error=cifar10_cfg_sample.err
#SBATCH --account=tx89
#SBATCH --time=90:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load miniforge3
mamba activate urc
CKPT=output/updated_cifar10_unet_eps_cfg_urc/ckpt_ema_0200000.pt  # <- pick an EMA checkpoint

rgen sample \
  --ckpt "$CKPT" \
  --out-dir samples/urc_final \
  --sampler auto --guidance auto --w 3 --steps 250 \
  --n-samples 5000

rgen eval-fid --gen-dir samples/urc_final --reference folder --ref-dir data/cifar10/test
