#!/bin/bash
#SBATCH --job-name=cifar10_cfg_sample_ddim
#SBATCH --output=cifar10_cfg_sample_ddim.out
#SBATCH --error=cifar10_cfg_sample_ddim.err
#SBATCH --account=tx89
#SBATCH --time=90:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load miniforge3
mamba activate urc
CKPT=output/cifar10_unet_eps_cfg/ckpt_ema_0200000.pt  # <- pick an EMA checkpoint

rgen sample \
  --ckpt "$CKPT" \
  --out-dir samples/cfg_ddim250_w3 \
  --sample-type ddim --guidance --w 3 --steps 250 \
  --n-samples 1000

rgen eval-fid --gen-dir samples/cfg_ddim250_w3 --reference folder --ref-dir data/cifar10/test
