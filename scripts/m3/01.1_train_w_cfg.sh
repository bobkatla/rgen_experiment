#!/bin/bash
#SBATCH --job-name=cifar10_cfg_train
#SBATCH --output=cifar10_cfg_train.out
#SBATCH --error=cifar10_cfg_train.err
#SBATCH --account=tx89
#SBATCH --time=90:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
#SBATCH --partition=gpu

module load miniforge3
mamba activate urc
rgen train \
  --data-dir data/cifar10/train \
  --out-dir runs/cifar10_unet_eps_cfg \
  --batch-size 256 --lr 2e-4 --timesteps 4000 \
  --iters 200000 --ema 0.9999 --amp --seed 0 \
  --label-drop-prob 0.1 --num-workers 3
