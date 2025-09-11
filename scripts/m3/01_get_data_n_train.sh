#!/bin/bash
#SBATCH --job-name=cifar10_base
#SBATCH --output=cifar10_base_train.out
#SBATCH --error=cifar10_base_train.err
#SBATCH --account=tx89
#SBATCH --time=90:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

module load miniforge3
mamba activate urc
rgen prep-data cifar10-export --root data
rgen train \
  --data-dir data/cifar10/train \
  --out-dir output/cifar10_unet_eps \
  --batch-size 256 --lr 2e-4 --timesteps 4000 --num-workers 4 \
  --iters 100000 --ema 0.9999 --amp --seed 0 --save-interval 10000
