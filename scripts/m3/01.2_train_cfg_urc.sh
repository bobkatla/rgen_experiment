#!/bin/bash
#SBATCH --job-name=cifar10_cfg_urc_train
#SBATCH --output=cifar10_cfg_urc_train.out
#SBATCH --error=cifar10_cfg_urc_train.err
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
  --out-dir output/updated_cifar10_unet_eps_cfg_urc \
  --batch-size 256 --lr 2e-4 --timesteps 4000 \
  --iters 200000 --ema 0.9999 --amp --seed 0 \
  --label-drop-prob 0.1 --num-workers 3 \
  --use-urc --urc-weight 0.2 --urc-alpha 0.10 --urc-dy 128
