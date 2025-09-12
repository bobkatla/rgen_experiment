#!/usr/bin/env bash
set -euo pipefail

uv run rgen eval-fid \
  --gen-dir samples/cfg_w3_250 \
  --reference folder \
  --ref-dir data/cifar10/test