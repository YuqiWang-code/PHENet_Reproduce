#!/usr/bin/env bash

set -euo pipefail

# Formal project rule:
# physical GPU 1 -> logical cuda:0
export CUDA_VISIBLE_DEVICES=1
unset LD_LIBRARY_PATH

PROJECT_ROOT="/home/yqwang/project/PHENet"
PYTHON_BIN="/home/yqwang/miniconda3/envs/phenet/bin/python"
SAVE_ROOT="/storage/yqwang/PHENet/saved_models/Ablations"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" models/train.py \
    --data-root /storage/BCD-foggy/LEVIR-CD-foggy \
    --dataset-name LEVIR-CD-foggy_pseudo-zero \
    --save-dir "${SAVE_ROOT}" \
    --val-split test \
    --label-dir label \
    --epochs 80 \
    --batch-size 8 \
    --test-batch-size 8 \
    --lr 1e-4 \
    --workers 8 \
    --gpu-ids 0 \
    --sync-bn \
    --pseudo-mode zero