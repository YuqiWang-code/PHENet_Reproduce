#!/usr/bin/env bash

set -euo pipefail

# Physical GPU 1 -> logical cuda:0.
export CUDA_VISIBLE_DEVICES=1

# Avoid CUDA-library contamination from other environments.
unset LD_LIBRARY_PATH

PROJECT_ROOT="/home/yqwang/project/PHENet"
PYTHON_BIN="/home/yqwang/miniconda3/envs/phenet/bin/python"
SAVE_ROOT="/storage/yqwang/PHENet/saved_models/Ablations"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing ${PYTHON_BIN}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" models/train.py \
    --data-root /storage/BCD-foggy/LEVIR-CD-foggy \
    --dataset-name LEVIR-CD-foggy_bce-fg-dice \
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
    --pseudo-mode frozen \
    --change-loss-mode bce_fg_dice