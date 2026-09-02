#!/usr/bin/env bash
# Server foreground command (keep the SSH terminal open to view progress):
# cd /home/yqwang/project/PHENet && CUDA_VISIBLE_DEVICES=1 bash train_scripts/train_sysu_cd_foggy.sh
# Server background command:
# cd /home/yqwang/project/PHENet && CUDA_VISIBLE_DEVICES=1 nohup bash train_scripts/train_sysu_cd_foggy.sh > /tmp/phenet_sysu_cd_foggy.launch.log 2>&1 &
# The test split (not SYSU's val split) is intentionally used to select best.
# Training log monitor command:
# tail -f /storage/yqwang/PHENet/saved_models/Run1/SYSU-CD-foggy/train.log

set -euo pipefail
# Expose only server physical GPU 1. It becomes logical cuda:0 in Python.
export CUDA_VISIBLE_DEVICES=1

# Prevent CUDA libraries from other environments from contaminating PHENet.
unset LD_LIBRARY_PATH

PROJECT_ROOT="/home/yqwang/project/PHENet"
SAVE_ROOT="/storage/yqwang/PHENet/saved_models/Run1"
PYTHON_BIN="${PYTHON_BIN:-/home/yqwang/miniconda3/envs/phenet/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}; verify the existing phenet conda environment." >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" models/train.py \
  --data-root /storage/BCD-foggy/SYSU-CD-foggy \
  --dataset-name SYSU-CD-foggy \
  --save-dir "${SAVE_ROOT}" \
  --val-split test \
  --label-dir GT \
  --epochs 100 \
  --batch-size 8 \
  --test-batch-size 8 \
  --lr 1e-4 \
  --workers 8 \
  --gpu-ids 0 \
  --sync-bn
