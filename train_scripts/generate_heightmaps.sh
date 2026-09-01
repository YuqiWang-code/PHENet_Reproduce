#!/usr/bin/env bash
# Server foreground command (forced to physical GPU 1 by this script):
# cd /home/yqwang/project/PHENet && CUDA_VISIBLE_DEVICES=1 bash train_scripts/generate_heightmaps.sh
# Server background command:
# cd /home/yqwang/project/PHENet && CUDA_VISIBLE_DEVICES=1 nohup bash train_scripts/generate_heightmaps.sh > /tmp/phenet_generate_heightmaps.log 2>&1 &
# Background log monitor command:
# tail -f /tmp/phenet_generate_heightmaps.log

set -euo pipefail

# Expose only server physical GPU 1. It becomes logical cuda:0 in Python.
export CUDA_VISIBLE_DEVICES=1

# Prevent CUDA libraries from other environments from contaminating PHENet.
unset LD_LIBRARY_PATH

PROJECT_ROOT="/home/yqwang/project/PHENet"
DATA_ROOT="/storage/BCD-foggy"

RDAH_CHECKPOINT="${PROJECT_ROOT}/pre_checkpoint/checkpoints-track1/104best_model.pth"
DEPTH_MODEL="${PROJECT_ROOT}/models/third_partys/Depth-Anything-V2-Large-hf"

PYTHON_BIN="${PYTHON_BIN:-/home/yqwang/miniconda3/envs/phenet/bin/python}"

export HF_HUB_OFFLINE=1

# Debug-only variables must not leak into the formal long-running job.
unset CUDA_LAUNCH_BLOCKING || true
unset PYTORCH_CUDA_ALLOC_CONF || true

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -d "${DEPTH_MODEL}" ]]; then
    echo "Missing Depth Anything V2 directory: ${DEPTH_MODEL}" >&2
    exit 1
fi

if [[ ! -f "${DEPTH_MODEL}/config.json" ]]; then
    echo "Missing ${DEPTH_MODEL}/config.json" >&2
    exit 1
fi

if [[ ! -f "${DEPTH_MODEL}/preprocessor_config.json" ]]; then
    echo "Missing ${DEPTH_MODEL}/preprocessor_config.json" >&2
    exit 1
fi

if [[ ! -f "${DEPTH_MODEL}/model.safetensors" ]]; then
    echo "Missing ${DEPTH_MODEL}/model.safetensors" >&2
    exit 1
fi

if [[ ! -f "${RDAH_CHECKPOINT}" ]]; then
    echo "Missing RDAH-Net checkpoint: ${RDAH_CHECKPOINT}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" \
models/third_partys/RDAH-Net-main/generate_heightmaps.py \
    --dataset-roots \
        "${DATA_ROOT}/LEVIR-CD-foggy" \
        "${DATA_ROOT}/SYSU-CD-foggy" \
        "${DATA_ROOT}/LEVIR-CD+foggy" \
    --rdah-checkpoint "${RDAH_CHECKPOINT}" \
    --depth-model "${DEPTH_MODEL}" \
    --batch-size 1 \
    --image-size 256 \
    --device cuda:0
