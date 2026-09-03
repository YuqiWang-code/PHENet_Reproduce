#!/usr/bin/env bash
# Zero-shot generalization:
# LEVIR-CD-foggy best_F1=0.7093 -> SYSU-CD-foggy/test
#
# Foreground:
#   cd /home/yqwang/project/PHENet && bash train_scripts/test_sysu_generalization.sh
#
# Background:
#   cd /home/yqwang/project/PHENet && \
#   nohup bash train_scripts/test_sysu_generalization.sh \
#   > /tmp/phenet_sysu_generalization.launch.log 2>&1 &
#
# Monitor:
#   tail -f /tmp/phenet_sysu_generalization.launch.log

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
unset LD_LIBRARY_PATH
unset CUDA_LAUNCH_BLOCKING || true
unset PYTORCH_CUDA_ALLOC_CONF || true

PROJECT_ROOT="/home/yqwang/project/PHENet"
PYTHON_BIN="/home/yqwang/miniconda3/envs/phenet/bin/python"

CHECKPOINT="/storage/yqwang/PHENet/saved_models/Run1/LEVIR-CD-foggy/best_F1=0.7093.pth"
DATA_ROOT="/storage/BCD-foggy/SYSU-CD-foggy"
EVALUATOR="${PROJECT_ROOT}/models/evaluate_generalization.py"
MOBILENET_CHECKPOINT="${PROJECT_ROOT}/models/modeling/backbone/mobilenet_v2-6a65762b.pth"

RESULT_ROOT="/storage/yqwang/PHENet/generalization/Run1/LEVIR-CD-foggy_best_F1_0.7093"
OUTPUT_DIR="${RESULT_ROOT}/SYSU-CD-foggy"
OUTPUT_LOG="${OUTPUT_DIR}/test.log"

EXPECTED_PAIRS=4000
BATCH_SIZE=8
WORKERS=8

count_images() {
    local directory="$1"
    find "${directory}" -maxdepth 1 -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o \
           -iname '*.tif' -o -iname '*.tiff' \) \
        -printf '.' | wc -c
}

require_image_count() {
    local directory="$1"
    local expected="$2"

    if [[ ! -d "${directory}" ]]; then
        echo "Missing directory: ${directory}" >&2
        exit 1
    fi

    local actual
    actual="$(count_images "${directory}")"

    if [[ "${actual}" -ne "${expected}" ]]; then
        echo "Image count mismatch: ${directory}: expected=${expected}, actual=${actual}" >&2
        exit 1
    fi

    echo "[OK] ${directory}: ${actual} images"
}

echo "=== PHENet zero-shot generalization: LEVIR -> SYSU ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data root: ${DATA_ROOT}"
echo "Output log: ${OUTPUT_LOG}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${EVALUATOR}" ]]; then
    echo "Missing evaluator: ${EVALUATOR}" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing PHENet checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -f "${MOBILENET_CHECKPOINT}" ]]; then
    echo "Missing MobileNetV2 checkpoint required during PHENet construction: ${MOBILENET_CHECKPOINT}" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "=== Physical GPU 1 status before evaluation ==="
    nvidia-smi -i 1
    echo
else
    echo "nvidia-smi not found." >&2
    exit 1
fi

echo "=== Checking SYSU-CD-foggy/test ==="
require_image_count "${DATA_ROOT}/test/A" "${EXPECTED_PAIRS}"
require_image_count "${DATA_ROOT}/test/B" "${EXPECTED_PAIRS}"
require_image_count "${DATA_ROOT}/test/GT" "${EXPECTED_PAIRS}"
require_image_count "${DATA_ROOT}/test/A_heightmap" "${EXPECTED_PAIRS}"
require_image_count "${DATA_ROOT}/test/B_heightmap" "${EXPECTED_PAIRS}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" \
    "${EVALUATOR}" \
    --data-root "${DATA_ROOT}" \
    --dataset-name "SYSU-CD-foggy" \
    --checkpoint "${CHECKPOINT}" \
    --split test \
    --label-dir GT \
    --batch-size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --crop-size 256 \
    --device cuda:0 \
    --output-log "${OUTPUT_LOG}"

echo
echo "[PASS] SYSU-CD-foggy generalization evaluation completed."
echo "Result log: ${OUTPUT_LOG}"
