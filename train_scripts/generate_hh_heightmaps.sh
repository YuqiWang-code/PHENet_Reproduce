#!/usr/bin/env bash
# HH height-map generation (physical GPU 1 -> logical cuda:0).
#
# Foreground:
#   cd /home/yqwang/project/PHENet && bash train_scripts/generate_hh_heightmaps.sh
#
# Background:
#   cd /home/yqwang/project/PHENet && \
#   nohup bash train_scripts/generate_hh_heightmaps.sh \
#   > /tmp/phenet_generate_hh_heightmaps.log 2>&1 &
#
# Monitor:
#   tail -f /tmp/phenet_generate_hh_heightmaps.log

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
unset LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
unset CUDA_LAUNCH_BLOCKING || true
unset PYTORCH_CUDA_ALLOC_CONF || true

PROJECT_ROOT="/home/yqwang/project/PHENet"
DATA_ROOT="/storage/BCD-foggy/HH"
PYTHON_BIN="/home/yqwang/miniconda3/envs/phenet/bin/python"

RDAH_CHECKPOINT="${PROJECT_ROOT}/pre_checkpoint/checkpoints-track1/104best_model.pth"
DEPTH_MODEL="${PROJECT_ROOT}/models/third_partys/Depth-Anything-V2-Large-hf"
GENERATOR="${PROJECT_ROOT}/models/third_partys/RDAH-Net-main/generate_heightmaps.py"

HH_ROOTS=(
    "${DATA_ROOT}/HH-fog-snow"
    "${DATA_ROOT}/HH-normal-fog"
    "${DATA_ROOT}/HH-normal-snow"
)

EXPECTED_PAIRS=152

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

echo "=== PHENet HH height-map generation ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${GENERATOR}" ]]; then
    echo "Missing height-map generator: ${GENERATOR}" >&2
    exit 1
fi

if [[ ! -f "${RDAH_CHECKPOINT}" ]]; then
    echo "Missing RDAH-Net checkpoint: ${RDAH_CHECKPOINT}" >&2
    exit 1
fi

for required_file in config.json preprocessor_config.json model.safetensors; do
    if [[ ! -f "${DEPTH_MODEL}/${required_file}" ]]; then
        echo "Missing Depth Anything V2 file: ${DEPTH_MODEL}/${required_file}" >&2
        exit 1
    fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "=== Physical GPU 1 status before generation ==="
    nvidia-smi -i 1
    echo
else
    echo "nvidia-smi not found." >&2
    exit 1
fi

echo "=== Checking HH source data ==="
for root in "${HH_ROOTS[@]}"; do
    require_image_count "${root}/test/A" "${EXPECTED_PAIRS}"
    require_image_count "${root}/test/B" "${EXPECTED_PAIRS}"
    require_image_count "${root}/test/label" "${EXPECTED_PAIRS}"
done

echo
echo "=== Generating missing HH height maps ==="
echo "Existing output files are skipped; --overwrite is intentionally NOT used."

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" \
    "${GENERATOR}" \
    --dataset-roots \
        "${HH_ROOTS[0]}" \
        "${HH_ROOTS[1]}" \
        "${HH_ROOTS[2]}" \
    --rdah-checkpoint "${RDAH_CHECKPOINT}" \
    --depth-model "${DEPTH_MODEL}" \
    --batch-size 1 \
    --image-size 256 \
    --device cuda:0

echo
echo "=== Verifying HH height-map counts ==="
for root in "${HH_ROOTS[@]}"; do
    require_image_count "${root}/test/A_heightmap" "${EXPECTED_PAIRS}"
    require_image_count "${root}/test/B_heightmap" "${EXPECTED_PAIRS}"
done

echo
echo "[PASS] HH height-map generation/count verification completed."
