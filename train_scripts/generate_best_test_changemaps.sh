#!/usr/bin/env bash
# Generate binary PHENet test changemaps for the three Run1 best checkpoints.
#
# Outputs:
#   /storage/BCD-foggy/LEVIR-CD-foggy/test/result_PHENet/
#   /storage/BCD-foggy/LEVIR-CD+foggy/test/result_PHENet/
#   /storage/BCD-foggy/SYSU-CD-foggy/test/result_PHENet/
#
# Background:
#   cd /home/yqwang/project/PHENet && \
#   nohup bash train_scripts/generate_best_test_changemaps.sh \
#   > /tmp/phenet_best_test_changemaps.launch.log 2>&1 &

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
unset LD_LIBRARY_PATH
unset CUDA_LAUNCH_BLOCKING || true
unset PYTORCH_CUDA_ALLOC_CONF || true

PROJECT_ROOT="/home/yqwang/project/PHENet"
PYTHON_BIN="/home/yqwang/miniconda3/envs/phenet/bin/python"
INFER_SCRIPT="${PROJECT_ROOT}/models/infer_changemaps.py"
MOBILENET_CHECKPOINT="${PROJECT_ROOT}/models/modeling/backbone/mobilenet_v2-6a65762b.pth"

BATCH_SIZE=8
WORKERS=8

count_png() {
    local directory="$1"
    find "${directory}" -maxdepth 1 -type f -iname '*.png' -printf '.' | wc -c
}

require_png_count() {
    local directory="$1"
    local expected="$2"

    if [[ ! -d "${directory}" ]]; then
        echo "Missing directory: ${directory}" >&2
        exit 1
    fi

    local actual
    actual="$(count_png "${directory}")"

    if [[ "${actual}" -ne "${expected}" ]]; then
        echo "PNG count mismatch: ${directory}: expected=${expected}, actual=${actual}" >&2
        exit 1
    fi

    echo "[OK] ${directory}: ${actual} PNG files"
}

run_dataset() {
    local dataset_name="$1"
    local data_root="$2"
    local label_dir="$3"
    local checkpoint="$4"
    local expected_pairs="$5"

    local split_root="${data_root}/test"
    local output_dir="${split_root}/result_PHENet"

    echo
    echo "================================================================"
    echo "Dataset: ${dataset_name}"
    echo "Checkpoint: ${checkpoint}"
    echo "Output: ${output_dir}"
    echo "================================================================"

    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing PHENet checkpoint: ${checkpoint}" >&2
        exit 1
    fi

    require_png_count "${split_root}/A" "${expected_pairs}"
    require_png_count "${split_root}/B" "${expected_pairs}"
    require_png_count "${split_root}/${label_dir}" "${expected_pairs}"
    require_png_count "${split_root}/A_heightmap" "${expected_pairs}"
    require_png_count "${split_root}/B_heightmap" "${expected_pairs}"

    mkdir -p "${output_dir}"

    "${PYTHON_BIN}" \
        "${INFER_SCRIPT}" \
        --data-root "${data_root}" \
        --dataset-name "${dataset_name}" \
        --checkpoint "${checkpoint}" \
        --split test \
        --label-dir "${label_dir}" \
        --output-dir "${output_dir}" \
        --batch-size "${BATCH_SIZE}" \
        --workers "${WORKERS}" \
        --crop-size 256 \
        --device cuda:0 \
        --overwrite

    require_png_count "${output_dir}" "${expected_pairs}"
    echo "[PASS] ${dataset_name}: ${expected_pairs} change maps saved."
}

echo "=== PHENet Run1 best-checkpoint test changemap generation ==="

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${INFER_SCRIPT}" ]]; then
    echo "Missing inference script: ${INFER_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${MOBILENET_CHECKPOINT}" ]]; then
    echo "Missing MobileNetV2 checkpoint: ${MOBILENET_CHECKPOINT}" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found." >&2
    exit 1
fi

echo
echo "=== Physical GPU 1 status before inference ==="
nvidia-smi -i 1
echo

cd "${PROJECT_ROOT}"

run_dataset \
    "LEVIR-CD-foggy" \
    "/storage/BCD-foggy/LEVIR-CD-foggy" \
    "label" \
    "/storage/yqwang/PHENet/saved_models/Run1/LEVIR-CD-foggy/best_F1=0.7093.pth" \
    "2048"

run_dataset \
    "LEVIR-CD+foggy" \
    "/storage/BCD-foggy/LEVIR-CD+foggy" \
    "GT" \
    "/storage/yqwang/PHENet/saved_models/Run1/LEVIR-CD+foggy/best_F1=0.6525.pth" \
    "5568"

run_dataset \
    "SYSU-CD-foggy" \
    "/storage/BCD-foggy/SYSU-CD-foggy" \
    "GT" \
    "/storage/yqwang/PHENet/saved_models/Run1/SYSU-CD-foggy/best_F1=0.7192.pth" \
    "4000"

echo
echo "[PASS] All three test-set PHENet changemap generations completed."
echo "Result directories:"
echo "  /storage/BCD-foggy/LEVIR-CD-foggy/test/result_PHENet/"
echo "  /storage/BCD-foggy/LEVIR-CD+foggy/test/result_PHENet/"
echo "  /storage/BCD-foggy/SYSU-CD-foggy/test/result_PHENet/"
