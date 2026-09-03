#!/usr/bin/env bash
# Zero-shot generalization:
# LEVIR-CD-foggy best_F1=0.7093 -> HH adverse-weather test subsets
#
# The three HH subsets are evaluated strictly serially.
#
# Foreground:
#   cd /home/yqwang/project/PHENet && bash train_scripts/test_hh_generalization.sh
#
# Background:
#   cd /home/yqwang/project/PHENet && \
#   nohup bash train_scripts/test_hh_generalization.sh \
#   > /tmp/phenet_hh_generalization.launch.log 2>&1 &
#
# Monitor:
#   tail -f /tmp/phenet_hh_generalization.launch.log

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
unset LD_LIBRARY_PATH
unset CUDA_LAUNCH_BLOCKING || true
unset PYTORCH_CUDA_ALLOC_CONF || true

PROJECT_ROOT="/home/yqwang/project/PHENet"
PYTHON_BIN="/home/yqwang/miniconda3/envs/phenet/bin/python"

CHECKPOINT="/storage/yqwang/PHENet/saved_models/Run1/LEVIR-CD-foggy/best_F1=0.7093.pth"
HH_ROOT="/storage/BCD-foggy/HH"
EVALUATOR="${PROJECT_ROOT}/models/evaluate_generalization.py"
MOBILENET_CHECKPOINT="${PROJECT_ROOT}/models/modeling/backbone/mobilenet_v2-6a65762b.pth"

RESULT_ROOT="/storage/yqwang/PHENet/generalization/Run1/LEVIR-CD-foggy_best_F1_0.7093/HH"
SUMMARY_FILE="${RESULT_ROOT}/summary.tsv"

SUBSETS=(
    "HH-fog-snow"
    "HH-normal-fog"
    "HH-normal-snow"
)

EXPECTED_PAIRS=152
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

echo "=== PHENet zero-shot generalization: LEVIR -> HH ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Checkpoint: ${CHECKPOINT}"
echo "HH root: ${HH_ROOT}"
echo "Result root: ${RESULT_ROOT}"

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

echo "=== Checking all HH test subsets before inference ==="
for subset in "${SUBSETS[@]}"; do
    data_root="${HH_ROOT}/${subset}"
    echo
    echo "--- Precheck ${subset} ---"
    require_image_count "${data_root}/test/A" "${EXPECTED_PAIRS}"
    require_image_count "${data_root}/test/B" "${EXPECTED_PAIRS}"
    require_image_count "${data_root}/test/label" "${EXPECTED_PAIRS}"
    require_image_count "${data_root}/test/A_heightmap" "${EXPECTED_PAIRS}"
    require_image_count "${data_root}/test/B_heightmap" "${EXPECTED_PAIRS}"
done

mkdir -p "${RESULT_ROOT}"
printf 'RESULT\tDataset\tSplit\tPairs\tTN\tFP\tFN\tTP\tRecall\tPrecision\tOA\tF1\tIoU\tKappa\n' \
    > "${SUMMARY_FILE}"

cd "${PROJECT_ROOT}"

for subset in "${SUBSETS[@]}"; do
    data_root="${HH_ROOT}/${subset}"
    output_dir="${RESULT_ROOT}/${subset}"
    output_log="${output_dir}/test.log"

    mkdir -p "${output_dir}"

    echo
    echo "=== Evaluating ${subset} ==="

    "${PYTHON_BIN}" \
        "${EVALUATOR}" \
        --data-root "${data_root}" \
        --dataset-name "${subset}" \
        --checkpoint "${CHECKPOINT}" \
        --split test \
        --label-dir label \
        --batch-size "${BATCH_SIZE}" \
        --workers "${WORKERS}" \
        --crop-size 256 \
        --device cuda:0 \
        --output-log "${output_log}"

    awk -F '\t' '$1 == "RESULT" && $2 != "Dataset" {print}' "${output_log}" \
        >> "${SUMMARY_FILE}"

    echo "[PASS] ${subset}: ${output_log}"
done

echo
echo "[PASS] All three HH generalization evaluations completed."
echo "HH summary: ${SUMMARY_FILE}"
