#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
unset LD_LIBRARY_PATH
unset CUDA_LAUNCH_BLOCKING || true
unset PYTORCH_CUDA_ALLOC_CONF || true

PROJECT_ROOT="/home/yqwang/project/PHENet"
PYTHON_BIN="/home/yqwang/miniconda3/envs/phenet/bin/python"
CHECKPOINT="/storage/yqwang/PHENet/saved_models/Run1/SYSU-CD-foggy/best_F1=0.7192.pth"
HH_ROOT="/storage/BCD-foggy/HH"
EVALUATOR="${PROJECT_ROOT}/models/evaluate_generalization.py"
MOBILENET_CHECKPOINT="${PROJECT_ROOT}/models/modeling/backbone/mobilenet_v2-6a65762b.pth"
RESULT_ROOT="/storage/yqwang/PHENet/generalization/Run1/SYSU-CD-foggy_best_F1_0.7192/HH"
SUMMARY_FILE="${RESULT_ROOT}/summary.tsv"
SUBSETS=("HH-fog-snow" "HH-normal-fog" "HH-normal-snow")
EXPECTED_PAIRS=152
BATCH_SIZE=8
WORKERS=8

count_images() {
    local directory="$1"
    find "${directory}" -maxdepth 1 -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.tif' -o -iname '*.tiff' \) \
        -printf '.' | wc -c
}

require_image_count() {
    local directory="$1"
    local expected="$2"
    [[ -d "${directory}" ]] || { echo "Missing directory: ${directory}" >&2; exit 1; }
    local actual
    actual="$(count_images "${directory}")"
    [[ "${actual}" -eq "${expected}" ]] || {
        echo "Image count mismatch: ${directory}: expected=${expected}, actual=${actual}" >&2
        exit 1
    }
    echo "[OK] ${directory}: ${actual} images"
}

echo "=== PHENet zero-shot cross-domain generalization: SYSU-CD-foggy -> HH ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Result root: ${RESULT_ROOT}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${EVALUATOR}" ]] || { echo "Missing evaluator: ${EVALUATOR}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing PHENet checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${MOBILENET_CHECKPOINT}" ]] || { echo "Missing MobileNetV2 checkpoint: ${MOBILENET_CHECKPOINT}" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found." >&2; exit 1; }

echo
echo "=== Physical GPU 1 status before evaluation ==="
nvidia-smi -i 1
echo

echo "=== Checking all HH test subsets before inference ==="
for subset in "${SUBSETS[@]}"; do
    data_root="${HH_ROOT}/${subset}"
    echo "--- Precheck ${subset} ---"
    require_image_count "${data_root}/test/A" "${EXPECTED_PAIRS}"
    require_image_count "${data_root}/test/B" "${EXPECTED_PAIRS}"
    require_image_count "${data_root}/test/label" "${EXPECTED_PAIRS}"
    require_image_count "${data_root}/test/A_heightmap" "${EXPECTED_PAIRS}"
    require_image_count "${data_root}/test/B_heightmap" "${EXPECTED_PAIRS}"
done

mkdir -p "${RESULT_ROOT}"
printf 'RESULT\tDataset\tSplit\tPairs\tTN\tFP\tFN\tTP\tRecall\tPrecision\tOA\tF1\tIoU\tKappa\n' > "${SUMMARY_FILE}"

cd "${PROJECT_ROOT}"
for subset in "${SUBSETS[@]}"; do
    data_root="${HH_ROOT}/${subset}"
    output_dir="${RESULT_ROOT}/${subset}"
    output_log="${output_dir}/test.log"
    mkdir -p "${output_dir}"

    echo
    echo "=== Evaluating SYSU-CD-foggy -> ${subset} ==="
    "${PYTHON_BIN}" "${EVALUATOR}" \
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

    awk -F '\t' '$1 == "RESULT" && $2 != "Dataset" {print}' "${output_log}" >> "${SUMMARY_FILE}"
    echo "[PASS] SYSU-CD-foggy -> ${subset}: ${output_log}"
done

echo
echo "[PASS] SYSU-CD-foggy -> all three HH subsets completed."
echo "Summary: ${SUMMARY_FILE}"
