#!/usr/bin/env bash
# Complete zero-shot generalization entry point:
#   1. SYSU-CD-foggy/test
#   2. HH-fog-snow/test
#   3. HH-normal-fog/test
#   4. HH-normal-snow/test
#
# All jobs are serial and forced to physical GPU 1.
#
# Foreground:
#   cd /home/yqwang/project/PHENet && bash train_scripts/test_generalization_all.sh
#
# Background:
#   cd /home/yqwang/project/PHENet && \
#   nohup bash train_scripts/test_generalization_all.sh \
#   > /tmp/phenet_generalization.launch.log 2>&1 &
#
# Monitor launcher:
#   tail -f /tmp/phenet_generalization.launch.log

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
unset LD_LIBRARY_PATH
unset CUDA_LAUNCH_BLOCKING || true
unset PYTORCH_CUDA_ALLOC_CONF || true

PROJECT_ROOT="/home/yqwang/project/PHENet"
RESULT_ROOT="/storage/yqwang/PHENet/generalization/Run1/LEVIR-CD-foggy_best_F1_0.7093"
SYSU_SCRIPT="${PROJECT_ROOT}/train_scripts/test_sysu_generalization.sh"
HH_SCRIPT="${PROJECT_ROOT}/train_scripts/test_hh_generalization.sh"
SUMMARY_FILE="${RESULT_ROOT}/summary.tsv"

echo "=== PHENet complete zero-shot generalization evaluation ==="
echo "Order: SYSU-CD-foggy -> HH-fog-snow -> HH-normal-fog -> HH-normal-snow"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [[ ! -f "${SYSU_SCRIPT}" ]]; then
    echo "Missing script: ${SYSU_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${HH_SCRIPT}" ]]; then
    echo "Missing script: ${HH_SCRIPT}" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "=== Physical GPU 1 status before all evaluations ==="
    nvidia-smi -i 1
    echo
else
    echo "nvidia-smi not found." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

echo "=== Step 1/2: SYSU-CD-foggy ==="
bash "${SYSU_SCRIPT}"

echo
echo "=== Step 2/2: HH adverse-weather subsets ==="
bash "${HH_SCRIPT}"

SYSU_LOG="${RESULT_ROOT}/SYSU-CD-foggy/test.log"
HH_SUMMARY="${RESULT_ROOT}/HH/summary.tsv"

if [[ ! -f "${SYSU_LOG}" ]]; then
    echo "Missing SYSU result log after evaluation: ${SYSU_LOG}" >&2
    exit 1
fi

if [[ ! -f "${HH_SUMMARY}" ]]; then
    echo "Missing HH summary after evaluation: ${HH_SUMMARY}" >&2
    exit 1
fi

mkdir -p "${RESULT_ROOT}"
printf 'RESULT\tDataset\tSplit\tPairs\tTN\tFP\tFN\tTP\tRecall\tPrecision\tOA\tF1\tIoU\tKappa\n' \
    > "${SUMMARY_FILE}"

awk -F '\t' '$1 == "RESULT" && $2 != "Dataset" {print}' "${SYSU_LOG}" \
    >> "${SUMMARY_FILE}"

awk -F '\t' '$1 == "RESULT" && $2 != "Dataset" {print}' "${HH_SUMMARY}" \
    >> "${SUMMARY_FILE}"

echo
echo "[PASS] Complete generalization evaluation finished."
echo "Combined summary: ${SUMMARY_FILE}"
