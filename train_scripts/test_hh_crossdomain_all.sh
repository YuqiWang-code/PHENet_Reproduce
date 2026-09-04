#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
unset LD_LIBRARY_PATH
unset CUDA_LAUNCH_BLOCKING || true
unset PYTORCH_CUDA_ALLOC_CONF || true

PROJECT_ROOT="/home/yqwang/project/PHENet"
RESULT_ROOT="/storage/yqwang/PHENet/generalization/Run1"
LEVIR_PLUS_SCRIPT="${PROJECT_ROOT}/train_scripts/test_levir_plus_hh_generalization.sh"
SYSU_SCRIPT="${PROJECT_ROOT}/train_scripts/test_sysu_hh_generalization.sh"
LEVIR_PLUS_SUMMARY="${RESULT_ROOT}/LEVIR-CD+foggy_best_F1_0.6525/HH/summary.tsv"
SYSU_SUMMARY="${RESULT_ROOT}/SYSU-CD-foggy_best_F1_0.7192/HH/summary.tsv"
COMBINED_SUMMARY="${RESULT_ROOT}/HH_crossdomain_summary.tsv"

echo "=== PHENet six-test HH cross-domain generalization ==="
echo "Order: LEVIR-CD+foggy -> HH (3), then SYSU-CD-foggy -> HH (3)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

[[ -f "${LEVIR_PLUS_SCRIPT}" ]] || { echo "Missing script: ${LEVIR_PLUS_SCRIPT}" >&2; exit 1; }
[[ -f "${SYSU_SCRIPT}" ]] || { echo "Missing script: ${SYSU_SCRIPT}" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found." >&2; exit 1; }

echo
echo "=== Physical GPU 1 status before all evaluations ==="
nvidia-smi -i 1
echo

cd "${PROJECT_ROOT}"

echo "=== Step 1/2: LEVIR-CD+foggy -> HH ==="
bash "${LEVIR_PLUS_SCRIPT}"

echo
echo "=== Step 2/2: SYSU-CD-foggy -> HH ==="
bash "${SYSU_SCRIPT}"

[[ -f "${LEVIR_PLUS_SUMMARY}" ]] || { echo "Missing summary: ${LEVIR_PLUS_SUMMARY}" >&2; exit 1; }
[[ -f "${SYSU_SUMMARY}" ]] || { echo "Missing summary: ${SYSU_SUMMARY}" >&2; exit 1; }

printf 'Source\tTarget\tSplit\tPairs\tTN\tFP\tFN\tTP\tRecall\tPrecision\tOA\tF1\tIoU\tKappa\n' > "${COMBINED_SUMMARY}"

awk -F '\t' -v source="LEVIR-CD+foggy" '
BEGIN {OFS="\t"}
$1 == "RESULT" && $2 != "Dataset" {
    print source,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14
}' "${LEVIR_PLUS_SUMMARY}" >> "${COMBINED_SUMMARY}"

awk -F '\t' -v source="SYSU-CD-foggy" '
BEGIN {OFS="\t"}
$1 == "RESULT" && $2 != "Dataset" {
    print source,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14
}' "${SYSU_SUMMARY}" >> "${COMBINED_SUMMARY}"

echo
echo "[PASS] All six HH cross-domain evaluations completed."
echo "Combined summary: ${COMBINED_SUMMARY}"
