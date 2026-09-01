#!/usr/bin/env bash
# Server foreground command (runs all three datasets sequentially in this terminal):
# cd /home/yqwang/project/PHENet && CUDA_VISIBLE_DEVICES=1 bash train_scripts/train_all_datasets.sh
# Server background command:
# cd /home/yqwang/project/PHENet && CUDA_VISIBLE_DEVICES=1 nohup bash train_scripts/train_all_datasets.sh > /tmp/phenet_train_all_datasets.log 2>&1 &
# Background launcher log monitor command:
# tail -f /tmp/phenet_train_all_datasets.log

set -euo pipefail

# Expose only server physical GPU 1 for every child training script.
export CUDA_VISIBLE_DEVICES=1

PROJECT_ROOT="/home/yqwang/project/PHENet"
cd "${PROJECT_ROOT}"

TRAINING_SCRIPTS=(
  "train_scripts/train_levir_cd_foggy.sh"
  "train_scripts/train_sysu_cd_foggy.sh"
  "train_scripts/train_levir_cd_plus_foggy.sh"
)

for training_script in "${TRAINING_SCRIPTS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${training_script}"
  bash "${training_script}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${training_script}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All PHENet dataset training runs completed successfully."
