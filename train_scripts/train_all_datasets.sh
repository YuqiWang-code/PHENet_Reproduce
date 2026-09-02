#!/usr/bin/env bash

# Run the remaining two PHENet foggy-dataset experiments sequentially:
#
#   1. SYSU-CD-foggy
#   2. LEVIR-CD+foggy
#
# Both child scripts internally expose only server physical GPU 1.
# Physical GPU 1 becomes logical cuda:0 inside each Python process.
#
# Foreground command:
# cd /home/yqwang/project/PHENet && bash train_scripts/train_all_datasets.sh
#
# Recommended screen command:
# screen -dmS phenet_remaining bash -lc \
#   'cd /home/yqwang/project/PHENet && bash train_scripts/train_all_datasets.sh > /tmp/phenet_train_remaining.log 2>&1'
#
# Attach to screen:
# screen -r phenet_remaining
#
# Launcher log:
# tail -f /tmp/phenet_train_remaining.log

set -euo pipefail

# Formal experiments are restricted to server physical GPU 1.
# It becomes logical cuda:0 inside Python.
export CUDA_VISIBLE_DEVICES=1

# Prevent CUDA libraries from another environment from contaminating
# the existing PHENet environment.
unset LD_LIBRARY_PATH

PROJECT_ROOT="/home/yqwang/project/PHENet"

cd "${PROJECT_ROOT}"

TRAINING_SCRIPTS=(
    "train_scripts/train_sysu_cd_foggy.sh"
    "train_scripts/train_levir_cd_plus_foggy.sh"
)

for training_script in "${TRAINING_SCRIPTS[@]}"; do
    echo "======================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${training_script}"
    echo "======================================================================"

    bash "${training_script}"

    echo "======================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${training_script}"
    echo "======================================================================"
done

echo "======================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remaining PHENet dataset training runs completed successfully."
echo "======================================================================"