#!/bin/bash
# Dense-window inference test (scripts/dense_window_eval.py): array task 0 = standard layout (the control),
# task 1 = dense layout (x1.6 slices -> ~92 windows). Existing exp018 checkpoints, no training.
#   python scripts/safe_sbatch.py -- --array=0-1 slurm/dense_window_eval.sbatch
# DISK_PROJECTION_GB: 0.2
#SBATCH --job-name=dense_eval
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=96GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1   # decode threads do the parallelism; keep BLAS from oversubscribing the 16 CPUs
LAYOUTS=(standard dense)
python scripts/dense_window_eval.py --layout "${LAYOUTS[${SLURM_ARRAY_TASK_ID:-0}]}" --scale 1.6 --workers 16
