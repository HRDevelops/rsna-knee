#!/bin/bash
#SBATCH --job-name=rsna_accept
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:45:00
#SBATCH --mem=64GB
#SBATCH --array=0-9

# Stability-bundle acceptance sweep (2026-09-07 strategy review, Priority 1 step 3):
# 2 seeds x 5 folds = 10 runs, each trained for a handful of epochs to check the loss escapes the
# collapse plateau by epoch 5. Not a full training run — --max-hours bounds it to ~8-9 epochs.
set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

SEEDS=(42 123)
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED=${SEEDS[$SEED_IDX]}

python scripts/train.py --config configs/exp001_dinov2s_slot.yaml --fold "$FOLD" \
    --out "$RSNA_ROOT/artifacts/acceptance_sweep" --resume none --max-hours 1.6 \
    --seed-override "$SEED"
