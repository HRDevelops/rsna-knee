#!/bin/bash
# exp021: exp018 retrained with seed 43, one fold per array task (submit with --array=0-4).
# Same backbone and checkpoint sizes as exp018 (train_wmil.sbatch): ~2.5GB per fold. safe_sbatch.py uses a
# DISK_PROJECTION_GB header as the TOTAL for the whole submission (it is not multiplied by the array size),
# so this is 5 folds x 3GB. Launch: python scripts/safe_sbatch.py -- --array=0-4 slurm/train_exp021.sbatch
# (sbatch options BEFORE the script, after `--`; placed after the script they reach it as $1 instead).
# DISK_PROJECTION_GB: 15
#SBATCH --job-name=exp021
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=96GB
#SBATCH --exclude=g005,g006

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
echo "[gpu-binding] host=$(hostname) job=${SLURM_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader || true

FOLD=${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}
python scripts/train_wmil.py --config configs/exp021_coatnet2_wmil_seed43.yaml --fold "$FOLD" \
    --out "$RSNA_ROOT/artifacts" --resume auto --max-hours 5.5
