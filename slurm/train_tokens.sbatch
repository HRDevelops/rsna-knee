#!/bin/bash
# exp018 (adjacent-slice Window-MIL, CoAtNet-2): one fold per array task. Same submission path as train.sbatch
# (via scripts/safe_sbatch.py). Per fold: checkpoint.pt (~1.2GB: model+AdamW) + 3 top-epoch files (~0.3GB each,
# deleted after averaging) + best.pt (0.3GB) -> ~2.5GB; projection is per-array-task-count x 3GB.
# DISK_PROJECTION_GB: 3
#SBATCH --job-name=p0c
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

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1           # backbone weights are in the shared HF cache (downloaded on the login node)
echo "[gpu-binding] host=$(hostname) job=${SLURM_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader || true

CONFIG=${1:?config required}
FOLD=${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}
python scripts/train_tokens.py --config "$CONFIG" --fold "$FOLD" --out "$RSNA_ROOT/artifacts" \
    --max-hours 5.5
