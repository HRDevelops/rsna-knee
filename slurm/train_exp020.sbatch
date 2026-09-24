#!/bin/bash
# exp020 (Phase 5): exp018's recipe plus model EMA (two decays tracked). FOLD 0 ONLY unless the decision
# gate promotes it. Same CoAtNet-2 backbone as exp018, so per-checkpoint sizes are exp018's; the EMA adds
# two extra ~0.3GB best-EMA checkpoints on top of exp018's ~2.5GB.
# DISK_PROJECTION_GB: 4
#SBATCH --job-name=exp020
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
python scripts/train_wmil.py --config configs/exp020_coatnet2_wmil_ema.yaml --fold "$FOLD" \
    --out "$RSNA_ROOT/artifacts" --resume auto --max-hours 5.5
