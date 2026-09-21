#!/bin/bash
# exp018 wiring check: 2 epochs on ~40 studies (train_wmil.py --smoke); the fold-0 job depends on it (afterok).
# DISK_PROJECTION_GB: 0.5
#SBATCH --job-name=exp018_smoke
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mem=64GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[gpu-binding] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
python scripts/train_wmil.py --config configs/exp018_coatnet2_wmil_v3labels.yaml --fold 0 \
    --out "$RSNA_ROOT/artifacts/_smoke_exp018" --smoke --workers 4
