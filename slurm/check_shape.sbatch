#!/bin/bash
# Pre-launch trainability check (scripts/check_train_shape.py): real training shape (8 studies x 12 windows,
# bf16, grad checkpointing, backward + step) and real eval shape (2 x 54 windows) for each candidate
# backbone, with exp018's CoAtNet-2 as the same-hardware reference row. Arguments are forwarded.
#   python scripts/safe_sbatch.py -- slurm/check_shape.sbatch --spec <arch:res[:img_size]> ...
# DISK_PROJECTION_GB: 0.2
#SBATCH --job-name=check_shape
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
python scripts/check_train_shape.py "$@"
