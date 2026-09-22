#!/bin/bash
# Probe #4 blend rehearsal (correctness + timing) on H100.
# DISK_PROJECTION_GB: 0.5
#SBATCH --job-name=blend018_rehearse
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=64GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
echo "[gpu-binding] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python scripts/rehearse_blend018.py --n 40 --weight-exp018 0.75 --decode-workers 6 \
    --out "$RSNA_ROOT/artifacts/rehearsal_blend018.csv"
