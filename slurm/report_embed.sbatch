#!/bin/bash
# exp016: embed all 4,407 reports with frozen BAAI/bge-m3 and write per-fold, train-split-only
# auxiliary targets (scripts/build_report_embeddings.py). Short GPU job (~few minutes of compute; the
# rest is the one-time ~2.3GB model download into ~/.cache/huggingface, compute nodes have internet).
# DISK_PROJECTION_GB: 3
#SBATCH --job-name=exp016_report_embed
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
echo "[gpu-binding] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader || true

python scripts/build_report_embeddings.py --device cuda --diagnostics
