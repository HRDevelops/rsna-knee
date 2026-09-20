#!/bin/bash
# exp016 pre-flight (scripts/smoke_exp016.py) — the exp016 5-fold arrays are submitted with
# --dependency=afterok on this, so a failed fold-safety/plumbing check stops them from ever starting.
# DISK_PROJECTION_GB: 0.5
#SBATCH --job-name=exp016_smoke
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=48GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
echo "[gpu-binding] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader || true

python scripts/smoke_exp016.py --config configs/exp016a_dinov2b_slot_v3labels_auxrep_w010.yaml
python scripts/smoke_exp016.py --config configs/exp016b_dinov2b_slot_v3labels_auxrep_w050.yaml --fold 3
