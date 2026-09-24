#!/bin/bash
# exp019 (Phase 5 capacity test): Window-MIL with CoAtNet-3 (coatnet_3_rw_224.sw_in12k @ img_size 384).
# FOLD 0 ONLY unless the decision gate promotes it.
#
# Disk: this backbone is 164M params (2.2x exp018's 73M), so every checkpoint is 2.2x exp018's too:
# checkpoint.pt (model + AdamW's 2 moments ~= 3 copies) ~2.0GB + 3 top-epoch files ~0.66GB each ~2.0GB
# + best.pt 0.66GB ~= 4.6GB. Declared 7GB for headroom -- exp018's train_wmil.sbatch declares 3GB, which
# is sized for CoAtNet-2 and would under-report this job to the disk guard.
# DISK_PROJECTION_GB: 7
#SBATCH --job-name=exp019
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
export HF_HUB_OFFLINE=1   # coatnet_3_rw_224.sw_in12k pre-downloaded to the shared HF cache on the login node
echo "[gpu-binding] host=$(hostname) job=${SLURM_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader || true

FOLD=${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}
python scripts/train_wmil.py --config configs/exp019_coatnet3_wmil.yaml --fold "$FOLD" \
    --out "$RSNA_ROOT/artifacts" --resume auto --max-hours 5.5
