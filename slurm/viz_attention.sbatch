#!/bin/bash
# Window-MIL attention extraction for the gold-58 figures (scripts/viz_attention.py): inference only, existing
# exp018 + exp021 checkpoints, no training and no scored artifact touched.
#   python scripts/safe_sbatch.py -- slurm/viz_attention.sbatch
# DISK_PROJECTION_GB: 0.2
#SBATCH --job-name=viz_attention
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mem=64GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
python scripts/viz_attention.py --selection "$RSNA_ROOT/artifacts/viz_attention/attention_selection.csv" --workers 12
