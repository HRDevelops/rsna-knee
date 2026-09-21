#!/bin/bash
# exp018 inference rehearsal on HPC (parity of the DICOM->fp16 inference path with the training path, and timing).
# DISK_PROJECTION_GB: 0.5
#SBATCH --job-name=exp018_rehearse
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=48GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
echo "== parity: fold-0 model, 60 studies =="
python scripts/rehearse_exp018.py --folds 0 --n 60 --decode-workers 4 --out "$RSNA_ROOT/artifacts/rehearsal_exp018_f0.csv"
echo "== timing: 5-fold ensemble, 60 studies =="
python scripts/rehearse_exp018.py --folds 0 1 2 3 4 --n 60 --decode-workers 4 --out "$RSNA_ROOT/artifacts/rehearsal_exp018_5f.csv"
