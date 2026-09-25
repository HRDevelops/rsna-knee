#!/bin/bash
# Phase-6 CPU job (ZEN4 partition, no GPU): profile_decode. See the script's docstring.
# DISK_PROJECTION_GB: 0.1
#SBATCH --job-name=profile_decode
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=48GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
echo "[cpu] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
python scripts/profile_decode_stages.py --n-studies 200 --workers 3 4
