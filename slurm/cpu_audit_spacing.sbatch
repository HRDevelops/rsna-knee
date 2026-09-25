#!/bin/bash
# Phase-6 CPU job (ZEN4 partition, no GPU): audit_spacing. See the script's docstring.
# DISK_PROJECTION_GB: 0.1
#SBATCH --job-name=audit_spacing
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --mem=48GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
echo "[cpu] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
python scripts/audit_slice_spacing.py --workers 32 --out $RSNA_ROOT/artifacts/slice_spacing_audit.md
