#!/bin/bash
# Offline blend gate (ZEN4 partition, no GPU): exp021 + exp005 re-gate, pre-registered 2026-10-01 (a05ad9f).
# DISK_PROJECTION_GB: 0.1
#SBATCH --job-name=blend_v5_gate_exp005
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --mem=16GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
echo "[cpu] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
python scripts/blend_v5_gate.py --slot-arm exp005 --out $RSNA_ROOT/artifacts/blend_v5_gate_exp005.md
