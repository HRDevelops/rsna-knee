#!/bin/bash
# exp018 fold-0 gate: runs after the fold-0 training job (afterany, so it also reports a failed fold 0).
# CPU only. Applies the pre-registered kill rule; launches folds 1-4 through safe_sbatch only if it passes.
# DISK_PROJECTION_GB: 0.1
#SBATCH --job-name=exp018_gate
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python scripts/exp018_gate.py
