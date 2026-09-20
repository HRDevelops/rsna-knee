#!/bin/bash
# exp016: applies the pre-registered bar (scripts/exp016_verdict.py) once both arms' training chains
# have finished. CPU only; incomplete arms are skipped (reported), not fatal.
# DISK_PROJECTION_GB: 0.1
#SBATCH --job-name=exp016_verdict
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=32GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python scripts/exp016_verdict.py \
    --arms exp016a_dinov2b_slot_v3labels_auxrep_w010 exp016b_dinov2b_slot_v3labels_auxrep_w050 \
    --out "$RSNA_ROOT/artifacts/exp016_verdict.md"
python scripts/compare_runs.py --baseline exp006_dinov2b_slot_v3labels \
    --runs exp016a_dinov2b_slot_v3labels_auxrep_w010 exp016b_dinov2b_slot_v3labels_auxrep_w050 \
    --out "$RSNA_ROOT/artifacts/exp016_compare_vs_exp006.md"
