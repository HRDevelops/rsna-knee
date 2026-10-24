#!/bin/bash
# Phase 0A (docs/research/PLAN_v1_3.md s5, amended 2026-10-24): rigid mutual-information registration audit of the
# retained DICOM geometry. 150 studies (same seed-0 sample as scripts/check_series_geometry.py), 15 shards x 10 studies,
# CPU only. Writes artifacts/phase0a/{pairs,series}_<shard>.parquet and qc/*.png (image data: local only, never pushed);
# then run scripts/phase0a_report.py on the directory.
# Pilot: PHASE0A_OUT=.../phase0a_pilot PHASE0A_EXTRA="--max-studies 2" with --array=0.
# DISK_PROJECTION_GB: 1
# DISK_PROJECTION_PATH: artifacts/phase0a
#SBATCH --job-name=rsna_phase0a
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=24GB
#SBATCH --array=0-14

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
OUT="${PHASE0A_OUT:-$RSNA_ROOT/artifacts/phase0a}"

python scripts/phase0a_registration.py --n-studies 150 --shard "$SLURM_ARRAY_TASK_ID" \
    --n-shards "${PHASE0A_SHARDS:-15}" --out "$OUT" --qc --threads "$SLURM_CPUS_PER_TASK" ${PHASE0A_EXTRA:-}
