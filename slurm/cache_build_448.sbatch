#!/bin/bash
# 2026-09-18: FULL 448px cache (exp015, resolution ablation vs exp006's 384px/T=24). Resamples DICOM
# to ~0.2571mm/px on a 448x448 grid = the same ~115.2mm field of view as the main 0.3mm/px 384px
# cache, i.e. genuinely finer spacing (more real pixels per mm), NOT an upsample of the 384px cache.
# Written to data/cache_448 — never touches the main data/cache. Resumable (skips existing .npy).
# Measured cost: the main cache is 113GB; the 448 variant scales by (448/384)^2 = 1.361 -> ~154GB.
# DISK_PROJECTION_GB: 160
#SBATCH --job-name=rsna_cache_448
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --array=0-19           # 20 shards over 4407 studies

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python scripts/build_cache.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 20 \
    --raw "$RSNA_ROOT/data/raw" --out "$RSNA_ROOT/data/cache_448" \
    --manifest-out "$RSNA_ROOT/artifacts/cache_manifests_448" \
    --spacing 0.2571 --size 448 --dtype uint8
