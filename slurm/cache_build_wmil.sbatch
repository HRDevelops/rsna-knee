#!/bin/bash
# exp018 cache (2026-09-21): per-STUDY uint8 [64,336,336] stacks, 140mm FOV, 6-94% slice span, 5 slots
# (18/14/12/8/12 slices). 4407 x 64 x 336 x 336 B = 31.8 GB (NOT ~155GB: that figure was the exp015 448px
# cache; 5 slots x 64 slices each would be 159GB, but series are 20-45 slices, so a 64-slice-per-slot layout
# would only duplicate slices). See docs/EXPERIMENTS.md 2026-09-21 (exp018).
# DISK_PROJECTION_GB: 34
# DISK_PROJECTION_PATH: data/cache_wmil
#SBATCH --job-name=rsna_cache_wmil
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --array=0-19           # 20 shards over 4407 studies

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python scripts/build_cache_wmil.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 20 \
    --raw "$RSNA_ROOT/data/raw" --out "$RSNA_ROOT/data/cache_wmil" \
    --manifest-out "$RSNA_ROOT/artifacts/cache_manifests_wmil"
