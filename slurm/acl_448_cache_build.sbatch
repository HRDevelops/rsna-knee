#!/bin/bash
# 2026-09-17 quota-cheap 448px resolution test: builds a cache for a 400-study SUBSET only
# (artifacts/acl_448_subset_studies.txt) at a finer spacing (~0.2571mm/px) than the main 0.3mm/px
# cache, same ~115mm field of view at 448px instead of 384px -- a genuine resolution increase, not
# an upsample of already-384px data (see docs/EXPERIMENTS.md's exp010 entry for why that
# distinction matters). Separate output dir (data/cache_448_subset) -- does NOT touch the main
# data/cache/ at all. DELETE data/cache_448_subset/ after the comparison is read out; it is not a
# permanent second cache variant.
#SBATCH --job-name=rsna_cache_448_subset
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --array=0-3            # 4 shards over 400 studies

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python scripts/build_cache.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4 \
    --raw "$RSNA_ROOT/data/raw" --out "$RSNA_ROOT/data/cache_448_subset" \
    --manifest-out "$RSNA_ROOT/artifacts/cache_manifests_448_subset" \
    --study-list "$RSNA_ROOT/artifacts/acl_448_subset_studies.txt" \
    --spacing 0.2571 --size 448 --dtype uint8
