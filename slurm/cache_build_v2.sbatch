#!/bin/bash
# Phase 0B (docs/research/PLAN_v1_3.md s4.1): cache v2 = every series, every native slice, 384 px at 0.4167 mm
# (160 mm field), exact affine + linear, no prefilter. 819,078 slices x 147 KB ~= 120 GB uint8.
# DISK_PROJECTION_GB: 125
# DISK_PROJECTION_PATH: data/cache_v2
#SBATCH --job-name=rsna_cache_v2
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=32GB
#SBATCH --array=0-23

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export OMP_NUM_THREADS=4

python scripts/build_cache_v2.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 24 \
    --raw "$RSNA_ROOT/data/raw" --out "${CACHE_V2_OUT:-$RSNA_ROOT/data/cache_v2}" \
    --manifest-out "${CACHE_V2_MANIFEST:-$RSNA_ROOT/artifacts/cache_manifests_v2}" ${CACHE_V2_EXTRA:-}
