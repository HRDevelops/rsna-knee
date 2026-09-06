#!/bin/bash
#SBATCH --job-name=rsna_cache
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
# No environment modules or conda on this cluster — plain venv on system python3.
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python scripts/build_cache.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 20 \
    --raw "$RSNA_ROOT/data/raw" --out "$RSNA_ROOT/data/cache" \
    --manifest-out "$RSNA_ROOT/artifacts/cache_manifests" \
    --spacing 0.3 --size 384 --dtype uint8
