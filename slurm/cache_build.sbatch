#!/bin/bash
#SBATCH --job-name=rsna_cache
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
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
module load Python/3.11.5-GCCcore-13.2.0
source ~/rsna_knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/scratch/$USER/rsna_knee}
cd ~/rsna_knee/code
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python scripts/build_cache.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 20 \
    --raw "$RSNA_ROOT/data/raw" --out "$RSNA_ROOT/data/cache" \
    --spacing 0.3 --size 384 --dtype uint8
