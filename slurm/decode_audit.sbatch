#!/bin/bash
#SBATCH --job-name=rsna_decode_audit
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --mem=64GB

# Pure CPU job (header parsing + pylibjpeg pixel decode, no GPU involved) — runs on ZEN4, not H100.
set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python scripts/decode_audit.py \
    --raw "$RSNA_ROOT/data/raw" \
    --out-data "$RSNA_ROOT/artifacts/decode_audit.parquet" \
    --out-report "$RSNA_ROOT/code/docs/DECODE_AUDIT.md" \
    --pixel-decode-sample 300 \
    --workers 32
