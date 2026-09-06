#!/bin/bash
#SBATCH --job-name=rsna_labels
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB

set -euo pipefail
source ~/.bashrc
# No environment modules or conda on this cluster — plain venv on system python3.
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
export HF_HOME=$RSNA_ROOT/hf_cache
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Confirmed 2026-09-06: H100 compute nodes have internet, so this can download on first run too —
# but pre-downloading via `huggingface-cli download` avoids burning GPU-allocated time on a
# multi-GB transfer. Second-opinion model: Meta never released an official Llama-3.3-70B AWQ
# quant, use the community one: casperhansen/llama-3.3-70b-instruct-awq (612k+ downloads on HF).
MODEL=${1:-Qwen/Qwen2.5-72B-Instruct-AWQ}
python scripts/label_reports.py --model "$MODEL" \
    --reports "$RSNA_ROOT/data/raw/train.csv" \
    --rubric configs/rubric_v1.yaml \
    --out "$RSNA_ROOT/artifacts/labels/$(basename $MODEL).parquet"
