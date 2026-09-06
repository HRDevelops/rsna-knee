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
module load Python/3.11.5-GCCcore-13.2.0
source ~/rsna_knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/scratch/$USER/rsna_knee}
export HF_HOME=$RSNA_ROOT/hf_cache
cd ~/rsna_knee/code
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

MODEL=${1:-Qwen/Qwen2.5-72B-Instruct-AWQ}
python scripts/label_reports.py --model "$MODEL" \
    --reports "$RSNA_ROOT/data/raw/train.csv" \
    --rubric configs/rubric_v1.yaml \
    --out "$RSNA_ROOT/artifacts/labels/$(basename $MODEL).parquet"
