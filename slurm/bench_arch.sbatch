#!/bin/bash
# Phase-5 capacity screen: time candidate Window-MIL backbones and project dual-T4 runtime, so exp019's
# runtime kill rule can be decided before committing GPU-hours to training.
# DISK_PROJECTION_GB: 0.5
#SBATCH --job-name=bench_arch
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mem=64GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1     # pretrained=False everywhere in this script; no hub access needed

python scripts/bench_arch_runtime.py --n-studies 40 --n-folds 5
