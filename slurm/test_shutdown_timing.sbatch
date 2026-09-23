#!/bin/bash
# Precise shutdown-latency measurement: bash's own `time` wraps the whole process, so we get an
# external, unbiased wall-clock delta from process start to process exit -- not just the script's own
# internal print timestamps (which could theoretically lie about their own hang).
# DISK_PROJECTION_GB: 0.5
#SBATCH --job-name=test_shutdown_timing
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=64GB

set -uo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1

echo "=== timing blend018 N=3 (external wall clock via bash 'time') ==="
time timeout 120 python scripts/test_blend_shutdown.py --n-studies 3
echo "EXIT=$?"
