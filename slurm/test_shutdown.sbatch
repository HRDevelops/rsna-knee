#!/bin/bash
# Shutdown-safety proof for probe #4 (2026-09-23): runs each standalone test under `timeout`, exactly
# as the task instructions specify, and reports the shell exit code (0 = clean exit within timeout;
# 124 = timeout killed it, i.e. the bug is still present).
# DISK_PROJECTION_GB: 0.5
#SBATCH --job-name=test_shutdown
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64GB

set -uo pipefail   # NOT -e: a `timeout`-killed sub-test must not abort the rest of the script
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1

echo "=== blend018, N=1 ==="
timeout 120 python scripts/test_blend_shutdown.py --n-studies 1; echo "EXIT=$?"
echo "=== blend018, N=2 ==="
timeout 120 python scripts/test_blend_shutdown.py --n-studies 2; echo "EXIT=$?"
echo "=== blend018, N=3 ==="
timeout 120 python scripts/test_blend_shutdown.py --n-studies 3; echo "EXIT=$?"

echo "=== exp018, N=3 ==="
timeout 120 python scripts/test_exp018_shutdown.py --n-studies 3; echo "EXIT=$?"

echo "=== prefetch_decode (used by submit-ensemble/efficiency.ipynb), N=3, break after 1 ==="
timeout 120 python scripts/test_prefetch_decode_shutdown.py --n-studies 3 --break-after 1; echo "EXIT=$?"
echo "=== prefetch_decode, N=3, consume all (normal exhaustion) ==="
timeout 120 python scripts/test_prefetch_decode_shutdown.py --n-studies 3 --break-after 99; echo "EXIT=$?"

echo "=== ALL TESTS DONE ==="
