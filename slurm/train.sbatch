#!/bin/bash
#SBATCH --job-name=rsna_train
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --array=0-4            # one job per fold

set -euo pipefail
source ~/.bashrc
# No environment modules or conda on this cluster — plain venv on system python3.
# CUDA is provided by the GPU node itself (verify with `nvidia-smi` inside the job).
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
# 2026-09-07: exp002 jobs were dying ~54min-2h in with no Python exception (external kill),
# including a job running with no co-located jobs of ours at all — points to CUDA memory
# fragmentation building up from gradient checkpointing combined with our variable-length
# per-batch backbone calls (valid-slot count varies per sample), not just the cluster's GPU
# co-location bug (documented in docs/HPC_WORKFLOW.md §3.5). This is PyTorch's own documented
# mitigation for exactly this "reserved but unallocated memory" growth pattern.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=${1:-configs/exp001_dinov2s_slot.yaml}
python scripts/train.py --config "$CONFIG" --fold "$SLURM_ARRAY_TASK_ID" \
    --out "$RSNA_ROOT/artifacts" --resume auto --max-hours 7.0
