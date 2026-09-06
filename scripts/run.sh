#!/usr/bin/env bash
# Submit a job on nu-hpc and tail its log. Run from repo root (Git Bash/WSL on Windows).
# Usage: scripts/run.sh slurm/train.sbatch [args passed through to the sbatch script...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <slurm-script-relative-path> [args...]" >&2
    exit 1
fi

HPC_HOST="${RSNA_HPC_HOST:-nu-hpc}"
HPC_CODE_DIR="${RSNA_HPC_CODE_DIR:-/shared/home/husnain.rasool/rsna-knee/code}"

SBATCH_SCRIPT="$1"
shift

echo "Submitting $SBATCH_SCRIPT on $HPC_HOST:$HPC_CODE_DIR"
JOB_ID=$(ssh "$HPC_HOST" "cd $HPC_CODE_DIR && mkdir -p logs && sbatch --parsable $SBATCH_SCRIPT $*")
echo "Submitted job(s): $JOB_ID"
echo "Watching queue (Ctrl-C to stop watching; job keeps running):"
ssh "$HPC_HOST" "squeue -u \$USER"

echo
echo "Tail logs with:"
echo "  ssh $HPC_HOST \"tail -f $HPC_CODE_DIR/logs/*${JOB_ID}*.out\""
