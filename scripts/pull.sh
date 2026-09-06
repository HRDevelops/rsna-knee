#!/usr/bin/env bash
# Pull artifacts/logs for one experiment back from nu-hpc scratch. Run from repo root.
# Usage: scripts/pull.sh <exp_name>
#   e.g. scripts/pull.sh exp001_dinov2s_slot
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <exp_name>" >&2
    exit 1
fi

EXP="$1"
HPC_HOST="${RSNA_HPC_HOST:-nu-hpc}"
HPC_CODE_DIR="${RSNA_HPC_CODE_DIR:-~/rsna_knee/code}"
HPC_SCRATCH="${RSNA_HPC_SCRATCH:-/scratch/\$USER/rsna_knee}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "artifacts/$EXP" "logs"

echo "Pulling artifacts/$EXP from $HPC_HOST"
rsync -avz "$HPC_HOST:$HPC_SCRATCH/artifacts/$EXP/" "artifacts/$EXP/" || \
    echo "  (no artifacts dir for $EXP yet on remote — skipping)"

echo "Pulling logs from $HPC_HOST:$HPC_CODE_DIR/logs/"
rsync -avz --include="*${EXP}*" --exclude='*' "$HPC_HOST:$HPC_CODE_DIR/logs/" "logs/" || \
    echo "  (no matching logs yet — skipping)"

echo "Pull complete: artifacts/$EXP/, logs/"
