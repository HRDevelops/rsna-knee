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
HPC_CODE_DIR="${RSNA_HPC_CODE_DIR:-/shared/home/husnain.rasool/rsna-knee/code}"
# No /scratch on this cluster — artifacts live alongside code on the single BeeGFS pool.
HPC_SCRATCH="${RSNA_HPC_SCRATCH:-/shared/home/husnain.rasool/rsna-knee}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "artifacts/$EXP" "logs"

if command -v rsync >/dev/null 2>&1; then
    echo "Pulling artifacts/$EXP from $HPC_HOST"
    rsync -avz "$HPC_HOST:$HPC_SCRATCH/artifacts/$EXP/" "artifacts/$EXP/" || \
        echo "  (no artifacts dir for $EXP yet on remote — skipping)"

    echo "Pulling logs from $HPC_HOST:$HPC_CODE_DIR/logs/"
    rsync -avz --include="*${EXP}*" --exclude='*' "$HPC_HOST:$HPC_CODE_DIR/logs/" "logs/" || \
        echo "  (no matching logs yet — skipping)"
else
    # No rsync locally — fall back to tar-over-ssh (whole-directory copy, no partial include filter).
    echo "rsync not found locally — falling back to tar-over-ssh."
    echo "Pulling artifacts/$EXP from $HPC_HOST"
    ssh "$HPC_HOST" "test -d '$HPC_SCRATCH/artifacts/$EXP' && tar czf - -C '$HPC_SCRATCH/artifacts' '$EXP'" \
        | tar xzf - -C "artifacts/" 2>/dev/null || \
        echo "  (no artifacts dir for $EXP yet on remote — skipping)"

    echo "Pulling logs from $HPC_HOST:$HPC_CODE_DIR/logs/"
    ssh "$HPC_HOST" "cd '$HPC_CODE_DIR/logs' 2>/dev/null && tar czf - -- *${EXP}* 2>/dev/null" \
        | tar xzf - -C "logs/" 2>/dev/null || \
        echo "  (no matching logs yet — skipping)"
fi

echo "Pull complete: artifacts/$EXP/, logs/"
