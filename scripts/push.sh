#!/usr/bin/env bash
# Push local code to nu-hpc. Run from repo root (Git Bash/WSL on Windows).
# Usage: scripts/push.sh
set -euo pipefail

HPC_HOST="${RSNA_HPC_HOST:-nu-hpc}"
# No /scratch on this cluster — single BeeGFS pool at /shared/home (see docs/HPC_WORKFLOW.md).
HPC_CODE_DIR="${RSNA_HPC_CODE_DIR:-/shared/home/husnain.rasool/rsna-knee/code}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Pushing $REPO_ROOT -> $HPC_HOST:$HPC_CODE_DIR"

if command -v rsync >/dev/null 2>&1; then
    rsync -avz --delete \
        --exclude '.git' \
        --exclude 'venv' \
        --exclude 'artifacts' \
        --exclude 'logs' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.ipynb_checkpoints' \
        --exclude 'kaggle_staging' \
        ./ "$HPC_HOST:$HPC_CODE_DIR/"
else
    # No rsync on this machine (common on plain Git Bash/Windows without WSL/cygwin) — fall back
    # to tar-over-ssh. This overwrites files but does NOT delete remote files removed locally.
    echo "rsync not found locally — falling back to tar-over-ssh (no delete-sync)."
    ssh "$HPC_HOST" "mkdir -p '$HPC_CODE_DIR'"
    tar czf - \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='artifacts' \
        --exclude='logs' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.ipynb_checkpoints' \
        --exclude='kaggle_staging' \
        . | ssh "$HPC_HOST" "tar xzf - -C '$HPC_CODE_DIR'"
fi

echo "Push complete."
