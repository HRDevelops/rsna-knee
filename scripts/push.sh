#!/usr/bin/env bash
# Push local code to nu-hpc. Run from repo root (Git Bash/WSL on Windows).
# Usage: scripts/push.sh
set -euo pipefail

HPC_HOST="${RSNA_HPC_HOST:-nu-hpc}"
HPC_CODE_DIR="${RSNA_HPC_CODE_DIR:-~/rsna_knee/code}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Pushing $REPO_ROOT -> $HPC_HOST:$HPC_CODE_DIR"
rsync -avz --delete \
    --exclude '.git' \
    --exclude 'venv' \
    --exclude 'artifacts' \
    --exclude 'logs' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.ipynb_checkpoints' \
    ./ "$HPC_HOST:$HPC_CODE_DIR/"

echo "Push complete."
