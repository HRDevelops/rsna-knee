#!/bin/bash
# 2026-09-18: runs automatically after every exp010/exp012/exp015 training chain has finished (submit
# with --dependency=afterany:<all last chain links>). CPU only. Produces, under artifacts/:
#   final_compare_vs_exp006.md            per-label OOF + gold-58 deltas of exp010/012/015 vs exp006
#   final_ensemble_reselection.txt        greedy forward selection over {exp003,005,006,010,012,015}
#   ensemble_recipe_reselect_20260918.csv the re-selected recipe (NEVER overwrites the deployed
#                                         artifacts/ensemble_recipe.csv — see build_ensemble.py --out)
# Incomplete runs are skipped by both scripts (reported as SKIP), not fatal.
# DISK_PROJECTION_GB: 0.1
#SBATCH --job-name=final_reselection
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=32GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python scripts/compare_runs.py --baseline exp006_dinov2b_slot_v3labels \
    --runs exp010_dinov2b_slot_v3labels_t32 exp012_dinov2b_slot_v3labels_acl_prior \
           exp015_dinov2b_slot_v3labels_448 \
    --out "$RSNA_ROOT/artifacts/final_compare_vs_exp006.md"

python scripts/build_ensemble.py \
    --candidates exp003_dinov2b_slot_v2labels exp005_dinov2s_slot_v3labels exp006_dinov2b_slot_v3labels \
                 exp010_dinov2b_slot_v3labels_t32 exp012_dinov2b_slot_v3labels_acl_prior \
                 exp015_dinov2b_slot_v3labels_448 \
    --out "$RSNA_ROOT/artifacts/ensemble_recipe_reselect_20260918.csv" \
    | tee "$RSNA_ROOT/artifacts/final_ensemble_reselection.txt"
