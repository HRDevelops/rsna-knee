#!/bin/bash
# Execute notebooks/demo_three_studies.ipynb on a GPU: full copy (with MRI, local only) + public copy (no MRI) + HTML.
# DISK_PROJECTION_GB: 1
#SBATCH --job-name=demo_nb
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=48GB

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1 MPLBACKEND=module://matplotlib_inline.backend_inline
OUT=$RSNA_ROOT/artifacts/demo
NB=notebooks/demo_three_studies.ipynb
DEMO_SHOW_MRI=1 jupyter nbconvert --to notebook --execute $NB --ExecutePreprocessor.timeout=900 --output-dir $OUT --output demo_three_studies_full.ipynb
jupyter nbconvert --to html $OUT/demo_three_studies_full.ipynb --output-dir $OUT
DEMO_SHOW_MRI=0 jupyter nbconvert --to notebook --execute $NB --ExecutePreprocessor.timeout=900 --output-dir $OUT --output demo_three_studies_public.ipynb
ls -la $OUT
