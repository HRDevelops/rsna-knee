#!/bin/bash
# Wiring check for the exp020 EMA path (2 epochs, ~24 studies): confirms ModelEmaV3 construction,
# per-step update, per-epoch EMA validation, best-EMA checkpointing and the final head-to-head block
# all run, before committing ~2.5 GPU-hours to the real fold-0 job.
# DISK_PROJECTION_GB: 2
#SBATCH --job-name=smoke_ema
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
#SBATCH --exclude=g005,g006

set -euo pipefail
source ~/.bashrc
source /shared/home/$USER/rsna-knee/venv/bin/activate
export RSNA_ROOT=${RSNA_ROOT:-/shared/home/$USER/rsna-knee}
cd "$RSNA_ROOT/code"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1

# eval_from_epoch is 4 in the config but the smoke run is only 2 epochs, so force EMA evaluation on
# from epoch 0 here -- otherwise the smoke test would silently skip the very code it exists to check.
python - <<'PY'
import yaml, pathlib
p = pathlib.Path("configs/exp020_coatnet2_wmil_ema.yaml")
c = yaml.safe_load(p.read_text())
c["train"]["ema"]["eval_from_epoch"] = 0
c["experiment"] = "_smoke_exp020_ema"
pathlib.Path("/tmp/_smoke_exp020.yaml").write_text(yaml.safe_dump(c))
print("smoke config written")
PY

python scripts/train_wmil.py --config /tmp/_smoke_exp020.yaml --fold 0 --out "$RSNA_ROOT/artifacts" --smoke
