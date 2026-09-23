# Copyright 2026 Husnain Rasool
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TTA test: laterality-swap (2026-09-09 instruction) — for each ensemble member/fold, run a
SECOND inference pass with every present slot's volume horizontally flipped (the same
deterministic transform training's `laterality_swap_prob` augmentation applies), then swap the
paired medial/lateral columns in the OUTPUT predictions to correct for the flip (dataset.py's
LATERALITY_SWAP_PAIRS), and average with the already-existing normal-orientation oof.parquet — no
need to recompute the normal pass, it's already on disk. Keep TTA only if it's OOF-positive on the
final ensemble macro vs labels_v2_final; report the delta and a 2x-inference-time runtime cost
estimate.

Usage:
    python scripts/eval_tta_lateral_swap.py --experiments exp003_dinov2b_slot_v2labels exp005_dinov2s_slot_v3labels exp006_dinov2b_slot_v3labels
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.dataset import LATERALITY_SWAP_PAIRS, TARGET_COLUMNS, RSNAKneeDataset, collate_fn  # noqa: E402
from rsna_knee.model import SlotModel  # noqa: E402

TARGETS = TARGET_COLUMNS


def expand_env(obj):
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


@torch.no_grad()
def run_flipped_inference(model, loader, device, autocast_dtype) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        volumes = batch["volumes"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        volumes = torch.flip(volumes, dims=[-1])  # flip every present slot's image
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            logits = model(volumes, mask)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        for i, j in LATERALITY_SWAP_PAIRS:
            probs[:, [i, j]] = probs[:, [j, i]]  # undo the flip's effect on paired labels
        for uid, p in zip(batch["study_uid"], probs):
            row = {"study_uid": uid}
            row.update({c: float(v) for c, v in zip(TARGETS, p)})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", required=True)
    args = parser.parse_args()

    rsna_root = Path(os.environ.get("RSNA_ROOT", "."))
    root = rsna_root / "artifacts"
    raw_dir = rsna_root / "data" / "raw"
    cache_manifests = root / "cache_manifests"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16

    t0 = time.time()
    for exp_name in args.experiments:
        exp_dir = root / exp_name
        for fold_dir in sorted(exp_dir.glob("fold*")):
            fold = int(fold_dir.name.replace("fold", ""))
            with open(fold_dir / "config.yaml") as f:
                cfg = expand_env(yaml.safe_load(f))

            common_kwargs = dict(
                cfg=cfg, cache_dir=Path(cfg["data"]["cache_dir"]),
                train_series_csv=raw_dir / "train_series.csv", cache_manifest_dir=cache_manifests,
                labels_parquet=Path(cfg["data"]["labels"]), folds_parquet=Path(cfg["data"]["folds"]),
                final_side_csv=root / "final_side.csv", decode_audit_parquet=root / "decode_audit.parquet",
                fold=fold,
            )
            val_ds = RSNAKneeDataset(split="val", **common_kwargs)
            gold_ds = RSNAKneeDataset(split="gold", **common_kwargs)
            bs = cfg["train"]["batch_size_studies"]
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate_fn)
            gold_loader = DataLoader(gold_ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate_fn)

            ckpt = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
            model = SlotModel(cfg["model"]["backbone"], num_targets=len(TARGETS), pretrained=False).to(device)
            model.load_state_dict(ckpt["ema"])

            val_flipped = run_flipped_inference(model, val_loader, device, autocast_dtype)
            gold_flipped = run_flipped_inference(model, gold_loader, device, autocast_dtype)
            val_flipped.to_parquet(fold_dir / "oof_flipped_tta.parquet", index=False)
            gold_flipped.to_parquet(fold_dir / "gold58_predictions_flipped_tta.parquet", index=False)
            print(f"{exp_name}/{fold_dir.name}: {len(val_flipped)} val + {len(gold_flipped)} gold "
                  f"flipped predictions written ({time.time() - t0:.0f}s elapsed total)", file=sys.stderr)


if __name__ == "__main__":
    main()
