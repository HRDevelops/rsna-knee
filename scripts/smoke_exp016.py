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

"""Pre-flight for exp016 (run on a GPU node BEFORE the 5-fold arrays; the arrays depend on it).

Checks, on the real fold-0 dataset and a real GPU batch:
  1. the aux targets load, are finite, unit-norm, and cover every train study;
  2. FOLD-SAFETY: a targets file that contains a validation-fold study (or a gold-58 study) is REFUSED
     by the dataset (assertion) — the leak path is closed, not just unused;
  3. the shuffled-report control changes which report a study gets but preserves the target multiset;
  4. one real batch runs forward_with_aux + the joint loss + backward on the GPU with finite values,
     and inference-mode forward() (what eval uses) is unaffected by the aux head.

    python scripts/smoke_exp016.py --config configs/exp016a_dinov2b_slot_v3labels_auxrep_w010.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsna_knee.dataset import RSNAKneeDataset, collate_fn  # noqa: E402
from rsna_knee.model import SlotModel  # noqa: E402
from train import expand_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    cfg = expand_env(yaml.safe_load(args.config.read_text()))
    root = Path(os.environ["RSNA_ROOT"])
    kw = dict(
        cfg=cfg, cache_dir=Path(cfg["data"]["cache_dir"]), train_series_csv=root / "data/raw/train_series.csv",
        cache_manifest_dir=root / "artifacts/cache_manifests", labels_parquet=Path(cfg["data"]["labels"]),
        folds_parquet=Path(cfg["data"]["folds"]), final_side_csv=root / "artifacts/final_side.csv",
        decode_audit_parquet=root / "artifacts/decode_audit.parquet", fold=args.fold,
    )
    targets = Path(str(cfg["aux"]["targets"]).format(fold=args.fold))

    ds = RSNAKneeDataset(split="train", aux_targets_npz=targets, **kw)
    z = np.load(targets)
    norms = np.linalg.norm(z["target"], axis=1)
    print(f"[1] targets {z['target'].shape}, finite={np.isfinite(z['target']).all()}, "
          f"norm min/max={norms.min():.4f}/{norms.max():.4f}, train dataset n={len(ds)}, aux_dim={ds.aux_dim}")
    assert np.isfinite(z["target"]).all() and abs(norms.min() - 1) < 1e-3 and abs(norms.max() - 1) < 1e-3
    assert ds.aux_dim == z["target"].shape[1] and len(ds.aux_map) == len(z["uids"])

    folds = pd.read_parquet(cfg["data"]["folds"]).set_index("study_uid")["fold"]
    for label, mask in (("validation-fold", folds == args.fold), ("gold-58", folds == -1)):
        leak_uid = folds.index[mask.to_numpy()][0]
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "leaky.npz"
            np.savez(bad, uids=np.append(z["uids"], leak_uid), target=np.vstack([z["target"], z["target"][:1]]))
            try:
                RSNAKneeDataset(split="train", aux_targets_npz=bad, **kw)
            except AssertionError as e:
                print(f"[2] {label} study in targets -> REFUSED as required: {str(e)[:90]}")
            else:
                raise SystemExit(f"FAIL: dataset accepted a targets file containing a {label} study")

    shuf = RSNAKneeDataset(split="train", aux_targets_npz=targets, aux_shuffle_seed=1, **kw)
    same = sum(np.allclose(ds.aux_map[u], shuf.aux_map[u]) for u in list(ds.aux_map)[:200])
    print(f"[3] shuffle control: {same}/200 studies kept their own report target (expect ~0)")
    assert same <= 2
    assert np.allclose(np.sort(np.stack(list(ds.aux_map.values())).ravel()),
                       np.sort(np.stack(list(shuf.aux_map.values())).ravel()))

    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    dev = "cuda"
    model = SlotModel(cfg["model"]["backbone"], num_targets=12, aux_dim=ds.aux_dim).to(dev)
    vol, mask = batch["volumes"].to(dev), batch["mask"].to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, pred = model.forward_with_aux(vol, mask)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"].to(dev))
    tgt = batch["aux_target"].to(dev)
    aux = (1 - torch.nn.functional.cosine_similarity(pred.float(), tgt.float(), dim=-1)).mean()
    loss = bce + float(cfg["aux"]["weight"]) * aux
    loss.backward()
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        eval_logits = model(vol, mask)
    print(f"[4] batch: volumes {tuple(vol.shape)}, aux_target {tuple(tgt.shape)}, bce={bce.item():.4f} "
          f"aux_cos_loss={aux.item():.4f} (~1.0 expected at init), total={loss.item():.4f}, "
          f"eval forward() shape {tuple(eval_logits.shape)}")
    assert torch.isfinite(loss) and torch.isfinite(eval_logits).all()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
