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

"""Stage the exp021 (seed 43) weights for Kaggle, run on HPC:
  kaggle_staging/exp021_weights/fold{k}_seed43_best.pt  -> NEW dataset `rsna-knee-exp021-weights`

Nothing else is staged, on purpose. The submission reuses the SCORED `rsna-knee-exp018-code` (8,245 B
`infer_wmil.py`, pre-2ec038d) and the SCORED `rsna-knee-exp018-weights` unchanged; this dataset only adds
the five seed-43 checkpoints. The lean format is exactly exp018's (model, arch, res, targets), which is
all `infer_wmil.load_fold_models` reads, so the scored driver loads these files without modification.
Refuses to stage unless all 5 folds exist: a partial seed-43 set would silently make a 5+k ensemble."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

EXP = "exp021_coatnet2_wmil_seed43"
root = Path(os.environ["RSNA_ROOT"])
w_dst = root / "kaggle_staging/exp021_weights"

srcs = [root / "artifacts" / EXP / f"fold{k}/best.pt" for k in range(5)]
missing = [str(p) for p in srcs if not p.exists()]
if missing:
    sys.exit(f"refusing to stage: {len(missing)} fold(s) missing: {missing}")

w_dst.mkdir(parents=True, exist_ok=True)
for k, p in enumerate(srcs):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert ck["arch"] == "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k" and int(ck["res"]) == 384, (ck["arch"], ck["res"])
    assert ck.get("img_size") is None, "exp021 must be built exactly like exp018 (no img_size)"
    lean = {"model": ck["model"], "arch": ck["arch"], "res": ck["res"], "targets": ck["targets"]}
    out = w_dst / f"fold{k}_seed43_best.pt"
    torch.save(lean, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"  fold {k}: {out.name} {out.stat().st_size:,} B sha256 {sha[:16]}... "
          f"val {ck['val_macro_auc']:.4f} gold {ck['gold58_macro_auc']:.4f}")

(w_dst / "dataset-metadata.json").write_text(json.dumps({
    "title": "rsna-knee-exp021-weights", "id": "hrdevelops/rsna-knee-exp021-weights",
    "licenses": [{"name": "apache-2.0"}]}, indent=1))
print(f"staged 5 files + dataset-metadata.json in {w_dst}")
