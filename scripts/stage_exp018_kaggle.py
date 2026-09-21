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

"""Stage the two exp018 Kaggle datasets (run on HPC, then download/upload by the user):
  kaggle_staging/exp018_code/rsna_knee/{__init__,preprocess,wmil,infer_wmil}.py     -> dataset `rsna-knee-exp018-code`
  kaggle_staging/exp018_weights/fold{k}_best.pt (arch, res, model, targets only)      -> dataset `rsna-knee-exp018-weights`
Separate datasets (not new versions of `rsna-knee-code` / an existing weights dataset) so the frozen deployed
notebooks' inputs are not touched."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch

root = Path(os.environ["RSNA_ROOT"])
stage = root / "kaggle_staging"
src = root / "code/src/rsna_knee"
code_dst = stage / "exp018_code/rsna_knee"
code_dst.mkdir(parents=True, exist_ok=True)
for f in ("__init__.py", "preprocess.py", "wmil.py", "infer_wmil.py"):
    shutil.copy(src / f, code_dst / f)
print("code staged:", sorted(p.name for p in code_dst.iterdir()))

exp = sys.argv[1] if len(sys.argv) > 1 else "exp018_coatnet2_wmil_v3labels"
w_dst = stage / "exp018_weights"
w_dst.mkdir(parents=True, exist_ok=True)
n = 0
for k in range(5):
    p = root / "artifacts" / exp / f"fold{k}/best.pt"
    if not p.exists():
        print(f"  fold {k}: no best.pt, skipped")
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    lean = {"model": ck["model"], "arch": ck["arch"], "res": ck["res"], "targets": ck["targets"]}
    torch.save(lean, w_dst / f"fold{k}_best.pt")
    n += 1
    print(f"  fold {k}: staged ({(w_dst / f'fold{k}_best.pt').stat().st_size / 1e6:.0f} MB) "
          f"val {ck['val_macro_auc']:.4f} gold {ck['gold58_macro_auc']:.4f}")
print(f"{n} weight file(s) in {w_dst}")
