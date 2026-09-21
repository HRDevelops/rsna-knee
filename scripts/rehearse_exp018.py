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

"""Rehearsal of the exp018 Kaggle inference path on HPC: treat N fold-0 validation studies + the gold-58 as a
"test set", run rsna_knee.infer_wmil.run_exp018 (DICOM decode -> orientation -> fold model(s) -> submission.csv)
with fp16 exactly as the notebook will, and compare with the training-time predictions of the same model
(cached stack, bf16). Reports agreement (Spearman per label, max |dp|), the AUC of the inference-path
predictions, and measured seconds/study (decode-bound vs GPU-bound) so the Kaggle T4 runtime can be projected.

    python scripts/rehearse_exp018.py --folds 0 --n 60 --out artifacts/rehearsal_exp018.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.infer_wmil import run_exp018  # noqa: E402
from rsna_knee.wmil import TARGET_COLUMNS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--experiment", default="exp018_coatnet2_wmil_v3labels")
ap.add_argument("--folds", type=int, nargs="+", default=[0])
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--decode-workers", type=int, default=4)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

root = Path(os.environ["RSNA_ROOT"])
art = root / "artifacts" / a.experiment
val = pd.read_parquet(art / "fold0/oof.parquet")            # training-path predictions for fold-0 val studies
uids = list(val["study_uid"].iloc[:a.n])
raw = root / "data/raw"
tmp = root / "artifacts/_rehearsal_exp018"
tmp.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"StudyInstanceUID": uids}).to_csv(tmp / "test.csv", index=False)
sdf = pd.read_csv(raw / "train_series.csv")
sdf[sdf["StudyInstanceUID"].isin(uids)].to_csv(tmp / "test_series.csv", index=False)

t0 = time.time()
sub = run_exp018(tmp / "test.csv", tmp / "test_series.csv", raw / "train_series",
                 [art / f"fold{k}/best.pt" for k in a.folds], a.out, num_decode_workers=a.decode_workers)
el = time.time() - t0
print(f"\nwall {el:.0f}s for {len(uids)} studies = {el / len(uids):.2f} s/study (all overhead incl. model load) "
      f"with {len(a.folds)} fold model(s)")
if a.folds == [0]:
    ref = val.set_index("study_uid").loc[uids, TARGET_COLUMNS]
    got = sub.set_index("StudyInstanceUID").loc[uids, TARGET_COLUMNS]
    print(f"max |p_inference - p_training| = {np.abs(got.to_numpy() - ref.to_numpy()).max():.4f}; "
          f"mean = {np.abs(got.to_numpy() - ref.to_numpy()).mean():.4f}")
    sp = [spearmanr(got[c], ref[c])[0] for c in TARGET_COLUMNS]
    print("Spearman per label (inference vs training path):", np.round(sp, 4).tolist())
