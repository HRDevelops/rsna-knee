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

"""HPC rehearsal of the probe #4 blend (exp018 + deployed ensemble, rank space) on real studies, using
the exact Kaggle-shaped weight staging under kaggle_staging/. Correctness + timing check before Kaggle.

    python scripts/rehearse_blend018.py --n 40 --weight-exp018 0.75 --out artifacts/rehearsal_blend018.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.infer_blend018 import run_blend018  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--weight-exp018", type=float, default=0.75)
ap.add_argument("--decode-workers", type=int, default=6)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

root = Path(os.environ["RSNA_ROOT"])
art = root / "artifacts"
stage = root / "kaggle_staging"
raw = root / "data/raw"

import pandas as pd  # noqa: E402

val = pd.read_parquet(art / "exp018_coatnet2_wmil_v3labels/fold0/oof.parquet")
uids = list(val["study_uid"].iloc[:a.n])
tmp = art / "_rehearsal_blend018"
tmp.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"StudyInstanceUID": uids}).to_csv(tmp / "test.csv", index=False)
sdf = pd.read_csv(raw / "train_series.csv")
sdf[sdf["StudyInstanceUID"].isin(uids)].to_csv(tmp / "test_series.csv", index=False)

t0 = time.time()
sub = run_blend018(
    tmp / "test.csv", tmp / "test_series.csv", raw / "train_series", raw / "train_series",
    root / "code/src/rsna_knee/ensemble_recipe_v1.yaml", stage,
    [art / f"exp018_coatnet2_wmil_v3labels/fold{k}/best.pt" for k in range(5)],
    art / "exp018_coatnet2_wmil_v3labels/rank_reference.npz", art / "ensemble_v1_blend_output/rank_reference.npz",
    a.out, weight_exp018=a.weight_exp018, num_decode_workers=a.decode_workers,
)
el = time.time() - t0
print(f"\nwall {el:.0f}s for {len(uids)} studies = {el / len(uids):.2f} s/study")
print(sub.describe().T[["min", "max", "mean"]])
