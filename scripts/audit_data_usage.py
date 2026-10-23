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

"""How much of the raw data does the exp018 cache actually use? Metadata only (no pixels): train_series.csv, the
decode audit (files, rows, columns, pixel spacing per series) and the cache's own slot rules (rsna_knee.wmil).

Reports: series kept vs dropped (and which kinds are dropped), slices kept vs available, in-plane field of view
against the fixed 140 mm crop, and how much in-plane resolution the 0.417 mm cache grid throws away.

    RSNA_ROOT=... python scripts/audit_data_usage.py
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee import wmil  # noqa: E402

root = Path(os.environ["RSNA_ROOT"])
sdf = pd.read_csv(root / "data/raw/train_series.csv").rename(columns={
    "StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid", "Anatomical_Plane": "plane",
    "Fluid_Sensitive": "fluid"})
aud = pd.read_parquet(root / "artifacts/decode_audit.parquet")[["series_uid", "n_files", "rows", "columns", "pixel_spacing"]]
df = sdf.merge(aud, on="series_uid", how="left")
df["ps_col"] = df.pixel_spacing.apply(lambda v: float(ast.literal_eval(v)[1]) if isinstance(v, str) and v.startswith("[") else np.nan)
df["ps_row"] = df.pixel_spacing.apply(lambda v: float(ast.literal_eval(v)[0]) if isinstance(v, str) and v.startswith("[") else np.nan)
df["fov_w"] = df["columns"] * df.ps_col
df["fov_h"] = df["rows"] * df.ps_row

used_rows = []
for u, g in df.groupby("study_uid"):
    rows = [dict(series_uid=r.series_uid, plane=r.plane, fluid=r.fluid, n_slices=int(r.n_files or 0))
            for r in g.itertuples() if (r.n_files or 0) > 0]
    chosen = wmil.pick_slots(rows)
    for i, r in chosen.items():
        n = r["n_slices"]
        lo, hi = int(round(wmil.SPAN[0] * (n - 1))), int(round(wmil.SPAN[1] * (n - 1)))
        take = np.clip(np.round(np.linspace(lo, hi, wmil.SLOT_LEN[i])).astype(int), 0, n - 1)
        used_rows.append(dict(study_uid=u, series_uid=r["series_uid"], slot=wmil.SLOT_SPECS[i][0], n_native=n,
                              n_span=hi - lo + 1, n_unique_used=len(set(take.tolist())), n_take=wmil.SLOT_LEN[i]))
used = pd.DataFrame(used_rows)
df["used"] = df.series_uid.isin(set(used.series_uid))

print(f"studies {df.study_uid.nunique():,} | series {len(df):,} | DICOM slices {int(df.n_files.sum()):,}")
print(f"series used by the cache: {df.used.mean():.1%} ({int(df.used.sum()):,} of {len(df):,})")
drop = df[~df.used]
print("dropped series by plane / fluid-sensitive:")
print(drop.groupby(["plane", "fluid"]).size().rename("n").to_string())
print(f"slices decoded into the cache (unique): {int(used.n_unique_used.sum()):,} = "
      f"{used.n_unique_used.sum() / df.n_files.sum():.1%} of all slices; "
      f"{used.n_unique_used.sum() / used.n_native.sum():.1%} of the slices in the series that ARE used")
print("per slot: native slices (median), slices kept (median), subsampling factor (median native-in-span / kept):")
for s, g in used.groupby("slot", sort=False):
    print(f"  {s:15s} native {g.n_native.median():5.0f} | kept {g.n_unique_used.median():4.0f} | "
          f"factor {np.median(g.n_span / g.n_take):4.2f} | slots with fewer native slices than the slot length: "
          f"{(g.n_native < g.n_take).mean():.1%}")

u = df[df.used]
fov = u[["fov_w", "fov_h"]].min(axis=1)
print("\nin-plane field of view of used series, min(width, height) mm, quantiles:",
      fov.quantile([.05, .25, .5, .75, .95]).round(0).to_dict())
big = u.fov_w.clip(lower=140) * u.fov_h.clip(lower=140)
print(f"used series with FOV > 140 mm on some axis (crop discards periphery): {((u.fov_w > 140) | (u.fov_h > 140)).mean():.1%}; "
      f"median share of the image area discarded by the crop: {np.median(1 - (140 * 140) / big):.1%}")
print(f"used series with FOV < 140 mm on some axis (padding): {((u.fov_w < 140) | (u.fov_h < 140)).mean():.1%}")
cache_px = 140 / 336
ratio = cache_px / u.ps_col
print(f"cache pixel {cache_px:.4f} mm: series downsampled (native pixel finer than the cache): {(ratio > 1.0).mean():.1%}; "
      f"downsampling factor median {ratio.median():.2f}, p90 {ratio.quantile(.9):.2f}")
