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

"""Acquired vs reconstructed in-plane resolution (header-only, one header per series). Used for
docs/research/PLAN_v1_2.md s1 (anti-aliasing rule of cache v2).

    RSNA_ROOT=... python scripts/audit_acquisition_resolution.py 300

MR scanners often reconstruct onto a finer grid than they acquire (zero-filled interpolation). Then PixelSpacing
overstates the real resolution, and downsampling to a coarser cache grid loses nothing as long as the cache pixel is
still finer than the ACQUIRED resolution. AcquisitionMatrix (0018,1310) = frequency rows \\ frequency columns \\ phase
rows \\ phase columns (two of the four are non-zero); InPlanePhaseEncodingDirection says which image axis is phase
(COL = vertical, ROW = horizontal). Acquired spacing per axis = field of view along that axis / acquired samples. A
rectangular phase field of view (PercentPhaseFieldOfView < 100) is handled by taking the smaller of the image extent and
the frequency field of view times that percentage.

Reports: zero-fill factors, acquired spacing per axis, and for the two cache grids (exp018 0.4167 mm, cache v2
0.365 mm) how many series are downsampled nominally (PixelSpacing finer than the grid) vs really (acquired spacing finer
than the grid). Also slice thickness and gap, field strength and manufacturer. Same random sample rule as
check_series_geometry.py (seed 0).
"""
import glob
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import pydicom

root = os.environ["RSNA_ROOT"] + "/data/raw"
n_studies = int(sys.argv[1]) if len(sys.argv) > 1 else 300
sdf = pd.read_csv(root + "/train_series.csv")
uids = sorted(sdf.StudyInstanceUID.unique())
random.seed(0)
pick = random.sample(uids, n_studies)
TAGS = ["Rows", "Columns", "PixelSpacing", "AcquisitionMatrix", "InPlanePhaseEncodingDirection",
        "PercentPhaseFieldOfView", "PercentSampling", "SliceThickness", "SpacingBetweenSlices", "MRAcquisitionType",
        "MagneticFieldStrength", "Manufacturer"]
GRIDS = {"exp018 0.4167 mm": 140.0 / 336, "cache v2 0.365 mm": 140.0 / 384}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


t0 = time.time()
rows = []
for u in pick:
    for _, s in sdf[sdf.StudyInstanceUID == u].iterrows():
        files = sorted(glob.glob(f"{root}/train_series/{u}/{s.SeriesInstanceUID}/*.dcm"))
        if not files:
            continue
        h = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True, specific_tags=TAGS)
        try:
            ps_r, ps_c = (float(v) for v in h.PixelSpacing)  # row spacing (vertical), column spacing (horizontal)
            n_r, n_c = int(h.Rows), int(h.Columns)
        except Exception:  # noqa: BLE001
            continue
        am = [int(v) for v in getattr(h, "AcquisitionMatrix", []) or []]
        ped = str(getattr(h, "InPlanePhaseEncodingDirection", "") or "").upper()
        rec = dict(plane=s.Anatomical_Plane, fluid=s.Fluid_Sensitive, ps=(ps_r + ps_c) / 2,
                   thick=num(getattr(h, "SliceThickness", None)), spacing=num(getattr(h, "SpacingBetweenSlices", None)),
                   acq=str(getattr(h, "MRAcquisitionType", "") or ""),
                   field=num(getattr(h, "MagneticFieldStrength", None)),
                   vendor=str(getattr(h, "Manufacturer", "") or "").split(" ")[0].upper(),
                   d_freq=np.nan, d_phase=np.nan, zf_freq=np.nan, zf_phase=np.nan)
        if len(am) == 4 and ped in ("ROW", "COL") and max(am[0], am[1]) > 0 and max(am[2], am[3]) > 0:
            n_freq, n_phase = max(am[0], am[1]), max(am[2], am[3])
            if ped == "COL":   # phase along the vertical image axis (rows), frequency along the horizontal one
                fov_freq, px_freq = n_c * ps_c, n_c
                fov_phase_img, px_phase = n_r * ps_r, n_r
            else:              # ROW: phase along the horizontal axis
                fov_freq, px_freq = n_r * ps_r, n_r
                fov_phase_img, px_phase = n_c * ps_c, n_c
            pfov = num(getattr(h, "PercentPhaseFieldOfView", None))
            fov_phase = min(fov_phase_img, fov_freq * pfov / 100.0) if np.isfinite(pfov) and pfov > 0 else fov_phase_img
            rec.update(d_freq=fov_freq / n_freq, d_phase=fov_phase / n_phase, zf_freq=px_freq / n_freq,
                       zf_phase=(px_phase * fov_phase / fov_phase_img) / n_phase)
        rows.append(rec)
df = pd.DataFrame(rows)
ok = df.dropna(subset=["d_freq", "d_phase"])
print(f"read {len(df)} series from {n_studies} studies in {time.time() - t0:.0f}s; "
      f"usable AcquisitionMatrix + phase direction: {len(ok)} ({len(ok) / len(df):.1%})")
q = [.1, .25, .5, .75, .9]
print("reconstructed pixel spacing (mm) quantiles:", ok.ps.quantile(q).round(3).tolist())
print("acquired spacing, frequency axis (mm):     ", ok.d_freq.quantile(q).round(3).tolist())
print("acquired spacing, phase axis (mm):         ", ok.d_phase.quantile(q).round(3).tolist())
print("zero-fill factor, frequency axis:          ", ok.zf_freq.quantile(q).round(2).tolist())
print("zero-fill factor, phase axis:              ", ok.zf_phase.quantile(q).round(2).tolist())
print(f"series reconstructed finer than acquired (zero-fill > 1.05 on the frequency axis): {(ok.zf_freq > 1.05).mean():.1%}")
for name, g in GRIDS.items():
    nominal = ok.ps < g
    real_f = ok.d_freq < g
    real_both = (ok.d_freq < g) & (ok.d_phase < g)
    print(f"[{name}] nominally downsampled (PixelSpacing finer than grid): {nominal.mean():.1%} | "
          f"acquired frequency spacing finer than grid: {real_f.mean():.1%} | both axes finer: {real_both.mean():.1%} | "
          f"real downsampling factor on the frequency axis among those: "
          f"{(g / ok.d_freq[real_f]).median() if real_f.any() else float('nan'):.2f} (median)")
for p, g in ok.groupby("plane"):
    print(f"  {p:9s} n={len(g):4d} | acquired freq median {g.d_freq.median():.3f} mm, phase median {g.d_phase.median():.3f} mm"
          f" | real sub-grid (freq < 0.365): {(g.d_freq < GRIDS['cache v2 0.365 mm']).mean():.1%}")
gap = df.spacing - df.thick
print("slice thickness (mm) quantiles:", df.thick.quantile(q).round(2).tolist())
print("slice gap = spacing - thickness (mm) quantiles:", gap.quantile(q).round(2).tolist(),
      f"| gap > 0.5 mm: {(gap > 0.5).mean():.1%} | missing spacing: {df.spacing.isna().mean():.1%}")
print("MRAcquisitionType:", df.acq.value_counts().to_dict())
print("field strength (T):", df.field.round(1).value_counts().to_dict())
print("manufacturer:", df.vendor.value_counts().to_dict())
