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

"""Feasibility check for geometry-based multi-plane fusion (header-only, no pixels). Used for docs/research/PLAN_v1.md s1.

    RSNA_ROOT=... python scripts/check_series_geometry.py 150

For a random sample of training studies: do all series share one FrameOfReferenceUID, and do the series' 3D bounding
boxes (from ImagePositionPatient / ImageOrientationPatient / PixelSpacing / Rows / Columns) overlap in patient space?
Also: 2D vs 3D acquisition, slice gap (SpacingBetweenSlices - SliceThickness), and the medial-lateral offset of the
study centre (used for laterality).
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
n_studies = int(sys.argv[1]) if len(sys.argv) > 1 else 150
sdf = pd.read_csv(root + "/train_series.csv")
uids = sorted(sdf.StudyInstanceUID.unique())
random.seed(0)
pick = random.sample(uids, n_studies)
TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns", "SliceThickness",
        "SpacingBetweenSlices", "FrameOfReferenceUID", "MRAcquisitionType"]
t0 = time.time()
rows = []
for u in pick:
    for _, s in sdf[sdf.StudyInstanceUID == u].iterrows():
        files = glob.glob(f"{root}/train_series/{u}/{s.SeriesInstanceUID}/*.dcm")
        if not files:
            continue
        corners, fors, th, sp, acq = [], set(), None, None, None
        for f in files:
            h = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=TAGS)
            o = np.array(h.ImagePositionPatient, float)
            iop = np.array(h.ImageOrientationPatient, float)
            r, c = iop[:3], iop[3:]
            dr, dc = (float(v) for v in h.PixelSpacing)        # row spacing, column spacing
            W, Hh = int(h.Columns), int(h.Rows)
            for a, b in ((0, 0), (W - 1, 0), (0, Hh - 1), (W - 1, Hh - 1)):
                corners.append(o + a * dc * r + b * dr * c)
            fors.add(str(getattr(h, "FrameOfReferenceUID", "")))
            th = float(getattr(h, "SliceThickness", np.nan) or np.nan)
            sp = float(getattr(h, "SpacingBetweenSlices", np.nan) or np.nan)
            acq = str(getattr(h, "MRAcquisitionType", ""))
        C = np.array(corners)
        rows.append(dict(study=u, series=s.SeriesInstanceUID, plane=s.Anatomical_Plane, n=len(files),
                         lo=C.min(0), hi=C.max(0), fors="|".join(sorted(fors)), thick=th, spacing=sp, acq=acq))
df = pd.DataFrame(rows)
print(f"read {len(df)} series from {df.study.nunique()} studies in {time.time() - t0:.0f}s")

same_for, all_overlap, min_iou, centre_spread, mid_x = [], [], [], [], []
for u, g in df.groupby("study"):
    fs = set("|".join(g.fors).split("|")) - {""}
    same_for.append(len(fs) == 1)
    lo = np.max(np.stack(g.lo.values), 0)
    hi = np.min(np.stack(g.hi.values), 0)
    inter = np.clip(hi - lo, 0, None)
    all_overlap.append(bool((inter > 0).all()))
    ious = []
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a_lo, a_hi, b_lo, b_hi = g.lo.iloc[i], g.hi.iloc[i], g.lo.iloc[j], g.hi.iloc[j]
            it = np.prod(np.clip(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo), 0, None))
            un = np.prod(a_hi - a_lo) + np.prod(b_hi - b_lo) - it
            ious.append(it / un if un > 0 else 0)
    min_iou.append(min(ious) if ious else np.nan)
    ctr = np.stack([(a + b) / 2 for a, b in zip(g.lo, g.hi)])
    centre_spread.append(float(np.linalg.norm(ctr - ctr.mean(0), axis=1).max()))
    mid_x.append(float(ctr[:, 0].mean()))

print(f"studies with ONE FrameOfReferenceUID across all series: {np.mean(same_for):.1%}")
print(f"studies whose series boxes have a common 3D intersection: {np.mean(all_overlap):.1%}")
print("min pairwise box IoU per study, quantiles:", np.nanquantile(min_iou, [.05, .25, .5]).round(3).tolist())
print("max distance of a series centre from the study mean (mm), quantiles:",
      np.quantile(centre_spread, [.5, .9, .99]).round(1).tolist())
print("study centre |x| (mm) >= 20 (side readable from geometry):", f"{np.mean(np.abs(mid_x) >= 20):.1%}")
print("MRAcquisitionType:", df.acq.value_counts().to_dict())
gap = (df.spacing - df.thick)
print("slice gap (spacing - thickness, mm) quantiles:", gap.quantile([.1, .5, .9]).round(2).to_dict(),
      "| missing spacing:", f"{df.spacing.isna().mean():.1%}")
