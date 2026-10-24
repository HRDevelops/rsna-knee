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

"""Nominal acquisition-matrix spacing vs reconstructed pixel spacing, plus header facts that the v1.3 plan needs.
Header-only, one header per series. Used for docs/research/PLAN_v1_3.md s1.3.

    RSNA_ROOT=... python scripts/audit_acquisition_resolution.py 600

AcquisitionMatrix (0018,1310) = dimensions of the acquired frequency/phase data before reconstruction (two of its four
values are non-zero); InPlanePhaseEncodingDirection says which image axis is phase (COL = vertical, ROW = horizontal).
"Nominal acquisition-matrix spacing" per axis = field of view along that axis / matrix samples. It is NOT the effective
resolution: partial Fourier, sampling percentage, parallel imaging, k-space apodization and reconstruction filters are
not (or only partly) visible in the headers, and they can only make the effective resolution coarser. A rectangular
phase field of view (PercentPhaseFieldOfView < 100) is handled by taking the smaller of the image extent and the
frequency field of view times that percentage.

PercentSampling (0018,0093) = fraction of acquisition-matrix lines acquired. Check: does the matrix phase count already
reflect it? ratio = phase samples / (frequency samples x PercentPhaseFieldOfView / 100), compared with
PercentSampling / 100 per vendor. Partial Fourier: ScanOptions defined terms PFP / PFF. ReconstructionDiameter is read
if present (it is not in this dataset's tag allowlist).

Also: plane label vs orientation, same-plane multi-series per study, and the share of tokens that would be mostly
padding on the 160 mm / 384 px cache grid (centred crop or pad, as in the current cache). Grid: 0.4167 mm = exp018's
pixel size = cache v2 (160 mm at 384 px). Same random sample rule as check_series_geometry.py (seed 0).
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
n_studies = int(sys.argv[1]) if len(sys.argv) > 1 else 600
sdf = pd.read_csv(root + "/train_series.csv")
uids = sorted(sdf.StudyInstanceUID.unique())
random.seed(0)
pick = random.sample(uids, n_studies)
TAGS = ["Rows", "Columns", "PixelSpacing", "AcquisitionMatrix", "InPlanePhaseEncodingDirection",
        "PercentPhaseFieldOfView", "PercentSampling", "SliceThickness", "SpacingBetweenSlices", "MRAcquisitionType",
        "MagneticFieldStrength", "Manufacturer", "ReconstructionDiameter", "ScanOptions", "ImageOrientationPatient"]
GRID = 160.0 / 384          # 0.4167 mm: exp018's pixel size and the cache-v2 grid
FIELD = 160.0               # cache-v2 field of view (mm), centred crop or pad
VENDOR = {"SIEMENS": "Siemens", "PHILIPS": "Philips", "GE": "GE", "GEHC": "GE", "TOSHIBA": "Canon/Toshiba",
          "CANON": "Canon/Toshiba", "CANON_MEC": "Canon/Toshiba", "FUJIFILM": "Fujifilm/Hitachi",
          "HITACHI": "Fujifilm/Hitachi"}
AXIS_PLANE = {0: "Sagittal", 1: "Coronal", 2: "Axial"}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def low_validity_share(fov_w, fov_h, g):
    """Share of the g x g tokens whose cell is < 50% real image when an image of fov_w x fov_h mm is centre-cropped or
    centre-padded to FIELD x FIELD mm."""
    edges = np.linspace(-FIELD / 2, FIELD / 2, g + 1)

    def frac(fov):
        lo, hi = max(-fov / 2, -FIELD / 2), min(fov / 2, FIELD / 2)
        return np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None) / (edges[1] - edges[0])

    v = np.outer(frac(fov_h), frac(fov_w))
    return float((v < 0.5).mean())


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
        try:
            iop = np.array([float(v) for v in h.ImageOrientationPatient])
            normal_axis = int(np.argmax(np.abs(np.cross(iop[:3], iop[3:]))))
        except Exception:  # noqa: BLE001
            normal_axis = -1
        am = [int(v) for v in getattr(h, "AcquisitionMatrix", []) or []]
        ped = str(getattr(h, "InPlanePhaseEncodingDirection", "") or "").upper()
        opts = getattr(h, "ScanOptions", "") or ""
        opts = {str(o).upper() for o in (opts if isinstance(opts, (list, pydicom.multival.MultiValue)) else [opts])}
        raw_vendor = str(getattr(h, "Manufacturer", "") or "").strip().upper().split(" ")[0]
        rec = dict(study=u, plane=s.Anatomical_Plane, ps=(ps_r + ps_c) / 2, fov_w=n_c * ps_c, fov_h=n_r * ps_r,
                   thick=num(getattr(h, "SliceThickness", None)), spacing=num(getattr(h, "SpacingBetweenSlices", None)),
                   acq=str(getattr(h, "MRAcquisitionType", "") or ""), field=num(getattr(h, "MagneticFieldStrength", None)),
                   vendor=VENDOR.get(raw_vendor, raw_vendor.title() or "missing"),
                   recon_diam=num(getattr(h, "ReconstructionDiameter", None)),
                   pct_sampling=num(getattr(h, "PercentSampling", None)),
                   pct_phase_fov=num(getattr(h, "PercentPhaseFieldOfView", None)),
                   pfp="PFP" in opts, pff="PFF" in opts,
                   plane_from_normal=AXIS_PLANE.get(normal_axis, "unknown"),
                   nominal_acq_spacing_freq=np.nan, nominal_acq_spacing_phase=np.nan,
                   zf_freq=np.nan, zf_phase=np.nan, phase_ratio=np.nan)
        if len(am) == 4 and ped in ("ROW", "COL") and max(am[0], am[1]) > 0 and max(am[2], am[3]) > 0:
            n_freq, n_phase = max(am[0], am[1]), max(am[2], am[3])
            if ped == "COL":   # phase along the vertical image axis (rows), frequency along the horizontal one
                fov_freq, px_freq = n_c * ps_c, n_c
                fov_phase_img, px_phase = n_r * ps_r, n_r
            else:              # ROW: phase along the horizontal axis
                fov_freq, px_freq = n_r * ps_r, n_r
                fov_phase_img, px_phase = n_c * ps_c, n_c
            pfov = rec["pct_phase_fov"]
            fov_phase = min(fov_phase_img, fov_freq * pfov / 100.0) if np.isfinite(pfov) and pfov > 0 else fov_phase_img
            rec.update(nominal_acq_spacing_freq=fov_freq / n_freq, nominal_acq_spacing_phase=fov_phase / n_phase,
                       zf_freq=px_freq / n_freq, zf_phase=(px_phase * fov_phase / fov_phase_img) / n_phase,
                       phase_ratio=(n_phase / (n_freq * pfov / 100.0)) if np.isfinite(pfov) and pfov > 0 else np.nan)
        rows.append(rec)
df = pd.DataFrame(rows)
ok = df.dropna(subset=["nominal_acq_spacing_freq", "nominal_acq_spacing_phase"])
print(f"read {len(df)} series from {n_studies} studies in {time.time() - t0:.0f}s; "
      f"usable AcquisitionMatrix + phase direction: {len(ok)} ({len(ok) / len(df):.1%}); "
      f"ReconstructionDiameter present: {df.recon_diam.notna().mean():.1%}")
q = [.1, .25, .5, .75, .9]
print("reconstructed pixel spacing (mm) quantiles:          ", ok.ps.quantile(q).round(3).tolist())
print("nominal acquisition-matrix spacing, frequency (mm):  ", ok.nominal_acq_spacing_freq.quantile(q).round(3).tolist())
print("nominal acquisition-matrix spacing, phase (mm):      ", ok.nominal_acq_spacing_phase.quantile(q).round(3).tolist())
print("reconstruction / matrix factor, frequency, phase:    ", ok.zf_freq.quantile(q).round(2).tolist(),
      ok.zf_phase.quantile(q).round(2).tolist())

print(f"\n[grid {GRID:.4f} mm] per vendor: n | recon finer than matrix (freq factor > 1.05) | pixel finer than grid | "
      f"matrix spacing finer than grid (freq) | (both axes) | median matrix spacing freq/phase")


def vendor_line(name, g):
    return (f"  {name:17s} n={len(g):4d} | {(g.zf_freq > 1.05).mean():6.1%} | {(g.ps < GRID).mean():6.1%} | "
            f"{(g.nominal_acq_spacing_freq < GRID).mean():6.1%} | "
            f"{((g.nominal_acq_spacing_freq < GRID) & (g.nominal_acq_spacing_phase < GRID)).mean():6.1%} | "
            f"{g.nominal_acq_spacing_freq.median():.3f} / {g.nominal_acq_spacing_phase.median():.3f}")


print(vendor_line("ALL", ok))
for v, g in ok.groupby("vendor"):
    print(vendor_line(v, g))

print("\nPercentSampling vs the matrix (does the phase count already include it?) and partial Fourier, per vendor:")
print("  vendor            PS present | PS median [p10, p90] | |phase ratio - PS/100| <= 0.03 | phase ratio median | "
      "ScanOptions PFP | PFF")
for v, g in df.groupby("vendor"):
    gp = g.dropna(subset=["pct_sampling"])
    agree = (gp.phase_ratio - gp.pct_sampling / 100).abs() <= 0.03
    agree_share = agree[gp.phase_ratio.notna()].mean() if gp.phase_ratio.notna().any() else float("nan")
    print(f"  {v:17s} {g.pct_sampling.notna().mean():6.1%} | "
          f"{gp.pct_sampling.median():5.1f} [{gp.pct_sampling.quantile(.1):5.1f}, {gp.pct_sampling.quantile(.9):5.1f}] | "
          f"{agree_share:6.1%} | {g.phase_ratio.median():.3f} | {g.pfp.mean():6.1%} | {g.pff.mean():6.1%}")
# If the matrix did NOT already include PercentSampling, the phase spacing would be coarser by 100/PS. Share of series
# whose phase spacing is finer than the grid under each reading:
alt = ok.nominal_acq_spacing_phase * 100.0 / ok.pct_sampling.where(ok.pct_sampling > 0)
print(f"phase spacing finer than the grid: matrix as given {(ok.nominal_acq_spacing_phase < GRID).mean():.1%}; "
      f"if PercentSampling were applied on top {(alt < GRID).mean():.1%} (of series with PS)")

print("\nplane label vs orientation (dominant axis of the slice normal):",
      f"{(df.plane == df.plane_from_normal).mean():.1%} agree")
per = df.groupby(["study", "plane"]).size().unstack(fill_value=0)
print("studies with >= 2 series in the same plane:",
      {p: f"{(per[p] >= 2).mean():.1%}" for p in per.columns}, f"| any plane: {(per >= 2).any(axis=1).mean():.1%}")

print(f"\ntokens that are mostly padding (< 50% real image) on the {FIELD:.0f} mm grid, mean over series:")
for g in (4, 6, 12):
    share = np.mean([low_validity_share(w, hh, g) for w, hh in zip(df.fov_w, df.fov_h)])
    print(f"  {g}x{g}: {share:.1%}")
print(f"series with field of view < {FIELD:.0f} mm on some axis (some padding): "
      f"{((df.fov_w < FIELD) | (df.fov_h < FIELD)).mean():.1%}")

gap = df.spacing - df.thick
print("\nslice thickness (mm) quantiles:", df.thick.quantile(q).round(2).tolist())
print("slice gap = spacing - thickness (mm) quantiles:", gap.quantile(q).round(2).tolist(),
      f"| gap > 0.5 mm: {(gap > 0.5).mean():.1%} | missing spacing: {df.spacing.isna().mean():.1%}")
print("MRAcquisitionType:", df.acq.value_counts().to_dict())
print("field strength (T):", df.field.round(1).value_counts().to_dict())
print("vendor:", df.vendor.value_counts().to_dict())
