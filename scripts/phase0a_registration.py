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

"""Phase 0A of docs/research/PLAN_v1_3.md: is the retained DICOM geometry consistent across the separately acquired
series of a study? CPU only. One shard of the study sample per call; scripts/phase0a_report.py applies the gates.

    RSNA_ROOT=... python scripts/phase0a_registration.py --n-studies 150 --shard 0 --n-shards 15 --out DIR [--qc]

Per study (same random sample rule as check_series_geometry.py, seed 0):
  * one series per plane family with the cache's slot rules (wmil.pick_slots: Sagittal FS / non-FS, Coronal FS /
    non-FS, Axial), each placed in patient space from its own DICOM geometry (IPP / IOP / PixelSpacing; slices ordered
    by IPP on the normal, as in the cache), binned in-plane to ~1 mm;
  * every pair of families: rigid 6-DOF Mattes mutual-information registration around the DICOM placement (identity in
    patient coordinates). The fixed image is the family earlier in the slot order. The 4x4 result maps fixed-domain
    points to moving-domain points, so a perfectly consistent pair gives the identity;
  * robust local search (pilot 1, 2026-10-25: single-start MI drifted up to 48 mm / 20 deg on partially overlapping
    thick-slice stacks while the DICOM placement was visibly correct; pilot 2: optimizing on a random 30% sample but
    selecting on all voxels made the two objectives disagree; pilot 3 and a benchmark: regular-step gradient descent
    stalls in the rotation-translation valley, Powell does not). The metric is Mattes MI over ALL voxels of a fixed mask:
    fixed foreground within 45 mm of the study centre and at least 10 mm inside the moving stack. Powell runs from the
    DICOM placement and from 2 perturbed starts (3 mm / 2 deg) on that same objective; the unmoved DICOM placement is
    also a candidate; candidates that push > 20% of the samples out of the moving stack are rejected; lowest MI wins;
  * cross-plane pairs only: injected-error recovery. The moving stack's geometry is moved by a known rigid transform P
    (5 mm / 3 deg and 10 mm / 5 deg, random axis and direction, about the study centre) and the same search is run;
    the expected result is P o T0, the error is the residual of (P o T0)^-1 o T';
  * all metrics at 9 probe points: the centre m of the study's common box intersection and the corners of that box
    scaled by 0.5 about m. Displacements are also given in footprint units d (plan s2.2: 6x6 tokens on the 160 mm grid,
    triplets of native-adjacent slices), d^2 = delta^T (Sigma_A + Sigma_B)^-1 delta.
Cycle consistency (triangles) is computed in the report from the stored matrices.

Outputs in DIR: pairs_<shard>.parquet, series_<shard>.parquet, qc/*.png (with --qc: fixed centre slice with the moving
stack resampled at the DICOM placement and at the selected transform, for pairs over tolerance plus one reference pair
per study; image data, local only, never pushed).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee import wmil  # noqa: E402
from rsna_knee.preprocess import _order_from_headers  # noqa: E402

FAMILIES = [s[0] for s in wmil.SLOT_SPECS]          # slot order = fixed-image priority
PLANE = {s[0]: s[1] for s in wmil.SLOT_SPECS}
INPLANE_MM = 1.0                                    # registration grid: ~1 mm in-plane, native slice spacing
CELL_MM = 160.0 / 6                                 # 6x6 token cell on the 160 mm cache grid (plan s2.2)
INJECT = [(5.0, 3.0), (10.0, 5.0)]                  # (mm, deg)
TOL_MM, TOL_DEG = 3.0, 2.0                          # QC trigger, same numbers as the raw-geometry gate
ROI_MM = 45.0                                       # metric region: sphere around the study centre (the joint)
N_STARTS = 2                                        # perturbed starts besides the DICOM placement
START_MM, START_DEG = 3.0, 2.0
MIN_MASK_VOXELS = 3000


# ------------------------------------------------------------------------------------------ series loading
def load_series(series_dir: Path) -> tuple[sitk.Image, dict]:
    files = sorted(glob.glob(str(series_dir / "*.dcm")))
    headers = [pydicom.dcmread(f, stop_before_pixels=True, force=True) for f in files]
    order, ometa = _order_from_headers(headers)
    files = [files[i] for i in order]
    headers = [headers[i] for i in order]
    iop = np.array([float(v) for v in headers[0].ImageOrientationPatient])
    r, c = iop[:3] / np.linalg.norm(iop[:3]), iop[3:] / np.linalg.norm(iop[3:])
    n = np.cross(r, c)
    iop_dev = max(float(np.abs(np.array([float(v) for v in h.ImageOrientationPatient]) - iop).max()) for h in headers)
    ipp = np.array([[float(v) for v in h.ImagePositionPatient] for h in headers])
    proj = ipp @ n
    keep = np.concatenate([[True], np.diff(proj) > 1e-3])          # drop repeated positions (keep the first)
    files = [f for f, k in zip(files, keep) if k]
    headers = [h for h, k in zip(headers, keep) if k]
    ipp, proj = ipp[keep], proj[keep]
    if len(files) < 3:
        raise ValueError(f"only {len(files)} distinct slice positions")
    gaps = np.diff(proj)
    sz = float(np.median(gaps))
    nonuniform = float(np.max(np.abs(gaps - sz)) / sz)
    off = ipp - ipp[0]
    shear = float(np.max(np.linalg.norm(off - np.outer(off @ n, n), axis=1)))   # in-plane drift of slice origins
    ps_r, ps_c = (float(v) for v in headers[0].PixelSpacing)
    vol = np.stack([pydicom.dcmread(f, force=True).pixel_array.astype(np.float32) for f in files])
    lo, hi = np.percentile(vol, [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    img = sitk.GetImageFromArray(vol)                              # index (x=column, y=row, z=slice)
    img.SetSpacing((ps_c, ps_r, sz))
    img.SetOrigin(tuple(float(v) for v in ipp[0]))
    img.SetDirection(tuple(float(v) for v in np.column_stack([r, c, n]).ravel()))
    f = max(1, int(round(INPLANE_MM / ((ps_r + ps_c) / 2))))
    if f > 1:
        img = sitk.BinShrink(img, [f, f, 1])
    h0 = headers[0]
    meta = dict(n_slices=len(files), ps_row=ps_r, ps_col=ps_c, slice_spacing=sz, nonuniform=nonuniform,
                shear_mm=shear, iop_dev=iop_dev, irregular=bool(ometa.get("irregular_spacing", False)),
                thickness=float(getattr(h0, "SliceThickness", np.nan) or np.nan),
                vendor=str(getattr(h0, "Manufacturer", "") or "").strip().upper().split(" ")[0],
                field=float(getattr(h0, "MagneticFieldStrength", np.nan) or np.nan),
                rx=r.tolist(), cx=c.tolist(), nx=n.tolist(),
                corners=_corners(ipp, r, c, n, ps_r, ps_c, int(h0.Rows), int(h0.Columns)).tolist())
    return img, meta


def _corners(ipp, r, c, n, ps_r, ps_c, rows, cols):
    pts = []
    for o in (ipp[0], ipp[-1]):
        for a, b in ((0, 0), (cols - 1, 0), (0, rows - 1), (cols - 1, rows - 1)):
            pts.append(o + a * ps_c * r + b * ps_r * c)
    return np.array(pts)


# ------------------------------------------------------------------------------------------ geometry helpers
def voxel_points(img: sitk.Image) -> np.ndarray:
    """Physical points of every voxel, array order (z, y, x) flattened -> (N, 3)."""
    sx, sy, sz = img.GetSize()
    D = np.array(img.GetDirection()).reshape(3, 3)
    sp = np.array(img.GetSpacing())
    k, j, i = np.meshgrid(np.arange(sz), np.arange(sy), np.arange(sx), indexing="ij")
    idx = np.stack([i.ravel(), j.ravel(), k.ravel()], 1).astype(np.float64)
    return np.array(img.GetOrigin()) + (idx * sp) @ D.T


def inside(img: sitk.Image, pts: np.ndarray, margin_mm: float) -> np.ndarray:
    D = np.array(img.GetDirection()).reshape(3, 3)
    sp = np.array(img.GetSpacing())
    q = ((pts - np.array(img.GetOrigin())) @ D) / sp               # continuous index (x, y, z)
    size = np.array(img.GetSize()) - 1
    m = np.array([margin_mm / sp[0], margin_mm / sp[1], 0.5])
    return np.all((q >= m) & (q <= size - m), axis=1)


def fixed_mask(fixed: sitk.Image, centre: np.ndarray, *movings: sitk.Image) -> tuple[sitk.Image, int]:
    """Fixed-image foreground within ROI_MM of the study centre and >= 10 mm inside EVERY given moving stack, so that
    small moves do not change the overlap."""
    pts = voxel_points(fixed)
    arr = sitk.GetArrayFromImage(fixed)
    keep = (arr.ravel() > 0.1) & inside(fixed, pts, 2.0) & (np.linalg.norm(pts - centre, axis=1) <= ROI_MM)
    for mv in movings:
        keep &= inside(mv, pts, 10.0)
    mask = sitk.GetImageFromArray(keep.reshape(arr.shape).astype(np.uint8))
    mask.CopyInformation(fixed)
    return mask, int(keep.sum())


def to_matrix(tx: sitk.Transform, m: np.ndarray) -> np.ndarray:
    """4x4 homogeneous matrix of a rigid sitk transform, fitted from 4 points (exact for rigid/affine maps)."""
    p = [m, m + [50.0, 0, 0], m + [0, 50.0, 0], m + [0, 0, 50.0]]
    q = [np.array(tx.TransformPoint(tuple(float(v) for v in x))) for x in p]
    A = np.column_stack([(q[i] - q[0]) / 50.0 for i in (1, 2, 3)])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = A, q[0] - A @ m
    return M


def euler(M: np.ndarray, centre: np.ndarray) -> sitk.Euler3DTransform:
    """Euler3DTransform centred at `centre` equal to the rigid 4x4 M (rotation re-orthonormalized)."""
    U, _, Vt = np.linalg.svd(M[:3, :3])
    R = U @ Vt
    t = sitk.Euler3DTransform()
    t.SetCenter(tuple(float(v) for v in centre))
    t.SetMatrix(tuple(float(v) for v in R.ravel()))
    t.SetTranslation(tuple(float(v) for v in M[:3, 3] + R @ centre - centre))
    return t


def deviation(M: np.ndarray, probes: np.ndarray, SA: np.ndarray | None = None, SB: np.ndarray | None = None) -> dict:
    """How far M is from the identity: rotation angle, max / centre displacement at the probes, max footprint d."""
    R = M[:3, :3]
    ang = math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    disp = probes @ R.T + M[:3, 3] - probes
    out = dict(rot_deg=ang, disp_max_mm=float(np.linalg.norm(disp, axis=1).max()),
               disp_centre_mm=float(np.linalg.norm(disp[0])))
    if SA is not None:
        Si = np.linalg.inv(SA + SB + np.eye(3))
        out["d_max"] = float(np.sqrt(np.max(np.einsum("ni,ij,nj->n", disp, Si, disp))))
    return out


def footprint_cov(meta: dict) -> np.ndarray:
    """Plan s2.2 at 6x6 on the 160 mm grid: in-plane cell variance + native-adjacent triplet spread along the normal."""
    r, c, n = (np.array(meta[k]) for k in ("rx", "cx", "nx"))
    tau = meta["thickness"] if np.isfinite(meta["thickness"]) else meta["slice_spacing"]
    v_n = (2.0 / 3.0) * meta["slice_spacing"] ** 2 + tau ** 2 / 12.0
    return CELL_MM ** 2 / 12.0 * (np.outer(r, r) + np.outer(c, c)) + v_n * np.outer(n, n)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def rigid(axis: np.ndarray, deg: float, t: np.ndarray, centre: np.ndarray) -> np.ndarray:
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = math.radians(deg)
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, (np.eye(3) - R) @ centre + t
    return M


def moved(img: sitk.Image, P: np.ndarray) -> sitk.Image:
    """Same voxels, geometry moved by P: every voxel's physical point x becomes P(x)."""
    out = sitk.Image(img)
    D = np.array(img.GetDirection()).reshape(3, 3)
    o = np.array(img.GetOrigin())
    out.SetOrigin(tuple(float(v) for v in P[:3, :3] @ o + P[:3, 3]))
    out.SetDirection(tuple(float(v) for v in (P[:3, :3] @ D).ravel()))
    return out


# ------------------------------------------------------------------------------------------ registration
# One objective everywhere: Mattes MI over ALL fixed-mask voxels (no random sampling), so the optimizer and the
# candidate selection agree (pilot 2 used 30% random sampling to optimize but all voxels to select, which made the
# unmoved placement win on a technicality and broke the injected-error test). No moving mask: the fixed sample set
# then stays the same for every transform, which removes the overlap bias; the overlap guard below rejects candidates
# that push samples out of the moving stack (pilot 2's 113 mm outlier).
MIN_OVERLAP = 0.8                                   # of the DICOM placement's in-stack sample fraction


def _method(fmask: sitk.Image) -> sitk.ImageRegistrationMethod:
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.NONE)
    R.SetMetricFixedMask(fmask)
    R.SetInterpolator(sitk.sitkLinear)
    return R


def exact_mi(fixed, moving, fmask, M: np.ndarray, centre: np.ndarray) -> float:
    """Mattes MI over all fixed-mask voxels at transform M (negative; lower is better)."""
    R = _method(fmask)
    R.SetInitialTransform(euler(M, centre))
    return float(R.MetricEvaluate(fixed, moving))


def overlap(moving: sitk.Image, mask_pts: np.ndarray, M: np.ndarray) -> float:
    """Share of the fixed-mask points that land inside the moving stack under M."""
    return float(inside(moving, mask_pts @ M[:3, :3].T + M[:3, 3], 0.0).mean())


def optimize(fixed, moving, fmask, centre: np.ndarray, start: np.ndarray):
    """Powell (Brent line searches). Benchmark 2026-10-25 (job 33420, real pilot pairs, injected 5 mm / 3 deg and
    10 mm / 5 deg): Powell returned to the pair's MI minimum in every case (0.00 mm / 0.00 deg), while regular-step
    gradient descent stalled in the rotation-translation valley (1.7-25 mm off, final MI worse than the start) and
    conjugate-gradient line search was erratic (1.4-72 mm)."""
    R = _method(fmask)
    R.SetOptimizerAsPowell(numberOfIterations=100, maximumLineIterations=100, stepLength=1.0,
                           stepTolerance=1e-4, valueTolerance=1e-7)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([1, 1, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(euler(start, centre), inPlace=False)
    out = R.Execute(fixed, moving)
    return to_matrix(out, centre), int(R.GetOptimizerIteration())


def register(fixed, moving, fmask, centre: np.ndarray, probes: np.ndarray, rng: np.random.Generator, seed: int):
    """Multi-start local search on one exact objective; the unmoved DICOM placement is also a candidate. Candidates
    that lose > 20% of the in-stack overlap are rejected; the lowest MI wins. `seed` is unused (kept for the API)."""
    pts = voxel_points(fixed)[sitk.GetArrayFromImage(fmask).ravel() > 0]
    starts = [np.eye(4)] + [rigid(rng.normal(size=3), START_DEG, START_MM * unit(rng.normal(size=3)), centre)
                            for _ in range(N_STARTS)]
    cands = [("dicom", np.eye(4), 0)]
    for i, S in enumerate(starts):
        try:
            M, it = optimize(fixed, moving, fmask, centre, S)
            cands.append((f"start{i}", M, it))
        except RuntimeError:                                        # e.g. too many samples outside the moving stack
            continue
    ov0 = overlap(moving, pts, np.eye(4))
    scored = []
    for name, M, it in cands:
        ov = overlap(moving, pts, M)
        if name == "dicom" or ov >= MIN_OVERLAP * ov0:
            scored.append((exact_mi(fixed, moving, fmask, M, centre), name, M, it, ov))
    scored.sort(key=lambda s: s[0])
    mi_best, best, M_best, it_best, ov_best = scored[0]
    mi_dicom = next(s[0] for s in scored if s[1] == "dicom")
    opt = [s for s in scored if s[1] != "dicom"]
    if opt:                                                         # do the optimized starts reach one basin?
        ref = opt[0][2]
        agree = sum(deviation(np.linalg.inv(ref) @ s[2], probes)["disp_max_mm"] <= 1.0 for s in opt)
    else:
        agree = 0
    return M_best, dict(best=best, mi_dicom=mi_dicom, mi_best=mi_best, n_starts_ok=len(opt),
                        n_rejected_overlap=len(cands) - len(scored), n_agree_1mm=int(agree), iterations=it_best,
                        overlap_ratio=ov_best / max(ov0, 1e-9))


def qc_png(fixed: sitk.Image, moving: sitk.Image, tx: sitk.Transform, path: Path) -> None:
    from PIL import Image
    f = sitk.GetArrayFromImage(fixed)
    k = f.shape[0] // 2
    panels = []
    for t in (sitk.Euler3DTransform(), tx):
        mv = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, t, sitk.sitkLinear, 0.0))
        rgb = np.stack([mv[k], f[k], mv[k]], -1)                  # fixed green, moving magenta
        panels.append((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
    Image.fromarray(np.concatenate([panels[0], gap, panels[1]], 1)).save(path)


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-studies", type=int, default=150)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qc", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--max-studies", type=int, default=0, help="pilot: stop after this many studies of the shard")
    a = ap.parse_args()
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(a.threads)
    raw = Path(os.environ["RSNA_ROOT"]) / "data" / "raw"
    out = Path(a.out)
    (out / "qc").mkdir(parents=True, exist_ok=True)
    sdf = pd.read_csv(raw / "train_series.csv")
    uids = sorted(sdf.StudyInstanceUID.unique())
    random.seed(0)
    pick = random.sample(uids, a.n_studies)
    mine = pick[a.shard::a.n_shards]
    if a.max_studies:
        mine = mine[:a.max_studies]
    pair_rows, series_rows = [], []
    t_start = time.time()
    for si, u in enumerate(mine):
        t0 = time.time()
        rows = []
        for _, s in sdf[sdf.StudyInstanceUID == u].iterrows():
            nf = len(glob.glob(str(raw / "train_series" / u / s.SeriesInstanceUID / "*.dcm")))
            if nf:
                rows.append(dict(series_uid=s.SeriesInstanceUID, plane=s.Anatomical_Plane,
                                 fluid=int(s.Fluid_Sensitive), n_slices=nf))
        chosen = wmil.pick_slots(rows)
        imgs, metas = {}, {}
        for i, r in chosen.items():
            fam = FAMILIES[i]
            try:
                imgs[fam], metas[fam] = load_series(raw / "train_series" / u / r["series_uid"])
                series_rows.append(dict(study=u, family=fam, plane=PLANE[fam], series_uid=r["series_uid"], ok=True,
                                        **{k: v for k, v in metas[fam].items() if k not in ("corners",)}))
            except Exception as e:  # noqa: BLE001
                series_rows.append(dict(study=u, family=fam, plane=PLANE[fam], series_uid=r["series_uid"], ok=False,
                                        error=f"{type(e).__name__}: {e}"[:200]))
        fams = [f for f in FAMILIES if f in imgs]
        if len(fams) < 2:
            continue
        allc = np.concatenate([np.array(metas[f]["corners"]) for f in fams])
        lo = np.max([np.array(metas[f]["corners"]).min(0) for f in fams], 0)
        hi = np.min([np.array(metas[f]["corners"]).max(0) for f in fams], 0)
        if np.any(hi <= lo):                                        # no common intersection: use the union centre
            lo, hi = allc.min(0), allc.max(0)
        m = (lo + hi) / 2
        half = (hi - lo) / 4
        probes = np.array([m] + [m + half * np.array(s) for s in
                                 [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]])
        cov = {f: footprint_cov(metas[f]) for f in fams}
        rng = np.random.default_rng(1000 + pick.index(u))
        n_qc_ref = 0
        for ia, A in enumerate(fams):
            for B in fams[ia + 1:]:
                cross = PLANE[A] != PLANE[B]
                base = dict(study=u, fixed=A, moving=B, fixed_plane=PLANE[A], moving_plane=PLANE[B],
                            pair_type="cross" if cross else "same", vendor=metas[A]["vendor"], field=metas[A]["field"],
                            centre=m.tolist(), half=half.tolist())
                try:
                    mask, nvox = fixed_mask(imgs[A], m, imgs[B])
                    if nvox < MIN_MASK_VOXELS:
                        pair_rows.append(dict(base, inject=0, ok=False, error=f"mask {nvox} voxels"))
                        continue
                    M0, info = register(imgs[A], imgs[B], mask, m, probes, rng, seed=17)
                    dev = deviation(M0, probes, cov[A], cov[B])
                    pair_rows.append(dict(base, inject=0, ok=True, mask_voxels=nvox, M=M0.ravel().tolist(), **dev, **info))
                    over = dev["disp_max_mm"] > TOL_MM or dev["rot_deg"] > TOL_DEG
                    if a.qc and (over or (cross and n_qc_ref < 1)):
                        n_qc_ref += 0 if over else 1
                        qc_png(imgs[A], imgs[B], euler(M0, m),
                               out / "qc" / f"{'over' if over else 'ref'}_{u[-12:]}_{A}_{B}.png")
                    if not cross:
                        continue
                    for lvl, (mm, deg) in enumerate(INJECT, start=1):
                        P = rigid(rng.normal(size=3), deg, mm * unit(rng.normal(size=3)), m)
                        Bp = moved(imgs[B], P)
                        mask_p, nvox_p = fixed_mask(imgs[A], m, imgs[B], Bp)
                        if nvox_p < MIN_MASK_VOXELS:
                            pair_rows.append(dict(base, inject=lvl, ok=False, error=f"mask {nvox_p} voxels"))
                            continue
                        Mp, infop = register(imgs[A], Bp, mask_p, m, probes, rng, seed=17 + 10 * lvl)
                        err = deviation(np.linalg.inv(P @ M0) @ Mp, probes)
                        pair_rows.append(dict(base, inject=lvl, inject_mm=mm, inject_deg=deg, ok=True,
                                              mask_voxels=nvox_p, M=Mp.ravel().tolist(), P=P.ravel().tolist(),
                                              rec_rot_deg=err["rot_deg"], rec_disp_max_mm=err["disp_max_mm"], **infop))
                except Exception as e:  # noqa: BLE001
                    pair_rows.append(dict(base, inject=0, ok=False, error=f"{type(e).__name__}: {e}"[:200]))
                    traceback.print_exc()
        print(f"[{si + 1}/{len(mine)}] {u[-12:]} families={len(fams)} {time.time() - t0:.0f}s "
              f"(total {time.time() - t_start:.0f}s)", flush=True)
    pd.DataFrame(pair_rows).to_parquet(out / f"pairs_{a.shard:02d}.parquet")
    pd.DataFrame(series_rows).to_parquet(out / f"series_{a.shard:02d}.parquet")
    print(f"shard {a.shard}: {len(mine)} studies, {len(pair_rows)} registrations, {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
