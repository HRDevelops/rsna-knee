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

"""Workstream D (zero-GPU): physical spacing between the slices a Window-MIL window actually stacks.

For every study and every slot exp018 filled (the exact series recorded in the wmil cache manifest), read
ImagePositionPatient/ImageOrientationPatient of every slice (header only, `specific_tags`), project onto the
slice normal as exp018's ordering does, and record:
  native_gap_mm   median gap between consecutive slices of the series
  eff_step_mm     gap between ADJACENT CACHED slices = native_gap x (hi - lo) / (n_take - 1), where lo/hi are
                  the 6%/94% span indices exp018 samples (wmil.decode_series_slot's linspace). This is what one
                  3-slice window spans per step; it exceeds native_gap when the series is longer than n_take
                  and falls below it when short series are padded with duplicated slices.
  dup_frac        fraction of adjacent cached slices that are the SAME native slice (duplicates)
Then: exp018 OOF macro-AUC by quartile of eff_step_mm per plane, vs labels_v2_final, each quartile compared with
the other three by a study bootstrap.

    python scripts/audit_slice_spacing.py --workers 32
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsna_knee import wmil  # noqa: E402
from rsna_knee.preprocess import slice_normal  # noqa: E402

PLANE_SLOTS = {"Sagittal": (0, 1), "Coronal": (2, 3), "Axial": (4,)}  # FS slot first, non-FS fallback


def slot_spacing(series_dir: Path, n_take: int) -> dict:
    files = sorted(series_dir.glob("*.dcm"))
    hs = [pydicom.dcmread(str(f), force=True, stop_before_pixels=True,
                          specific_tags=["ImagePositionPatient", "ImageOrientationPatient"]) for f in files]
    iop = hs[0].get("ImageOrientationPatient")
    if iop is None or len(hs) < 2:
        return {"native_gap_mm": np.nan, "eff_step_mm": np.nan, "dup_frac": np.nan, "n_native": len(hs)}
    nrm = slice_normal([float(x) for x in iop])
    proj = np.sort([float(np.dot([float(x) for x in h.ImagePositionPatient], nrm)) for h in hs
                    if h.get("ImagePositionPatient") is not None])
    n = len(proj)
    gaps = np.diff(proj)
    native = float(np.median(np.abs(gaps))) if len(gaps) else np.nan
    lo, hi = int(round(wmil.SPAN[0] * (n - 1))), int(round(wmil.SPAN[1] * (n - 1)))
    take = np.clip(np.round(np.linspace(lo, hi, n_take)).astype(int), 0, n - 1)
    steps = np.abs(np.diff(proj[take]))  # physical step between adjacent cached slices
    return {"native_gap_mm": native, "eff_step_mm": float(np.median(steps)) if len(steps) else np.nan,
            "dup_frac": float(np.mean(np.diff(take) == 0)) if len(take) > 1 else np.nan, "n_native": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = Path(os.environ["RSNA_ROOT"])
    art, raw = root / "artifacts", root / "data/raw"
    man = pd.concat([pd.read_parquet(f) for f in sorted((art / "cache_manifests_wmil").glob("shard_*.parquet"))],
                    ignore_index=True)
    man = man[man["ok"]]

    cache = art / "slice_spacing_audit.parquet"
    if cache.exists():
        sp = pd.read_parquet(cache)
    else:
        jobs = [(r.study_uid, i, str(getattr(r, f"series_{i}"))) for r in man.itertuples()
                for i in range(wmil.NUM_SLOTS) if bool(getattr(r, f"present_{i}"))]

        def run(j):
            u, i, s = j
            try:
                return {"study_uid": u, "slot": i, **slot_spacing(raw / "train_series" / u / s, wmil.SLOT_LEN[i])}
            except Exception as e:  # noqa: BLE001
                return {"study_uid": u, "slot": i, "error": f"{type(e).__name__}: {e}"}
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            sp = pd.DataFrame(list(pool.map(run, jobs)))
        sp.to_parquet(cache, index=False)
    print(f"{sp.study_uid.nunique()} studies, {len(sp)} slot-series; errors: {int(sp.get('error', pd.Series()).notna().sum())}")

    L = ["# Slice-spacing audit (workstream D, zero-GPU)", "",
         "Effective step = physical gap between ADJACENT CACHED slices (what one 3-slice window spans per step).", "",
         "## Distribution per slot", "", "| slot | n | native gap mm (p10/p50/p90) | effective step mm (p10/p25/p50/p75/p90) | studies with duplicated slices |",
         "|---|---|---|---|---|"]
    for i in range(wmil.NUM_SLOTS):
        s = sp[sp.slot == i].dropna(subset=["eff_step_mm"])
        q = np.percentile(s.eff_step_mm, [10, 25, 50, 75, 90]); g = np.percentile(s.native_gap_mm, [10, 50, 90])
        L.append(f"| {i} {wmil.SLOT_SPECS[i][0]} | {len(s)} | {g[0]:.2f} / {g[1]:.2f} / {g[2]:.2f} | "
                 + " / ".join(f"{v:.2f}" for v in q) + f" | {float((s.dup_frac > 0).mean()):.1%} |")

    # ---- OOF by quartile, per plane (FS slot, else the plane's non-FS slot)
    from build_ensemble import TARGETS, load_full_oof, load_truth  # noqa: E402
    from sklearn.metrics import roc_auc_score
    oof = load_full_oof(art / "exp018_coatnet2_wmil_v3labels")
    truth = load_truth(art / "labels/labels_v2_final.parquet")

    def macro(uids):
        y = np.stack([truth[u] for u in uids]); p = oof.loc[uids, TARGETS].to_numpy()
        return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12) if len(set(y[:, j])) == 2]))

    rng = np.random.default_rng(0)
    L += ["", "## exp018 OOF macro-AUC by effective-step quartile (vs labels_v2_final)", "",
          "Δ = quartile minus the other three quartiles; 95% CI from a study bootstrap within each group.", "",
          "| plane | quartile | step range mm | n | OOF macro | Δ vs rest | 95% CI |", "|---|---|---|---|---|---|---|"]
    for plane, slots in PLANE_SLOTS.items():
        per = {}
        for i in slots:  # FS slot wins; non-FS fills studies without one
            for r in sp[(sp.slot == i)].dropna(subset=["eff_step_mm"]).itertuples():
                per.setdefault(r.study_uid, r.eff_step_mm)
        d = pd.Series(per)
        d = d[[u for u in d.index if u in oof.index and u in truth]]
        qs = pd.qcut(d, 4, labels=False, duplicates="drop")
        for k in sorted(qs.unique()):
            inq = list(d.index[qs == k]); rest = list(d.index[qs != k])
            delta = macro(inq) - macro(rest)
            boots = []
            for _ in range(a.reps // 5):  # 200 reps: each macro is 12 AUCs over ~1,000 studies
                bi = list(rng.choice(inq, len(inq))); br = list(rng.choice(rest, len(rest)))
                boots.append(macro(bi) - macro(br))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            L.append(f"| {plane} | Q{k + 1} | {d[inq].min():.2f}–{d[inq].max():.2f} | {len(inq)} | {macro(inq):.4f} | "
                     f"{delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    rep = "\n".join(L) + "\n"
    print(rep)
    if a.out:
        a.out.write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
