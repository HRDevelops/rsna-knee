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

"""Phase 0A report: aggregates scripts/phase0a_registration.py shards and applies the pre-registered gates of
docs/research/PLAN_v1_3.md s5 (amended after the final review). Gates are on CROSS-PLANE pairs and on triangles with one
series from each plane; same-plane pairs and mixed triangles are reported but not gated.

    python scripts/phase0a_report.py DIR [PILOT_LIST]     (writes DIR/report.md and prints it)

PILOT_LIST (optional): file with one StudyInstanceUID per line. Those studies were used to develop the audit tool
(pilots 1-3, 2026-10-25), so the gates are computed WITHOUT them; their numbers are shown separately.

Gates:
  G1 optimizer precision: injected errors (5 mm/3 deg and 10 mm/5 deg, cross-plane pairs) recovered within 1 mm and
     0.5 deg in >= 95% of cases, for each level separately;
  G2 cycle consistency: median residual of Sag -> Cor -> Ax -> Sag triangles <= 1 mm (max displacement at the probes);
  G3 raw DICOM geometry: >= 97% of cross-plane pairs need a correction <= 3 mm (max over probes) and <= 2 deg.
Decision: G1 and G2 must pass for the audit to count. Then G3 pass -> registration-free (raw geometry; pairs over
tolerance are marked); G3 fail -> train with misregistration noise drawn from the measured cross-plane corrections.
"""
from __future__ import annotations

import glob
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL_MM, TOL_DEG = 3.0, 2.0
PREC_MM, PREC_DEG = 1.0, 0.5
GROSS_MM, GROSS_DEG = 10.0, 5.0


def dev(M: np.ndarray, probes: np.ndarray) -> tuple[float, float]:
    R = M[:3, :3]
    ang = math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    d = probes @ R.T + M[:3, 3] - probes
    return float(np.linalg.norm(d, axis=1).max()), ang


def probes_of(row) -> np.ndarray:
    m, h = np.array(row.centre), np.array(row.half)
    return np.array([m] + [m + h * np.array(s) for s in itertools.product((-1, 1), repeat=3)])


def q(s: pd.Series, qs=(0.5, 0.9, 0.95, 0.99)) -> str:
    return " / ".join(f"{v:.2f}" for v in s.quantile(list(qs))) if len(s) else "n/a"


def main(d: Path, pilot: set | None = None) -> None:
    pairs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(d / "pairs_*.parquet")))], ignore_index=True)
    series = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(d / "series_*.parquet")))], ignore_index=True)
    out = []
    p = out.append
    pilot = pilot or set()
    pilot_pairs = pairs[pairs.study.isin(pilot) & (pairs.inject == 0) & pairs.ok & (pairs.pair_type == "cross")]
    pairs = pairs[~pairs.study.isin(pilot)]
    series = series[~series.study.isin(pilot)]
    n_st = series.study.nunique()
    p(f"# Phase 0A report ({d.name})\n")
    if pilot:
        within = ((pilot_pairs.disp_max_mm <= TOL_MM) & (pilot_pairs.rot_deg <= TOL_DEG)).mean()
        p(f"Gates use the {n_st} studies NOT used to develop the audit tool. The {len(pilot)} pilot studies are excluded "
          f"(their cross-plane pairs, for the record: {len(pilot_pairs)}, within 3 mm and 2°: {within:.1%}).\n")
    p(f"Studies: {n_st}. Series loaded: {int(series.ok.sum())} / {len(series)} "
      f"(failures: {series[~series.ok].error.str.split(':').str[0].value_counts().to_dict() if (~series.ok).any() else 0}).")
    ser_ok = series[series.ok]
    p(f"Series flags: non-uniform slice spacing > 10%: {(ser_ok.nonuniform > 0.1).mean():.1%}; "
      f"in-plane drift of slice origins > 0.5 mm: {(ser_ok.shear_mm > 0.5).mean():.1%}; "
      f"orientation varies across slices (> 1e-3): {(ser_ok.iop_dev > 1e-3).mean():.1%}.\n")

    base = pairs[(pairs.inject == 0)]
    fail = base[~base.ok]
    base = base[base.ok].copy()
    base["within"] = (base.disp_max_mm <= TOL_MM) & (base.rot_deg <= TOL_DEG)
    base["gross"] = (base.disp_max_mm > GROSS_MM) | (base.rot_deg > GROSS_DEG)
    base["kind"] = base.fixed_plane.str[:3] + "-" + base.moving_plane.str[:3]
    p(f"Registrations without injection: {len(base)} ok, {len(fail)} failed "
      f"({fail.error.str.split(' ').str[0].value_counts().to_dict() if len(fail) else 0}).\n")

    # ---------------------------------------------------------------- G1 optimizer precision (cross-plane injections)
    inj = pairs[(pairs.inject > 0) & pairs.ok].copy()
    inj["recovered"] = (inj.rec_disp_max_mm <= PREC_MM) & (inj.rec_rot_deg <= PREC_DEG)
    p("## G1 optimizer precision (cross-plane pairs, injected rigid errors)\n")
    p("| Injection | n | Recovered within 1 mm and 0.5° | Recovery error mm, median / p90 / p95 / p99 | Recovery error °, median / p90 / p95 / p99 |")
    p("|---|---|---|---|---|")
    g1 = True
    for lvl, g in inj.groupby("inject"):
        share = g.recovered.mean()
        g1 &= bool(share >= 0.95)
        p(f"| {g.inject_mm.iloc[0]:.0f} mm / {g.inject_deg.iloc[0]:.0f}° | {len(g)} | {share:.1%} | {q(g.rec_disp_max_mm)} | {q(g.rec_rot_deg)} |")
    if inj.empty:
        g1 = False
    p(f"\n**G1 {'PASS' if g1 else 'FAIL'}** (≥ 95% at each level).\n")

    # ---------------------------------------------------------------- G2 cycle consistency
    rows = []
    for u, g in base.groupby("study"):
        M = {(r.fixed, r.moving): np.array(r.M).reshape(4, 4) for r in g.itertuples()}
        plane = {**dict(zip(g.fixed, g.fixed_plane)), **dict(zip(g.moving, g.moving_plane))}
        pr = probes_of(g.iloc[0])
        fams = sorted(plane, key=lambda f: ["Sagittal_FS", "Sagittal_nonFS", "Coronal_FS", "Coronal_nonFS",
                                            "Axial_any"].index(f))
        for a, b, c in itertools.combinations(fams, 3):
            if (a, b) in M and (b, c) in M and (a, c) in M:
                cyc = np.linalg.inv(M[(a, c)]) @ M[(b, c)] @ M[(a, b)]
                mm, deg = dev(cyc, pr)
                kind = "all-three-plane" if len({plane[a], plane[b], plane[c]}) == 3 else "mixed"
                rows.append(dict(study=u, kind=kind, mm=mm, deg=deg))
    cyc = pd.DataFrame(rows)
    p("## G2 cycle consistency (A → B → C → A, max displacement at the probes)\n")
    p("| Triangle | n | Residual mm, median / p90 / p95 / p99 | Residual °, median / p90 / p95 / p99 |")
    p("|---|---|---|---|")
    g2 = False
    for kind in ("all-three-plane", "mixed"):
        g = cyc[cyc.kind == kind] if len(cyc) else cyc
        if len(g):
            p(f"| {kind} | {len(g)} | {q(g.mm)} | {q(g.deg)} |")
            if kind == "all-three-plane":
                g2 = bool(g.mm.median() <= 1.0)
    p(f"\n**G2 {'PASS' if g2 else 'FAIL'}** (all-three-plane median ≤ 1 mm).\n")

    # ---------------------------------------------------------------- G3 raw DICOM geometry
    p("## G3 raw DICOM geometry: correction needed by each pair\n")
    p("| Pairs | n | Within 3 mm and 2° | Gross (> 10 mm or > 5°) | Correction mm, median / p90 / p95 / p99 | Rotation °, median / p90 / p95 / p99 | Footprint d (6×6), median / p90 / p95 / p99 |")
    p("|---|---|---|---|---|---|---|")
    for name, g in [("**cross-plane (gated)**", base[base.pair_type == "cross"]),
                    ("same-plane (reported)", base[base.pair_type == "same"])] + \
                   [(f"  {k}", g) for k, g in base.groupby("kind")]:
        if len(g):
            p(f"| {name} | {len(g)} | {g.within.mean():.1%} | {g.gross.mean():.1%} | {q(g.disp_max_mm)} | "
              f"{q(g.rot_deg)} | {q(g.d_max)} |")
    cross = base[base.pair_type == "cross"]
    g3 = bool(len(cross) and cross.within.mean() >= 0.97)
    p(f"\n**G3 {'PASS' if g3 else 'FAIL'}** (≥ 97% of cross-plane pairs within 3 mm and 2°).\n")
    p("Cross-plane, by vendor and field strength:\n")
    p("| Group | n | Within 3 mm and 2° | Correction mm, median / p90 / p95 / p99 |")
    p("|---|---|---|---|")
    for col in ("vendor", "field"):
        for k, g in cross.groupby(col):
            p(f"| {col} {k} | {len(g)} | {g.within.mean():.1%} | {q(g.disp_max_mm)} |")
    gain = cross.mi_dicom - cross.mi_best
    p("\nSearch diagnostics (cross-plane; exact Mattes MI over all mask voxels, negative, lower is better; the unmoved "
      f"DICOM placement is one of the candidates): DICOM placement itself best in {(cross.best == 'dicom').mean():.1%}; "
      f"MI gain of the selected candidate over the DICOM placement median {gain.median():.4f}, p90 "
      f"{gain.quantile(.9):.4f}; optimized starts agreeing within 1 mm with each other: median "
      f"{cross.n_agree_1mm.median():.0f} of {cross.n_starts_ok.median():.0f} (all agree in "
      f"{(cross.n_agree_1mm == cross.n_starts_ok).mean():.1%}); candidates rejected by the overlap guard: "
      f"{cross.n_rejected_overlap.sum():.0f}.\n")

    # ---------------------------------------------------------------- decision
    p("## Decision\n")
    if not (g1 and g2):
        p("**AUDIT INCONCLUSIVE.** The registration is not precise or self-consistent enough to judge the raw "
          "geometry (G1 and G2 must pass). Fix the audit (settings, multi-start) before deciding; no geometry claim yet.")
    elif g3:
        p("**REGISTRATION-FREE.** Raw DICOM geometry is used as is. Cross-plane pairs over tolerance are marked "
          "(their tokens get no geometric bias), and the measured residuals define the dose–response scale.")
    else:
        p("**RAW GEOMETRY NOT SUFFICIENT AS IS.** Train with misregistration noise drawn from the measured cross-plane "
          "corrections (registration cannot run on the hidden test); look at the QC overlays of the gross cases first.")
    over = cross[~cross.within].sort_values("disp_max_mm", ascending=False)
    if len(over):
        p("\nCross-plane pairs over tolerance (worst 15; QC overlays in qc/ if run with --qc):\n")
        p("| Study (tail) | Pair | Correction mm | Rotation ° | d | MI gain over DICOM | Starts agreeing |")
        p("|---|---|---|---|---|---|---|")
        for r in over.head(15).itertuples():
            p(f"| …{r.study[-10:]} | {r.fixed} → {r.moving} | {r.disp_max_mm:.1f} | {r.rot_deg:.1f} | {r.d_max:.2f} | "
              f"{r.mi_dicom - r.mi_best:.4f} | {r.n_agree_1mm}/{r.n_starts_ok} |")
    text = "\n".join(out) + "\n"
    (d / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    pil = set()
    if len(sys.argv) > 2:
        pil = {ln.strip() for ln in Path(sys.argv[2]).read_text().splitlines() if ln.strip()}
    main(Path(sys.argv[1]), pil)
