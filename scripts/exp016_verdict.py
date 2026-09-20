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

"""Pre-registered exp016 verdict (2026-09-20): solo OOF macro-AUC of each exp016 arm vs exp006, paired
cluster bootstrap over report clusters (same method as scripts/bootstrap_ensemble_gain.py), and the
mechanical bar: an arm "clears the bar" iff its POINT macro gain over exp006 exceeds +0.0082 (2 x the
0.0041 bootstrap SD of the metric). Also prints per-label deltas with their own paired CIs, the
gold-58 tripwire (n=58: reports, never gates), and the arm count so the multiple-comparison cost of
running two weights is visible instead of implicit.

    python scripts/exp016_verdict.py --arms exp016a_dinov2b_slot_v3labels_auxrep_w010 \\
        exp016b_dinov2b_slot_v3labels_auxrep_w050 --out artifacts/exp016_verdict.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import TARGETS, load_full_oof, load_gold_fold_ensemble, load_gold_truth, load_truth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default="exp006_dinov2b_slot_v3labels")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--bar", type=float, default=0.0082)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    art = root / "artifacts"
    truth = load_truth(art / "labels/labels_v2_final.parquet")
    gold_truth = load_gold_truth(root / "data/raw/train.csv")
    clusters = pd.read_parquet(art / "report_clusters.parquet").set_index("study_uid")["report_cluster_id"]

    names = [args.baseline] + args.arms
    oof, gold = {}, {}
    for n in names:
        o = load_full_oof(art / n)
        if o is None:
            print(f"[skip] {n}: fewer than 5 folds with oof.parquet (incomplete)", file=sys.stderr)
            continue
        oof[n], gold[n] = o, load_gold_fold_ensemble(art / n)
    if args.baseline not in oof:
        sys.exit("baseline incomplete")
    arms = [a for a in args.arms if a in oof]

    uids = [u for u in oof[args.baseline].index if u in truth and u in clusters.index]
    y = np.stack([[truth[u][i] for u in uids] for i in range(len(TARGETS))]).astype(int)   # [12, N]
    sc = {n: oof[n].loc[uids, TARGETS].to_numpy().T for n in oof}                            # [12, N]

    def per_label(s: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return np.array([roc_auc_score(y[i, idx], s[i, idx]) for i in range(len(TARGETS))])

    full = np.arange(len(uids))
    point = {n: per_label(sc[n], full) for n in oof}
    cid = clusters.loc[uids].to_numpy()
    groups = {c: np.where(cid == c)[0] for c in np.unique(cid)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(0)
    boot = {n: [] for n in oof}
    for _ in range(args.reps):
        idx = np.concatenate([groups[c] for c in rng.choice(keys, size=len(keys), replace=True)])
        for n in oof:
            boot[n].append(per_label(sc[n], idx))
    boot = {n: np.array(v) for n, v in boot.items()}   # [reps, 12]

    base = args.baseline
    lines = [f"# exp016 verdict vs `{base}` (bar: point macro gain > +{args.bar:.4f}; arms tested: {len(arms)})", "",
             f"{len(uids)} non-gold studies, {len(keys)} report clusters, {args.reps} paired cluster-bootstrap reps. "
             f"Baseline macro {point[base].mean():.4f} (bootstrap SD {boot[base].mean(1).std():.4f}).", "",
             "| arm | macro | Δ vs exp006 | 95% CI (paired) | P(Δ>0) | gold-58 macro (Δ, tripwire) | verdict |",
             "|---|---|---|---|---|---|---|"]
    gold_macro = lambda n: float(np.mean([  # noqa: E731
        roc_auc_score([gold_truth[u][i] for u in gold[n].index if u in gold_truth],
                      gold[n].loc[[u for u in gold[n].index if u in gold_truth], TARGETS[i]]) for i in range(len(TARGETS))]))
    gb = gold_macro(base)
    any_clear = False
    for a in arms:
        d = boot[a].mean(1) - boot[base].mean(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        delta = point[a].mean() - point[base].mean()
        clears = delta > args.bar
        any_clear |= clears
        ga = gold_macro(a)
        lines.append(f"| {a} | {point[a].mean():.4f} | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {(d > 0).mean():.2f} | "
                     f"{ga:.4f} ({ga - gb:+.4f}) | **{'CLEARS BAR' if clears else 'does not clear'}** |")
    lines += ["", "## Per-label Δ vs exp006 (point; 95% paired CI)", "",
              "| arm | " + " | ".join(TARGETS) + " |", "|---|" + "---|" * len(TARGETS)]
    for a in arms:
        cells = []
        for i in range(len(TARGETS)):
            d = boot[a][:, i] - boot[base][:, i]
            lo, hi = np.percentile(d, [2.5, 97.5])
            cells.append(f"{point[a][i] - point[base][i]:+.4f} [{lo:+.3f},{hi:+.3f}]")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines += ["", f"**Overall: {'AT LEAST ONE ARM CLEARS the pre-registered bar' if any_clear else 'NO ARM clears the pre-registered bar'}.** "
              f"Caveats fixed in advance: (1) two arms were run, so 'best of two' is mildly optimistic; (2) OOF is "
              f"scored against report-derived labels, and a model trained to predict report text can gain on that "
              f"proxy by absorbing report idiosyncrasies (over-calling) that the image-reviewed truth does not share "
              f"— the gold-58 column is the only clean check and n=58 cannot resolve ~0.008; (3) one seed per arm."]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        args.out.write_text(report)


if __name__ == "__main__":
    main()
