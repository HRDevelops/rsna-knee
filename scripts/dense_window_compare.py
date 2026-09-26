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

"""Compare dense-window vs standard-layout inference for the existing exp018 checkpoints
(outputs of scripts/dense_window_eval.py). Report only; nothing is deployed.

- OOF macro-AUC vs labels_v2_final on the non-gold studies both layouts decoded, with the same paired cluster
  bootstrap over report clusters as every earlier verdict; per-label deltas.
- Gold-58 (5-fold mean), with a study bootstrap CI of the paired delta.
- Control: the standard-layout run vs exp018's recorded training-path OOF (artifacts/<exp>/fold*/oof.parquet).
  They differ only in the side rule (inference uses tag -> geometry; the training path used the validated side
  table), so they should agree closely; this shows the evaluation path itself is sound.

    python scripts/dense_window_compare.py --out artifacts/dense_window_eval.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import TARGETS, load_full_oof, load_gold_truth, load_truth  # noqa: E402


def macro(s, y, idx=None):
    idx = np.arange(y.shape[1]) if idx is None else idx
    return float(np.mean([roc_auc_score(y[i, idx], s[i, idx]) for i in range(12) if len(set(y[i, idx])) == 2]))


def per_label(s, y):
    return [roc_auc_score(y[i], s[i]) if len(set(y[i])) == 2 else float("nan") for i in range(12)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    art = root / "artifacts"
    std = pd.read_parquet(art / "dense_window_eval_standard.parquet").set_index("study_uid")
    den = pd.read_parquet(art / "dense_window_eval_dense.parquet").set_index("study_uid")
    truth = load_truth(art / "labels/labels_v2_final.parquet")
    gtruth = load_gold_truth(root / "data/raw/train.csv")
    clusters = pd.read_parquet(art / "report_clusters.parquet").set_index("study_uid")["report_cluster_id"]
    train_oof = load_full_oof(art / "exp018_coatnet2_wmil_v3labels")

    both = std.index.intersection(den.index)
    oof_u = sorted(u for u in both if std.loc[u, "fold"] >= 0 and u in truth and u in clusters.index)
    gold_u = sorted(u for u in both if std.loc[u, "fold"] == -1 and u in gtruth)
    y = np.stack([[truth[u][i] for u in oof_u] for i in range(12)]).astype(int)
    S, D = std.loc[oof_u, TARGETS].to_numpy().T, den.loc[oof_u, TARGETS].to_numpy().T

    cid = clusters.loc[oof_u].to_numpy()
    groups = {c: np.where(cid == c)[0] for c in np.unique(cid)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(0)
    bd = []
    for _ in range(a.reps):
        ix = np.concatenate([groups[c] for c in rng.choice(keys, len(keys), replace=True)])
        bd.append(macro(D, y, ix) - macro(S, y, ix))
    bd = np.array(bd)
    lo, hi = np.percentile(bd, [2.5, 97.5])

    gy = np.stack([[gtruth[u][i] for u in gold_u] for i in range(12)]).astype(int)
    GS, GD = std.loc[gold_u, TARGETS].to_numpy().T, den.loc[gold_u, TARGETS].to_numpy().T
    grng = np.random.default_rng(1)
    gb = [macro(GD, gy, ix) - macro(GS, gy, ix) for ix in (grng.integers(0, len(gold_u), len(gold_u)) for _ in range(a.reps))]
    glo, ghi = np.percentile(gb, [2.5, 97.5])

    ctrl_u = [u for u in oof_u if u in train_oof.index]
    T = train_oof.loc[ctrl_u, TARGETS].to_numpy().T
    Sc = std.loc[ctrl_u, TARGETS].to_numpy().T
    yc = np.stack([[truth[u][i] for u in ctrl_u] for i in range(12)]).astype(int)
    rho = [spearmanr(Sc[i], T[i]).correlation for i in range(12)]

    L = ["# Dense-window inference on the existing exp018 checkpoints (report only, not deployed)", "",
         f"Standard layout: 64 slices, mean {std['n_windows'].mean():.1f} windows/study. Dense (x1.6): 102 slices, "
         f"mean {den['n_windows'].mean():.1f} windows/study. Same inference path, fp16, own-fold model per study.", "",
         "## Control: is the evaluation path sound?", "",
         f"Standard-layout run vs exp018's recorded training-path OOF on {len(ctrl_u)} studies: OOF macro "
         f"{macro(Sc, yc):.4f} vs {macro(T, yc):.4f}; per-label Spearman {min(rho):.4f}-{max(rho):.4f} (the paths differ "
         f"only in the side rule).", "",
         "## Result", "",
         "| | standard | dense | Δ (dense − standard) | 95% CI | P(Δ ≤ 0) |", "|---|---|---|---|---|---|",
         f"| OOF macro (n={len(oof_u)}, paired cluster bootstrap, {a.reps} reps) | {macro(S, y):.4f} | {macro(D, y):.4f} | "
         f"{macro(D, y) - macro(S, y):+.4f} | [{lo:+.4f}, {hi:+.4f}] | {(bd <= 0).mean():.3f} |",
         f"| gold-58 macro, 5-fold mean (n={len(gold_u)}, study bootstrap) | {macro(GS, gy):.4f} | {macro(GD, gy):.4f} | "
         f"{macro(GD, gy) - macro(GS, gy):+.4f} | [{glo:+.4f}, {ghi:+.4f}] | {(np.array(gb) <= 0).mean():.3f} |", "",
         "## Per label (OOF)", "", "| label | standard | dense | Δ |", "|---|---|---|---|"]
    ps, pd_ = per_label(S, y), per_label(D, y)
    for i, t in enumerate(TARGETS):
        L.append(f"| {t} | {ps[i]:.4f} | {pd_[i]:.4f} | {pd_[i] - ps[i]:+.4f} |")
    rep = "\n".join(L) + "\n"
    print(rep)
    if a.out:
        a.out.write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
