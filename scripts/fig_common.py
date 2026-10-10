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

"""Shared helpers for the presentation / report figures (docs/figures/). Analysis only: nothing here trains a
model or touches a scored artifact. Inputs are small per-study prediction files copied from the HPC into
docs/figures/_data/ (gitignored: they are derived from competition labels).

Conventions (same as docs/EXPERIMENTS.md):
  OOF    = 5-fold out-of-fold predictions on the 4,349 non-gold studies, scored against `labels_v2_final`
           binarised at 0.5 (a noisy report-derived proxy).
  gold58 = the 58 image-reviewed studies, never trained on; each model's prediction is the mean over fold models.
  exp018 = 5 checkpoints (seed 42). exp021 = 10 checkpoints (seeds 42 + 43), i.e. the mean of exp018 and the
           seed-43 run (the submitted main-track model).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs/figures/_data"
FIG = ROOT / "docs/figures"

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]

# Okabe-Ito (colour-blind safe); every series that matters also differs in line style / marker.
C = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "orange": "#E69F00", "sky": "#56B4E9",
     "purple": "#CC79A7", "grey": "#6B7280", "black": "#111111", "light": "#E5E7EB"}


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,           # embed TrueType: editable text in the PDFs
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


def save(fig, name: str) -> None:
    """PNG (300 dpi) + PDF into docs/figures/."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300)
    fig.savefig(FIG / f"{name}.pdf")
    print(f"wrote docs/figures/{name}.png + .pdf")


def _concat(exp: str, fname: str) -> pd.DataFrame:
    parts = [pd.read_parquet(DATA / exp / f"fold{k}" / fname) for k in range(5)]
    return pd.concat(parts, ignore_index=True)


def load_predictions() -> dict:
    """{'oof': {name: DataFrame[study x 12]}, 'gold': {...}, 'y_oof': ndarray[12, N], 'y_gold': ndarray[12, 58]} with
    studies aligned across models. Asserts the headline numbers of the ledger are reproduced."""
    oof18 = _concat("exp018", "oof.parquet").set_index("study_uid")[TARGETS]
    oof21s = _concat("exp021", "oof.parquet").set_index("study_uid")[TARGETS]
    g18 = _concat("exp018", "gold58_predictions.parquet").groupby("study_uid")[TARGETS].mean()
    g21s = _concat("exp021", "gold58_predictions.parquet").groupby("study_uid")[TARGETS].mean()

    lab = pd.read_parquet(DATA / "labels_v2_final.parquet").set_index("study_uid")
    uids = sorted(set(oof18.index) & set(oof21s.index) & set(lab.index))
    y_oof = np.stack([(lab.loc[uids, f"{t}__probability"].to_numpy() >= 0.5).astype(int) for t in TARGETS])
    gold = pd.read_csv(DATA / "_gold58_labels.csv").rename(columns={"StudyInstanceUID": "study_uid"}).set_index("study_uid")
    gu = sorted(set(g18.index) & set(g21s.index) & set(gold.index))
    y_gold = gold.loc[gu, TARGETS].to_numpy().T.astype(int)

    o18, o21s = oof18.loc[uids], oof21s.loc[uids]
    out = {
        "uids_oof": uids, "uids_gold": gu, "y_oof": y_oof, "y_gold": y_gold,
        "oof": {"exp018": o18, "exp021": (o18 + o21s) / 2},
        "gold": {"exp018": g18.loc[gu], "exp021": (g18.loc[gu] + g21s.loc[gu]) / 2},
    }
    exp = {("oof", "exp018"): 0.8816, ("oof", "exp021"): 0.8840, ("gold", "exp018"): 0.9091, ("gold", "exp021"): 0.9111}
    for (split, name), want in exp.items():
        got = macro_auc(out[split][name].to_numpy().T, out["y_" + ("oof" if split == "oof" else "gold")])
        assert abs(got - want) < 6e-4, f"{split} {name}: {got:.4f} != ledger {want}"
    return out


def macro_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """scores, y: [12, N]."""
    return float(np.mean([roc_auc_score(y[i], scores[i]) for i in range(y.shape[0])]))
