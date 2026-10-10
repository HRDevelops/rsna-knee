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

"""Per-label ROC curves for exp018 and exp021: one 12-panel figure for OOF (n=4,349, proxy labels) and one for
gold-58 (n=58, image-review labels, with a study-bootstrap 95% CI on every AUC).

    python scripts/fig_roc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_common as fc  # noqa: E402

STYLE = {"exp018": dict(color=fc.C["blue"], ls="--", lw=1.1, label="exp018 (5 ckpt)"),
         "exp021": dict(color=fc.C["vermillion"], ls="-", lw=1.3, label="exp021 (10 ckpt)")}


def boot_ci(s: np.ndarray, y: np.ndarray, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals, n = [], len(y)
    while len(vals) < reps:
        ix = rng.integers(0, n, n)
        if len(set(y[ix])) == 2:
            vals.append(roc_auc_score(y[ix], s[ix]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def panel_figure(split: str, d: dict, title: str, name: str, ci: bool) -> None:
    y = d["y_oof" if split == "oof" else "y_gold"]
    fig, axes = plt.subplots(3, 4, figsize=(7.16, 5.7), sharex=True, sharey=True)
    for i, (ax, t) in enumerate(zip(axes.ravel(), fc.TARGETS)):
        npos, n = int(y[i].sum()), y.shape[1]
        ax.plot([0, 1], [0, 1], color=fc.C["grey"], lw=0.6, ls=":")
        txt = []
        for m in ("exp018", "exp021"):
            s = d[split][m][t].to_numpy()
            fpr, tpr, _ = roc_curve(y[i], s)
            ax.plot(fpr, tpr, **{k: v for k, v in STYLE[m].items() if k != "label"})
            auc = roc_auc_score(y[i], s)
            if ci and m == "exp021":
                lo, hi = boot_ci(s, y[i])
                txt.append(f"{auc:.3f} [{lo:.2f}, {hi:.2f}]")
            elif m == "exp021":
                txt.append(f"{auc:.3f}")
            else:
                txt.append(f"{auc:.3f}")
        ax.set_title(t, loc="left", fontsize=7.8, fontweight="bold")
        ax.text(0.97, 0.05, f"n+ = {npos}/{n}\nexp018 {txt[0]}\nexp021 {txt[1]}", ha="right", va="bottom",
                fontsize=6.0, transform=ax.transAxes, linespacing=1.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal", adjustable="box")
        ax.grid(color=fc.C["light"], lw=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("False-positive rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("True-positive rate")
    h = [plt.Line2D([], [], **{k: v for k, v in STYLE[m].items() if k != "label"}, label=STYLE[m]["label"]) for m in STYLE]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    ma = {m: fc.macro_auc(d[split][m].to_numpy().T, y) for m in STYLE}
    fig.suptitle(f"{title}\nmacro AUC: exp018 {ma['exp018']:.4f}, exp021 {ma['exp021']:.4f}", x=0.01, ha="left",
                 fontsize=8.3, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fc.save(fig, name)
    plt.close(fig)


def main() -> None:
    fc.set_style()
    d = fc.load_predictions()
    print(f"OOF n={len(d['uids_oof'])}, gold n={len(d['uids_gold'])}")
    panel_figure("oof", d, "Out-of-fold ROC vs report-derived proxy labels (n = 4,349)", "roc_oof_exp018_exp021", ci=False)
    panel_figure("gold", d, "Gold-58 ROC vs image-review labels (n = 58; exp021 AUC with [95% study-bootstrap CI])", "roc_gold58_exp018_exp021", ci=True)


if __name__ == "__main__":
    main()
