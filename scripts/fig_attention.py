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

"""Window-MIL attention figures from `scripts/viz_attention.py` output (docs/figures/_data/attention_gold58.npz).

  attention_windows_gold58   six pre-selected gold-58 studies (scripts/fig_select_attention_studies.py: the most
                             confidently detected gold-positive study per finding): the top-4 attended windows for
                             that finding, plus the attention over all of the study's windows. Best-case illustration.
  attention_by_slot_gold58   NOT selected: attention mass per slot for each finding, averaged over all 58 gold
                             studies, next to the share of windows each slot has (what uniform attention would give).

Attention = mean over the 10 exp021 checkpoints (windows are identical across checkpoints). Attention shows where the
model's pooled representation draws from; it is not proof that those windows contain the finding.

    python scripts/fig_attention.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_common as fc  # noqa: E402

SLOT_NAMES = ["Sagittal FS", "Sagittal non-FS", "Coronal FS", "Coronal non-FS", "Axial"]
SLOT_SHORT = ["Sag FS", "Sag", "Cor FS", "Cor", "Ax"]
SLOT_LEN = [18, 14, 12, 8, 12]
SLOT_OFF = [0, 18, 32, 44, 52]
SLOT_COL = [fc.C["blue"], fc.C["sky"], fc.C["green"], fc.C["orange"], fc.C["purple"]]
N_SEED = 5   # checkpoint_order: exp018 folds 0-4 first, then exp021 (seed 43) folds 0-4


def load() -> dict:
    z = np.load(fc.DATA / "attention_gold58.npz", allow_pickle=False)
    uids = sorted({k.split("|")[0] for k in z.files})
    return {u: {k.split("|")[1]: z[k] for k in z.files if k.startswith(u + "|")} for u in uids}


def sanity_check(d: dict, pred: dict) -> None:
    """The 10-checkpoint probabilities from this extraction must reproduce exp021's gold-58 numbers."""
    uids = pred["uids_gold"]
    p = np.stack([d[u]["prob"].mean(0) for u in uids])                      # [58, 12]
    ref = pred["gold"]["exp021"].to_numpy()
    y = pred["y_gold"]
    auc = np.mean([roc_auc_score(y[i], p[:, i]) for i in range(12)])
    print(f"sanity: macro AUC from extracted probs {auc:.4f} (ledger exp021 gold-58 0.9111); "
          f"max |dprob| vs stored gold predictions {np.abs(p - ref).max():.4f}, mean {np.abs(p - ref).mean():.5f}")
    assert abs(auc - 0.9111) < 0.003, auc


def windows_figure(d: dict, sel: pd.DataFrame, name: str = "attention_windows_gold58") -> None:
    n = len(sel)
    fig = plt.figure(figsize=(7.16, 1.45 * n + 0.75))
    gs = fig.add_gridspec(n, 7, width_ratios=[1.45, 1, 1, 1, 1, 0.3, 1.4], wspace=0.12, hspace=0.5,
                          left=0.01, right=0.995, top=0.915, bottom=0.05)
    for r, row in sel.reset_index(drop=True).iterrows():
        e = d[row.study_uid]
        o = fc.TARGETS.index(row.finding)
        a = e["attn"].mean(0)[:, o]                                         # [K], sums to 1
        cs, so, stack = e["centers"], e["slot_of"], e["stack"]
        top = np.argsort(-a)[:4]
        ax = fig.add_subplot(gs[r, 0]); ax.axis("off")
        ax.text(0, 1.0, row.finding, fontsize=8.3, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0, 0.80, f"gold-positive, p = {e['prob'].mean(0)[o]:.3f}\ntop-4 of {len(a)} windows\n"
                f"= {a[top].sum():.0%} of attention", fontsize=6.0, va="top", transform=ax.transAxes, linespacing=1.4)
        also = [t for t in row.all_gold_positive.split("; ") if t != row.finding]
        ax.text(0, 0.30, "also gold+:\n" + textwrap.fill(", ".join(also), 26), fontsize=5.0, color=fc.C["grey"],
                va="top", transform=ax.transAxes, linespacing=1.3)
        for j, k in enumerate(top):
            axi = fig.add_subplot(gs[r, 1 + j])
            axi.imshow(stack[cs[k]], cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
            axi.set_xticks([]); axi.set_yticks([])
            for sp in axi.spines.values():
                sp.set_edgecolor(SLOT_COL[so[k]]); sp.set_linewidth(1.6)
            pos = cs[k] - SLOT_OFF[so[k]] + 1
            axi.set_title(f"#{j + 1} {SLOT_SHORT[so[k]]} {pos}/{SLOT_LEN[so[k]]}\nattn {a[k]:.2f}",
                          fontsize=5.6, pad=2, linespacing=1.2)
        axp = fig.add_subplot(gs[r, 6])
        cols = [SLOT_COL[s] for s in so]
        axp.bar(np.arange(len(a)), a, color=cols, width=0.9, lw=0)
        axp.bar(top, a[top], color=[cols[k] for k in top], width=0.9, edgecolor=fc.C["black"], lw=0.7)
        axp.set_xlim(-1, len(a)); axp.set_xticks([]); axp.tick_params(axis="y", labelsize=5.0, length=2, pad=1)
        axp.set_ylim(0, max(a.max() * 1.15, 0.05))
        axp.yaxis.set_major_locator(plt.MaxNLocator(3))
        if r == 0:
            axp.set_title("attention over all windows", fontsize=6.0, pad=3)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SLOT_COL]
    fig.legend(h, SLOT_NAMES, loc="lower center", ncol=5, frameon=False, fontsize=6.2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Window-MIL: the four most-attended windows per finding (exp021, mean of 10 checkpoints)",
                 x=0.01, ha="left", fontsize=8.3, y=0.998)
    fig.text(0.01, 0.945, "Studies chosen by a fixed rule before any map was viewed: the most confidently detected "
             "gold-positive study per finding. Best-case illustration, not a hit rate.", fontsize=5.8, color=fc.C["grey"])
    fc.save(fig, name)
    plt.close(fig)


def slot_figure(d: dict) -> pd.DataFrame:
    uids = sorted(d)
    mass = np.zeros((len(uids), 12, 5)); share = np.zeros((len(uids), 5)); neff = np.zeros((len(uids), 12))
    agree = np.zeros((len(uids), 12))
    for i, u in enumerate(uids):
        e = d[u]
        a = e["attn"].mean(0)                                               # [K, 12]
        for s in range(5):
            m = e["slot_of"] == s
            mass[i, :, s] = a[m].sum(0)
            share[i, s] = m.mean()
        neff[i] = 1.0 / (a ** 2).sum(0)
        a42, a43 = e["attn"][:N_SEED].mean(0), e["attn"][N_SEED:].mean(0)
        agree[i] = [spearmanr(a42[:, o], a43[:, o]).statistic for o in range(12)]
    M, S = mass.mean(0), share.mean(0)
    K = np.mean([len(d[u]["centers"]) for u in uids])
    df = pd.DataFrame(M, index=fc.TARGETS, columns=SLOT_NAMES)
    df["effective_windows"] = neff.mean(0)
    df["seed42_vs_seed43_spearman"] = np.nanmean(agree, axis=0)
    df.loc["(uniform attention = window share)", SLOT_NAMES] = S
    df.round(4).to_csv(fc.FIG / "attention_slot_mass_gold58.csv")

    fig = plt.figure(figsize=(7.16, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42], wspace=0.04)
    ax = fig.add_subplot(gs[0])
    allM = np.vstack([M, S[None]])
    im = ax.imshow(allM, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
    for i in range(allM.shape[0]):
        for j in range(5):
            ax.text(j, i, f"{allM[i, j]:.2f}", ha="center", va="center", fontsize=6.4,
                    color="white" if allM[i, j] > 0.45 else fc.C["black"])
    ax.axhline(11.5, color=fc.C["black"], lw=0.8)
    ax.set_xticks(range(5), SLOT_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(13), fc.TARGETS + ["uniform (window share)"])
    ax.set_title("Share of attention per slot, mean over all 58 gold studies (not selected)", loc="left", fontsize=7.8)
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    ax2 = fig.add_subplot(gs[1], sharey=ax)
    ax2.barh(range(12), df["effective_windows"].iloc[:12], color=fc.C["vermillion"], height=0.62)
    ax2.axvline(K, color=fc.C["grey"], ls=":", lw=0.8)
    ax2.text(K, 12.2, f"all {K:.0f}", fontsize=5.8, color=fc.C["grey"], ha="right", va="center")
    for i, v in enumerate(df["effective_windows"].iloc[:12]):
        ax2.text(v + 0.8, i, f"{v:.0f}", fontsize=6.2, va="center")
    ax2.set_xlim(0, K * 1.12); ax2.tick_params(labelleft=False, length=0)
    ax2.set_xlabel("effective windows (1/sum a^2)", fontsize=6.5)
    ax2.set_title("How peaked", loc="left", fontsize=7.8)
    fc.save(fig, "attention_by_slot_gold58")
    plt.close(fig)
    print(df.round(3).to_string())
    return df


def main() -> None:
    fc.set_style()
    d = load()
    pred = fc.load_predictions()
    sanity_check(d, pred)
    sel = pd.read_csv(fc.DATA / "attention_selection.csv")
    windows_figure(d, sel)
    # slide-sized variant: three of the six rows, chosen for covering three different planes (a presentation choice)
    windows_figure(d, sel[sel.finding.isin(["ACL", "Medial Meniscus", "Baker's"])], "attention_windows_gold58_slide")
    slot_figure(d)


if __name__ == "__main__":
    main()
