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

"""Ablation figure: (A) the public-leaderboard path 0.809 -> 0.893 -> 0.929, (B) every tested change as a paired OOF
delta with its 95% cluster-bootstrap CI, grouped by the baseline it was measured against.

Every number below is copied from docs/EXPERIMENTS.md (section named in the comment); nothing is recomputed here.
The 0.893 is the DINOv2 slot pipeline, MOST LIKELY exp005 alone after the notebook's T4 runtime fallback (inferred,
not confirmed; EXPERIMENTS.md 2026-09-27 / 2026-10-01), so its OOF is exp005's 0.8631, not the 3-member 0.8744.

    python scripts/fig_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_common as fc  # noqa: E402

# ---- Panel A: public leaderboard path. (label, public LB, OOF note, kind)
PATH = [
    ("Public DINOv2-S baseline notebook\n(third-party, project start)", 0.809, "not our submission", "ref"),            # CLAUDE.md s5
    ("exp008: distilled DINOv2-S\n(efficiency probe, fold count unknown)", 0.885, "OOF 0.8708", "ours"),                 # METHOD_WRITEUP s11
    ("DINOv2 slot pipeline, probe #1\n(most likely exp005 alone)", 0.893, "OOF 0.8631", "ours"),     # EXPERIMENTS 2026-10-01
    ("exp018: Window-MIL CoAtNet-2\n(5 folds)", 0.929, "OOF 0.8816", "final"),                                          # EXPERIMENTS 2026-09-22
    ("exp021: 10 checkpoints\n(final, main track)", 0.929, "OOF 0.8840", "final"),                                       # EXPERIMENTS 2026-09-25
    ("exp018 2-fold\n(final, efficiency track)", 0.925, "~20.5 min on T4", "final"),                                      # EXPERIMENTS 2026-09-26
]
LB_TOP = (0.95, 0.96)   # METHOD_WRITEUP s11, "public leaderboard top at time of writing" (2026-09-16/17)

# ---- Panel B: (label, delta OOF macro, ci_lo, ci_hi or None, adopted?, source)
GROUPS = [
    ("Slot family vs exp006 (OOF 0.8681)", [
        ("448 px (vs 384)", -0.0001, -0.0024, 0.0025, False),                       # ledger C, exp015
        ("T = 32 slices (vs 24)", -0.0001, -0.0022, 0.0025, False),                  # ledger C, exp010
        ("ACL-aware head init", 0.0010, -0.0010, 0.0032, False),                     # exp010/012/015 results, exp012
        ("Report-embedding aux loss, w 0.1", 0.0036, 0.0012, 0.0061, False),         # ledger C, exp016a
        ("Report-embedding aux loss, w 0.5", 0.0012, -0.0016, 0.0037, False),        # ledger C, exp016b
    ]),
    ("Ensembling vs 3-member slot ensemble (OOF 0.8744)", [
        ("Laterality-swap TTA", -0.0014, None, None, False),                         # ledger D
        ("Per-label recipe (vs global)", 0.0010, None, None, False),                 # ledger D
        ("6-member re-selected (in-sample)", 0.0020, 0.0011, 0.0027, False),         # ledger D
    ]),
    ("Window-MIL vs exp018 (OOF 0.8816)", [
        ("Dense windows at inference (87 vs 51)", -0.0001, -0.0013, 0.0010, False),  # EXPERIMENTS 2026-09-26 dense
        ("Second seed alone (seed 43)", -0.0011, -0.0031, 0.0008, False),            # EXPERIMENTS 2026-09-25 exp021
        ("Seed ensemble, 10 checkpoints (exp021)", 0.0024, 0.0014, 0.0033, True),    # EXPERIMENTS 2026-09-25 exp021
    ]),
    ("Blends vs exp021 (OOF 0.8840)", [
        ("+ DINOv2-B Window-MIL (exp022), w 0.5", -0.0032, -0.0045, -0.0019, False), # EXPERIMENTS 2026-09-26 exp022
        ("+ exp005 slot model, alpha 0.1", 0.0009, 0.0005, 0.0012, False),           # EXPERIMENTS 2026-10-01 re-gate
        ("+ 3-member slot ensemble, alpha 0.2", 0.0024, 0.0017, 0.0031, False),      # EXPERIMENTS 2026-09-26 gate
    ]),
]
NOTES = {  # short reasons shown at the right edge
    "+ exp005 slot model, alpha 0.1": ("fails +0.0020 bar", 0.0025),
    "+ 3-member slot ensemble, alpha 0.2": ("gate pass;\n~21 h on T4", 0.0041),
    "Seed ensemble, 10 checkpoints (exp021)": ("adopted", 0.0045),
}


def panel_a(ax) -> None:
    n = len(PATH)
    ys = list(range(n))[::-1]
    ax.axvspan(*LB_TOP, color=fc.C["light"], zorder=0)
    ax.text(0.955, n - 0.35, "public top\n~0.95-0.96", ha="center", va="top", fontsize=5.8, color=fc.C["grey"])
    col = {"ref": fc.C["grey"], "ours": fc.C["blue"], "final": fc.C["vermillion"]}
    for y, (lab, lb, note, kind) in zip(ys, PATH):
        ax.plot([0.78, lb], [y, y], color=fc.C["light"], lw=0.8, zorder=1)
        ax.plot(lb, y, "o", ms=6.5, color=col[kind], mfc=col[kind] if kind != "ref" else "white", mew=1.2, zorder=3)
        ax.text(lb + 0.004, y + 0.05, f"{lb:.3f}", va="center", fontsize=7.2, fontweight="bold", color=col[kind])
        ax.text(0.782, y - 0.38, note, fontsize=5.8, color=fc.C["grey"], va="center")
    ax.set_yticks(ys, [p[0] for p in PATH], fontsize=6.3)
    ax.set_xlim(0.78, 0.97); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Public leaderboard macro-AUC")
    ax.set_title("A  Public leaderboard path", loc="left", fontweight="bold")
    ax.annotate("", xy=(0.929, ys[3] + 0.3), xytext=(0.893, ys[2] - 0.3),
                arrowprops=dict(arrowstyle="->", color=fc.C["black"], lw=0.8))
    ax.text(0.915, (ys[2] + ys[3]) / 2 + 0.1, "+0.036\nwindow-MIL pipeline\n(changes bundled)", fontsize=5.8,
            va="center", ha="left", fontweight="bold")
    ax.grid(axis="x", color=fc.C["light"], lw=0.4)
    ax.set_axisbelow(True)


def panel_b(ax) -> None:
    y = 0.0
    yt, yl = [], []
    for g, (title, rows) in enumerate(GROUPS):
        ax.text(-0.0098, y, title, fontsize=6.3, fontweight="bold", color=fc.C["black"], va="center", zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.8))
        top = y
        y -= 1.0
        for lab, d, lo, hi, adopted in rows:
            c = fc.C["green"] if adopted else (fc.C["vermillion"] if d < 0 and lo is not None and hi < 0 else fc.C["blue"])
            if lo is not None:
                ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="butt", zorder=2)
            ax.plot(d, y, "D" if adopted else "o", ms=4.6, color=c, mfc=c if adopted else "white", mew=1.1, zorder=3)
            yt.append(y); yl.append(lab)
            if lab in NOTES:
                ax.text(NOTES[lab][1], y, NOTES[lab][0], fontsize=5.6, color=c, va="center", ha="left",
                        fontweight="bold", linespacing=1.1)
            y -= 1.0
        if g == 0:   # pre-registered bar only applies to the slot-family block
            ax.plot([0.0082, 0.0082], [top - 0.4, y + 0.6], color=fc.C["grey"], lw=0.8, ls=":")
            ax.text(0.0084, top - 0.55, "pre-registered\nbar +0.0082", fontsize=5.4, color=fc.C["grey"], va="top", ha="left")
        y -= 0.45
    ax.axvline(0, color=fc.C["black"], lw=0.7)
    ax.set_yticks(yt, yl, fontsize=6.2)
    ax.set_xlim(-0.0098, 0.0098)
    ax.set_ylim(y + 0.3, 0.7)
    ax.set_xlabel("Change in OOF macro-AUC vs baseline (95% paired CI)")
    ax.set_title("B  What else was tried", loc="left", fontweight="bold")
    ax.grid(axis="x", color=fc.C["light"], lw=0.4)
    ax.set_axisbelow(True)


def main() -> None:
    fc.set_style()
    fig = plt.figure(figsize=(7.16, 4.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=1.05)
    panel_a(fig.add_subplot(gs[0]))
    panel_b(fig.add_subplot(gs[1]))
    fig.text(0.0, -0.035, "Hollow = null or rejected, filled green diamond = adopted, red = significantly negative. "
             "No CI drawn where none was computed.",
             fontsize=5.8, color=fc.C["grey"], ha="left")
    fc.save(fig, "ablation_path_and_nulls")
    plt.close(fig)


if __name__ == "__main__":
    main()
