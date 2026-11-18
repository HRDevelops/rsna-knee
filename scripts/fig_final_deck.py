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

"""Result figures for the final course presentation and the publishability audit (2026-11-18).

Every number is copied from docs/EXPERIMENTS.md (our runs) or from the public Kaggle solution writeups (read
2026-11-18; marked "writeup" below). No number is computed here except means of two seeds.

    python scripts/fig_final_deck.py          # writes docs/figures/final/*.png and *.pdf
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path.home() / ".claude/skills/sci-paper-figures/scripts"))
from plot_style import PAL, use_style  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs/figures/final"
OUT.mkdir(parents=True, exist_ok=True)
use_style(base_size=9)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=250 if ext == "png" else None)
    plt.close(fig)


# ---------------------------------------------------------------- 1. leaderboard context and the label effect
# Private LB, from the writeups (1st place: final private not stated in the writeup; the top private score is 0.960).
lb = [("2nd (VLM image labels)", 0.960), ("3rd (VLM labels + OAI)", 0.959), ("4th (VLM labels, CoAtNet MIL)", 0.958),
      ("6th (report labels + MOAKS)", 0.958), ("7th (VLM labels, no ext. data)", 0.957),
      ("10th (LLM/VLM labels)", 0.957), ("Public-notebook plateau", 0.942), ("Ours (report labels, CoAtNet MIL)", 0.912)]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.25, 1]}, constrained_layout=True)
names, vals = zip(*lb)
y = np.arange(len(lb))[::-1]
cols = [PAL["grey_fill"]] * 6 + [PAL["blue_fill"], PAL["orange"]]
a.barh(y, vals, color=cols, edgecolor=[PAL["grey"]] * 6 + [PAL["blue"], PAL["orange"]], linewidth=0.8, height=0.65)
for yi, v in zip(y, vals):
    a.text(v + 0.0015, yi, f"{v:.3f}", va="center", fontsize=8)
a.set_yticks(y, names)
a.set_xlim(0.90, 0.968)
a.set_xlabel("Private leaderboard macro-AUC (about 1,300 expert-labelled test studies)")
a.set_title("(a) Where we finished: rank 3,970 / 6,425", loc="left", fontweight="bold")
a.grid(axis="y", visible=False)

# Same-architecture label swaps reported in writeups, plus ours.
pairs = [("4th place, same CoAtNet-2\nwindow-MIL (one fold)", 0.925, 0.952, "report labels", "VLM image-read labels"),
         ("13th place, CoAtNet-224", 0.906, 0.921, "hard 0/1 labels", "graded soft labels")]
for i, (n, lo, hi, l1, l2) in enumerate(pairs):
    yy = 1 - i
    b.plot([lo, hi], [yy, yy], color=PAL["grey"], lw=1.2, zorder=1)
    b.scatter([lo], [yy], color=PAL["grey"], s=40, zorder=2)
    b.scatter([hi], [yy], color=PAL["green"], s=40, zorder=2)
    b.text(lo, yy - 0.3, l1, ha="center", fontsize=7.5, color=PAL["grey"])
    b.text(hi, yy + 0.18, l2, ha="center", fontsize=7.5, color=PAL["green"])
    b.text(hi + 0.002, yy - 0.05, f"+{hi - lo:.3f}", fontsize=8, color=PAL["green"], va="center")
b.scatter([0.912], [-1], color=PAL["orange"], s=50, zorder=2)
b.text(0.912, -0.82, "ours: report labels", ha="center", fontsize=7.5, color=PAL["orange"])
b.set_yticks([1, 0, -1], [p[0] for p in pairs] + ["Ours, CoAtNet-2\nwindow-MIL (10 ckpt)"])
b.set_ylim(-1.5, 1.6)
b.set_xlim(0.90, 0.965)
b.set_xlabel("Private leaderboard macro-AUC")
b.set_title("(b) Same model, different labels", loc="left", fontweight="bold")
b.grid(axis="y", visible=False)
save(fig, "fig1_leaderboard_labels")

# ---------------------------------------------------------------- 2. preprocessing ladder and ACL forensics
rungs = ["B0\nexp018", "B1\nexact\npixels", "B2\nnative\nwindows", "B3\nall\nseries", "B4\nplane\nbudget", "B5\n160 mm\nfield"]
mac = np.array([[0.9567, 0.9547], [0.9562, 0.9523], [0.9544, 0.9551], [0.9546, 0.9566], [0.9567, 0.9568],
                [0.9569, 0.9567]])
acl = np.array([[0.9533, 0.9317], [0.9400, 0.9247], [0.9198, 0.9199], [0.9209, 0.9197], [0.9093, 0.9124],
                [0.9303, 0.9176]])
fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]},
                              constrained_layout=True)
x = np.arange(6)
for ax, d, t, yl in ((a, mac, "(a) Macro-AUC: flat within seed noise", (0.950, 0.960)),
                     (b, acl, "(b) ACL alone: drops, but seeds differ by 0.022", (0.90, 0.96))):
    ax.scatter(np.repeat(x, 2) + np.tile([-0.08, 0.08], 6), d.ravel(), s=18, color=PAL["blue_fill"],
               edgecolor=PAL["blue"], linewidth=0.6, label="seed 42 / 43", zorder=2)
    ax.plot(x, d.mean(1), color=PAL["blue"], marker="o", ms=4, label="seed mean", zorder=3)
    ax.set_xticks(x, rungs, fontsize=7.5)
    ax.set_ylim(*yl)
    ax.set_title(t, loc="left", fontweight="bold")
    ax.set_ylabel("Fold-0 validation AUC")
a.legend(loc="lower right", fontsize=7.5)
# Triplet-spacing test (pre-registered), paired bootstrap vs B5.
arms = ["±7 mm", "±10.5 mm"]
d_acl = [(-0.0074, -0.0221, 0.0073), (-0.0219, -0.0384, -0.0065)]
d_mac = [(-0.0020, -0.0053, 0.0008), (-0.0034, -0.0059, -0.0013)]
for i, (v, lo, hi) in enumerate(d_acl):
    c.errorbar(i - 0.1, v, yerr=[[v - lo], [hi - v]], fmt="o", color=PAL["red"], capsize=3,
               label="ACL" if i == 0 else None)
for i, (v, lo, hi) in enumerate(d_mac):
    c.errorbar(i + 0.1, v, yerr=[[v - lo], [hi - v]], fmt="s", color=PAL["grey"], capsize=3,
               label="macro" if i == 0 else None)
c.axhline(0, color=PAL["ink"], lw=0.8)
c.set_xticks([0, 1], arms)
c.set_xlim(-0.5, 1.5)
c.set_ylabel("Δ AUC vs B5 (95% cluster bootstrap)")
c.set_title("(c) Wider triplets hurt ACL", loc="left", fontweight="bold")
c.legend(fontsize=7.5, loc="lower left")
save(fig, "fig2_ladder_acl")

# ---------------------------------------------------------------- 3. geometry experiments: causal controls
fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.2), constrained_layout=True)
groups = ["Final-stage\ntokens (Phase 1)", "Local tokens\n(follow-up)"]
dgeo = [(-0.0044, 0.0005), (0.0044, -0.0087)]
dswap = [(0.0000, 0.0000), (0.00002, 0.0000)]
boff = [(0.0000, 0.0000), (0.0002, 0.0001)]
w = 0.25
for gi in range(2):
    for k, (vals, col, lab) in enumerate(((dgeo, PAL["blue"], "Δgeo = G − B (paired baseline)"),
                                          (dswap, PAL["orange"], "Δswap = true − donor geometry"),
                                          (boff, PAL["green"], "β-off = G − G(β=0)"))):
        xs = gi + (k - 1) * w + np.array([-0.04, 0.04])
        a.scatter(xs, vals[gi], color=col, s=26, label=lab if gi == 0 else None, zorder=3)
a.axhline(0, color=PAL["ink"], lw=0.8)
a.axhspan(-0.004, 0.004, color=PAL["grey_fill"], alpha=0.35, lw=0, label="±0.004 (≈ 1 bootstrap SD of macro)")
a.set_xticks([0, 1], groups)
a.set_ylabel("Δ macro-AUC, fold 0 (dots = seeds 42, 43)")
a.set_title("(a) Correct geometry adds nothing", loc="left", fontweight="bold")
a.legend(fontsize=7, loc="lower left")
# Mechanism: bias strength (median, p5) in logits.
med = [(-0.77, -0.67), (-3.3, -3.6)]
p5 = [(-4.0, -4.3), (-18.6, -18.7)]
for gi in range(2):
    b.bar(gi - 0.17, -np.mean(med[gi]), width=0.3, color=PAL["teal_fill"], edgecolor=PAL["teal"],
          label="median |bias|" if gi == 0 else None)
    b.bar(gi + 0.17, -np.mean(p5[gi]), width=0.3, color=PAL["red_fill"], edgecolor=PAL["red"],
          label="5th-percentile |bias|" if gi == 0 else None)
b.set_xticks([0, 1], ["Final-stage: pathway dormant", "Local: active but uninformative"])
b.set_yscale("log")
b.set_ylabel("Cross-plane attention bias (logits)")
b.set_title("(b) The model learns locality, and it does not help", loc="left", fontweight="bold")
b.legend(fontsize=7.5, loc="upper left")
save(fig, "fig3_geometry_controls")

# ---------------------------------------------------------------- 4. label noise against gold-58
T = ["ACL", "MCL", "Med. men.", "Lat. men.", "Med. OA", "Lat. OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
eta_p = [0.06, 0.04, 0.06, 0.09, 0.05, 0.04, 0.05, 0.13, 0.16, 0.02, 0.08, 0.10]
eta_m = [0.08, 0.44, 0.12, 0.22, 0.27, 0.55, 0.29, 0.03, 0.30, 0.25, 0.42, 0.39]
npos = [24, 9, 26, 23, 15, 11, 21, 35, 27, 12, 19, 18]
fig, ax = plt.subplots(figsize=(10, 3.0), constrained_layout=True)
x = np.arange(12)
ax.bar(x - 0.2, eta_m, 0.4, color=PAL["red_fill"], edgecolor=PAL["red"], label="missed positives η⁻ (label 0, gold 1)")
ax.bar(x + 0.2, eta_p, 0.4, color=PAL["blue_fill"], edgecolor=PAL["blue"],
       label="false positives η⁺ (label 1, gold 0)")
for xi, n in zip(x, npos):
    ax.text(xi, -0.07, f"n+={n}", ha="center", fontsize=7, color=PAL["grey"])
ax.set_xticks(x, T, fontsize=8)
ax.set_ylim(-0.1, 0.62)
ax.set_ylabel("Error rate vs gold-58")
ax.set_title("Our training labels (v3) against the 58 expert-read studies: errors are mostly missed positives",
             loc="left", fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
save(fig, "fig4_label_noise")
print("wrote", sorted(p.name for p in OUT.iterdir()))
