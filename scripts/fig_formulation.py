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

"""Before/after diagram of the formulation change: series-CLS compression (DINOv2 slot model, src/rsna_knee/model.py)
vs per-finding window MIL (CoAtNet-2, src/rsna_knee/wmil.py). Pure schematic; every number in the boxes is read from
those two files / docs/METHOD_WRITEUP.md. The 0.893 -> 0.929 jump bundled several changes (backbone family, window
input, FOV, laterality fix); no ablation isolates which one matters, so the diagram states the architectural difference
and does not attribute the gain to it.

Coordinates are inches (the axes span the 7.16 x 4.4 in figure 1:1), so text sizes and box sizes are directly comparable.

    python scripts/fig_formulation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_common as fc  # noqa: E402

W, H = 7.16, 4.4
BOX_H = 0.92
TITLE_LH, BODY_LH = 0.125, 0.108     # line heights in inches for 7 pt bold / 6 pt text


def box(ax, x, y, w, title, body="", edge=fc.C["grey"], lw=0.9, fs=6.0):
    ax.add_patch(FancyBboxPatch((x, y - BOX_H / 2), w, BOX_H, boxstyle="round,pad=0,rounding_size=0.06", fc="white",
                                ec=edge, lw=lw))
    nt, nb = len(title.split("\n")), len(body.split("\n")) if body else 0
    block = nt * TITLE_LH + (0.05 + nb * BODY_LH if nb else 0)
    top = y + block / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=fs + 0.9, fontweight="bold", linespacing=1.15)
    if body:
        ax.text(x + w / 2, top - nt * TITLE_LH - 0.05, body, ha="center", va="top", fontsize=fs, linespacing=1.3)


def arrow(ax, x0, x1, y, color=fc.C["grey"]):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8, color=color, lw=1.0))


def row(ax, y, specs, hot_idx, hot_color):
    """specs: list of (width, title, body, edge, lw). Boxes are laid out left to right with 0.17 in gaps."""
    x, xs = 0.03, []
    for w, t, b, e, lw in specs:
        box(ax, x, y, w, t, b, edge=e, lw=lw)
        xs.append((x, w))
        x += w + 0.17
    for i in range(len(specs) - 1):
        arrow(ax, xs[i][0] + xs[i][1] + 0.02, xs[i + 1][0] - 0.02, y, hot_color if i == hot_idx else fc.C["grey"])
    return xs


def main() -> None:
    fc.set_style()
    fig = plt.figure(figsize=(W, H - 0.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0.2, H); ax.axis("off")
    RED, GRN, G = fc.C["vermillion"], fc.C["green"], fc.C["grey"]

    # ---------------------------------------------------------------- before
    ax.text(0.03, 4.33, "Before: each series is compressed to one token, then slots are mixed", fontsize=8.3,
            fontweight="bold", va="top")
    ax.text(0.03, 4.12, "DINOv2 slot model (exp003/005/006). Public LB 0.893 (most likely exp005 alone); OOF 0.863-0.874.",
            fontsize=6.3, color=G, va="top")
    y1 = 3.4
    row(ax, y1, [
        (0.9, "Study", "3-14 series\n(one plane each)", G, 0.9),
        (1.2, "6 slots", "{Sag, Cor, Ax} x FS\n24 slices, 384 px", G, 0.9),
        (1.2, "DINOv2 ViT", "per slice, then\nslice attention", G, 0.9),
        (1.15, "ONE token\nper series", "(CLS): the bottleneck", RED, 1.8),
        (1.2, "Cross-slot\ntransformer", "then 12 finding\nqueries over slots", G, 0.9),
        (0.55, "12", "logits", G, 0.9),
    ], hot_idx=2, hot_color=RED)
    ax.text(0.03, 2.77, "Everything a series shows must fit in one vector before any finding is considered: "
            "a finding reads 6 summaries, never the slices.", fontsize=6.4, color=RED, va="center")

    ax.plot([0.03, W - 0.03], [2.52, 2.52], color=fc.C["light"], lw=1.0)

    # ----------------------------------------------------------------- after
    ax.text(0.03, 2.43, "After: every window is kept, and each finding attends over all of them", fontsize=8.3,
            fontweight="bold", va="top")
    ax.text(0.03, 2.22, "CoAtNet-2 Window-MIL (exp018, exp021). Public LB 0.929; OOF 0.8816-0.8840.", fontsize=6.3,
            color=G, va="top")
    y2 = 1.5
    xs = row(ax, y2, [
        (0.9, "Study", "3-14 series\n(one plane each)", G, 0.9),
        (1.2, "5 slots, 64 slices", "140 mm FOV, 336 px\nlaterality-normalised", G, 0.9),
        (1.2, "Windows", "3 adjacent slices\nas RGB, up to 54", G, 0.9),
        (1.15, "CoAtNet-2", "fully fine-tuned;\none feature / window", G, 0.9),
        (1.2, "Per-finding\nattention MIL", "softmax over ALL\nwindows, per finding", GRN, 1.8),
        (0.55, "12", "logits", G, 0.9),
    ], hot_idx=3, hot_color=GRN)
    wx, ww = xs[2]                                       # little row of windows, a few highlighted (attended)
    s, g, n = 0.085, 0.022, 8
    x0 = wx + (ww - (n * s + (n - 1) * g)) / 2
    for i in range(n):
        ax.add_patch(Rectangle((x0 + i * (s + g), y2 - 0.37), s, s * 1.2, fc=fc.C["orange"] if i in (2, 3, 6) else fc.C["light"],
                               ec=G, lw=0.4))
    ax.text(0.03, 0.82, "No per-series token and no slot bottleneck: each finding pools directly from the windows it finds "
            "informative (see the attention figures).", fontsize=6.4, color=GRN, va="center")

    ax.text(0.03, 0.55, "The 0.893 -> 0.929 change bundled four things (backbone family, window input, 140 mm FOV, corrected "
            "laterality); no ablation isolates which one matters.\nDINOv2-B placed inside the window pipeline (exp022) reached OOF "
            "0.8709 vs 0.8816 for CoAtNet-2, so the pipeline, more than the backbone, sets what the model sees.",
            fontsize=5.8, color=G, va="top", linespacing=1.45)
    fc.save(fig, "formulation_before_after")
    plt.close(fig)


if __name__ == "__main__":
    main()
