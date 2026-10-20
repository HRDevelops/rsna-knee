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

"""End-to-end system overview for the introduction slide: the report -> label engine -> soft labels lane (training
only) and the image-only pipeline (training and test). Schematic; every number is read from README.md /
docs/METHOD_WRITEUP.md. Coordinates are inches (axes span the 7.16 x 3.75 in figure 1:1).

    python scripts/fig_system.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_common as fc  # noqa: E402

W, H = 7.16, 3.75
BH = 0.82
TLH, BLH = 0.125, 0.105


def box(ax, x, y, w, title, body="", edge=fc.C["grey"], lw=0.9, fs=6.0, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y - BH / 2), w, BH, boxstyle="round,pad=0,rounding_size=0.06", fc=fill, ec=edge, lw=lw))
    nt, nb = len(title.split("\n")), len(body.split("\n")) if body else 0
    block = nt * TLH + (0.05 + nb * BLH if nb else 0)
    top = y + block / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=fs + 0.9, fontweight="bold", linespacing=1.15)
    if body:
        ax.text(x + w / 2, top - nt * TLH - 0.05, body, ha="center", va="top", fontsize=fs, linespacing=1.3)


def arrow(ax, p0, p1, color=fc.C["grey"], ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=color, lw=1.0, linestyle=ls))


def lane(ax, x0, ys, specs, gap, hot=None, fs=6.0):
    x, pos = x0, []
    for w, t, b, e, lw in specs:
        box(ax, x, ys, w, t, b, edge=e, lw=lw, fs=fs)
        pos.append((x, w))
        x += w + gap
    for i in range(len(specs) - 1):
        arrow(ax, (pos[i][0] + pos[i][1] + 0.02, ys), (pos[i + 1][0] - 0.02, ys), hot if hot and i == len(specs) - 2 else fc.C["grey"])
    return pos


def main() -> None:
    fc.set_style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    G, B, V, GR = fc.C["grey"], fc.C["blue"], fc.C["vermillion"], fc.C["green"]

    ax.text(0.03, 3.66, "TRAINING ONLY: turning reports into labels (reports do not exist at test time)", fontsize=6.6,
            fontweight="bold", color=V, va="top")
    ytop = 2.95
    top = lane(ax, 0.03, ytop, [
        (1.45, "Radiology reports", "4,407 studies\n12+ languages", G, 0.9),
        (2.05, "Label engine", "Qwen2.5-72B + Llama-3.3-70B,\nrubric of expert thresholds,\ncalibrated on the 58 real labels", V, 1.5),
        (1.5, "Soft labels (v3)", "12 probabilities / study\n(one noisy-student round)", G, 0.9),
        (1.45, "Training loss", "BCE vs soft labels", G, 0.9),
    ], gap=0.2)

    ax.plot([0.03, W - 0.03], [2.28, 2.28], color=fc.C["light"], lw=1.0)
    ax.text(0.03, 2.2, "TRAINING AND TEST: the image-only model (this is all the competition notebook runs)", fontsize=6.6,
            fontweight="bold", color=B, va="top")
    ybot = 1.38
    bot = lane(ax, 0.03, ybot, [
        (0.92, "MRI study", "3-14 series\nDICOM", G, 0.9),
        (1.2, "5 slots, 64 slices", "140 mm FOV, 336 px\nlaterality-fixed", G, 0.9),
        (1.05, "Windows", "3 adjacent slices\nas RGB, <= 54", G, 0.9),
        (0.98, "CoAtNet-2", "fine-tuned;\nfeature / window", G, 0.9),
        (1.2, "Per-finding\nattention", "softmax over ALL\nwindows", GR, 1.5),
        (0.95, "12 probabilities", "ACL, MCL, menisci,\nOA x3, effusion, ...", B, 1.5),
    ], gap=0.155)

    # the soft labels only reach the model through the training loss (dashed), which scores the 12 outputs
    lx, lw_ = top[3]
    ox, ow = bot[5]
    arrow(ax, (lx + lw_ * 0.5, ytop - BH / 2 - 0.02), (ox + ow * 0.5, ybot + BH / 2 + 0.03), V, "--")
    ax.text(lx + lw_ * 0.5 + 0.08, 2.46, "scores the\n12 outputs", fontsize=5.6, color=V, va="center", ha="left", linespacing=1.2)
    ax.text(0.03, 0.62, "Scored by macro ROC-AUC over the 12 findings (mean of 12 per-finding AUCs). Test: ~1,300 hidden "
            "studies, images only.", fontsize=6.3, color=G, va="top")
    ax.text(0.03, 0.4, "Kaggle code competition: notebook with GPU <= 9 h, no internet. A second track scores AUC against runtime.",
            fontsize=6.3, color=G, va="top")
    fc.save(fig, "system_overview")
    plt.close(fig)




def main_wide() -> None:
    """Slide-size version (12.1 x 2.95 in, text 9-10 pt) for the deck's introduction slide."""
    global W, H, BH, TLH, BLH
    W, H, BH, TLH, BLH = 12.1, 3.0, 0.8, 0.165, 0.14
    fc.set_style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    G, B, V, GR = fc.C["grey"], fc.C["blue"], fc.C["vermillion"], fc.C["green"]
    ax.text(0.03, 2.95, "TRAINING ONLY: reports become labels (there are no reports at test time)", fontsize=10.5,
            fontweight="bold", color=V, va="top")
    ytop = 2.3
    top = lane(ax, 0.03, ytop, [
        (2.4, "Radiology reports", "4,407 studies, 12+ languages", G, 0.9),
        (4.0, "Label engine", "Qwen2.5-72B + Llama-3.3-70B read each report;\ncalibrated on the 58 real labels", V, 1.6),
        (2.9, "Soft labels (v3)", "12 probabilities per study", G, 0.9),
        (2.1, "Training loss", "BCE against soft labels", G, 0.9),
    ], gap=0.22, fs=9.0)
    ax.plot([0.03, W - 0.03], [1.72, 1.72], color=fc.C["light"], lw=1.0)
    ax.text(0.03, 1.66, "TRAINING AND TEST: the image-only model (the only part the competition notebook runs)", fontsize=10.5,
            fontweight="bold", color=B, va="top")
    ybot = 0.82
    bot = lane(ax, 0.03, ybot, [
        (1.5, "MRI study", "3-14 series, DICOM", G, 0.9),
        (2.15, "5 slots, 64 slices", "140 mm, 336 px, laterality fixed", G, 0.9),
        (1.85, "Windows", "3 adjacent slices, <= 54", G, 0.9),
        (1.7, "CoAtNet-2", "one feature per window", G, 0.9),
        (2.15, "Per-finding attention", "softmax over ALL windows", GR, 1.6),
        (1.7, "12 probabilities", "macro ROC-AUC", B, 1.6),
    ], gap=0.17, fs=9.0)
    lx, lw_ = top[3]
    ox, ow = bot[5]
    arrow(ax, (lx + lw_ * 0.5, ytop - BH / 2 - 0.02), (ox + ow * 0.5, ybot + BH / 2 + 0.03), V, "--")
    ax.text(lx + lw_ * 0.5 + 0.1, 1.78, "scores the outputs", fontsize=8.5, color=V, va="center", ha="left")
    fc.save(fig, "system_overview_wide")
    plt.close(fig)


if __name__ == "__main__":
    main()
    main_wide()
