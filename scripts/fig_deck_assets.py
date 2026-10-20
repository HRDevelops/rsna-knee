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

"""Figures made at slide size for the progress deck (fonts are in points at the size they are placed, so 12 pt here
is 12 pt on the slide):

  deck_mri_strip     one real knee study: sagittal, coronal, axial slice (from the attention-extraction stacks)
  deck_data_scale    4,407 studies drawn to scale, with the 58 expert-labelled ones
  deck_models_ladder out-of-fold macro-AUC of every model that was trained and scored (numbers: docs/EXPERIMENTS.md
                     consolidated ablation ledger, exp018/021 results, exp022 result)
  deck_lane_before / deck_lane_after   crops of the formulation diagram (baseline lane, window-MIL lane)

    python scripts/fig_deck_assets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_common as fc  # noqa: E402

SLOT_OFF, SLOT_LEN = [0, 18, 32, 44, 52], [18, 14, 12, 8, 12]


def mri_strip() -> None:
    z = np.load(fc.DATA / "attention_gold58.npz")
    uids = sorted({k.split("|")[0] for k in z.files if k.endswith("|stack")})
    pick = next(u for u in uids if all(z[f"{u}|present"][i] for i in (0, 2, 4)))
    st = z[f"{pick}|stack"]
    tiles = [("Sagittal", st[SLOT_OFF[0] + 9]), ("Coronal", st[SLOT_OFF[2] + 6]), ("Axial", st[SLOT_OFF[4] + 6])]
    fig, axes = plt.subplots(1, 3, figsize=(4.95, 1.85))
    for ax, (name, img) in zip(axes, tiles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(name, fontsize=12, pad=3, color="#374151")
    fig.subplots_adjust(left=0, right=1, top=0.88, bottom=0, wspace=0.03)
    fc.save(fig, "deck_mri_strip")
    plt.close(fig)


def data_scale() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 1.55))
    n, g = 4407, 58
    ax.barh([0], [n - g], left=[g], color="#CBD5E1", height=0.5)
    ax.barh([0], [g], left=[0], color=fc.C["vermillion"], height=0.5)
    ax.set_xlim(-60, n + 60); ax.set_ylim(-0.55, 0.9); ax.axis("off")
    ax.text(n / 2, 0.0, "4,407 training studies, each with a radiology report", ha="center", va="center", fontsize=15, color="#374151")
    ax.annotate("58 with expert image-review labels (1.3%)", xy=(g / 2, 0.25), xytext=(g + 40, 0.72), fontsize=15,
                fontweight="bold", color=fc.C["vermillion"], va="center",
                arrowprops=dict(arrowstyle="-", color=fc.C["vermillion"], lw=1.2))
    ax.text(n, -0.5, "drawn to scale", ha="right", va="center", fontsize=11, color="#9CA3AF")
    fc.save(fig, "deck_data_scale")
    plt.close(fig)


ROWS = [  # (label, OOF macro-AUC, family, public LB note)
    ("exp002  ViT-S, report labels", 0.8489, "slot", ""),
    ("exp003  ViT-B, report labels", 0.8536, "slot", ""),
    ("exp005  ViT-S, v3 labels", 0.8631, "slot", "0.893\n(most likely this one)"),
    ("exp006  ViT-B, v3 labels", 0.8681, "slot", ""),
    ("exp009  ConvNeXt-V2-B", 0.8652, "slot", ""),
    ("exp004  DINOv3-S", 0.8237, "rej", "worse: dropped"),
    ("exp008  distilled ViT-S", 0.8708, "slot", "0.885"),
    ("3-model ensemble", 0.8744, "slot", ""),
    ("exp022  DINOv2-B windows", 0.8709, "win", ""),
    ("exp018  CoAtNet-2 windows", 0.8816, "win", "0.929"),
    ("exp021  CoAtNet-2, 2 seeds", 0.8840, "win", "0.929"),
]


def models_ladder() -> None:
    fig, ax = plt.subplots(figsize=(7.7, 4.75))
    col = {"slot": fc.C["blue"], "win": fc.C["vermillion"], "rej": "#9CA3AF"}
    ys = np.arange(len(ROWS))[::-1].astype(float)
    ys[:8] += 0.0
    ys[8:] -= 0.0
    gap_after = 8                       # a little space between the two families
    ys = np.array([y + (0.6 if i < gap_after else 0) for i, y in enumerate(ys)])
    for y, (lab, v, fam, note) in zip(ys, ROWS):
        ax.plot([0.82, v], [y, y], color="#E5E7EB", lw=1.2, zorder=1)
        ax.plot(v, y, "o", ms=9, color=col[fam], mfc=col[fam] if fam != "rej" else "white", mew=1.6, zorder=3)
        ax.text(v + 0.0012, y + 0.02, f"{v:.4f}", va="center", fontsize=11, color=col[fam], fontweight="bold")
        if note:    # own column to the right of the axes, so it can never touch a value label
            ax.text(1.03, y, note, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=10, color="#6B7280",
                    clip_on=False, linespacing=1.15)
    ax.set_yticks(ys, [r[0] for r in ROWS], fontsize=11)
    ax.set_xlim(0.82, 0.8975); ax.set_ylim(-0.7, ys.max() + 0.9)
    ax.set_xlabel("Out-of-fold macro-AUC (same 5 folds for every model)", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", color="#F3F4F6")
    ax.set_axisbelow(True)
    ax.text(0.8205, ys[0] + 0.62, "DINOv2 / slot family", fontsize=11, fontweight="bold", color=fc.C["blue"], va="center")
    ax.text(0.8205, ys[8] + 0.62, "Window-attention family", fontsize=11, fontweight="bold", color=fc.C["vermillion"], va="center")
    ax.text(1.03, ys[0] + 0.62, "public leaderboard", transform=ax.get_yaxis_transform(), fontsize=10, fontweight="bold",
            color="#6B7280", va="center", clip_on=False)
    fig.subplots_adjust(left=0.30, right=0.76, top=0.97, bottom=0.12)
    fc.save(fig, "deck_models_ladder")
    plt.close(fig)


def lanes() -> None:
    im = Image.open(fc.FIG / "formulation_before_after.png")
    w, h = im.size
    sx, sy = w / 2362, h / 1284        # crop boxes were measured on the 2362 x 1284 render
    im.crop((0, 0, int(2150 * sx), int(530 * sy))).save(fc.FIG / "deck_lane_before.png")
    im.crop((0, int(585 * sy), int(2150 * sx), int(1120 * sy))).save(fc.FIG / "deck_lane_after.png")
    print("wrote docs/figures/deck_lane_before.png + deck_lane_after.png")


if __name__ == "__main__":
    fc.set_style()
    mri_strip()
    data_scale()
    models_ladder()
    lanes()
