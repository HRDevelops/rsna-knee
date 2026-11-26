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

"""Silent demo clip for slide 11 (the presenter narrates): renders the outputs saved by scripts/demo_study.py.
Per study: input series -> 12 predictions next to the expert labels -> the four most-attended windows for one
finding. The finding shown is fixed by rule: the expert-positive finding with the highest predicted probability, or,
if the study has one expert-positive finding, that finding (so a miss is shown as a miss).

    python scripts/make_demo_video.py     # artifacts/demo/*.npz -> docs/presentation/Team10_demo.mp4 (gitignored)
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path.home() / ".claude/skills/sci-paper-figures/scripts"))
from plot_style import PAL, use_style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "artifacts/demo"
FR = DEMO / "frames"
OUT = ROOT / "docs/presentation/Team10_demo.mp4"
SLOT_LABEL = ["Sagittal FS", "Sagittal", "Coronal FS", "Coronal", "Axial"]
SLOT_LEN = [18, 14, 12, 8, 12]
SLOT_OFF = [0, 18, 32, 44, 52]
use_style(base_size=16)
plt.rcParams["axes.grid"] = False


def canvas(title, sub=None):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.93, title, fontsize=30, fontweight="bold", color="#0F172A")
    if sub:
        fig.text(0.04, 0.885, sub, fontsize=17, color="#475569")
    fig.text(0.04, 0.03, "Team 10 · live demo · efficiency-track model (2 checkpoints, as submitted) · "
             "gold-58 studies: never used for training or model selection", fontsize=12, color="#64748B")
    return fig


def save(fig, name, frames):
    p = FR / f"{len(frames):02d}_{name}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    frames.append(p)


def main():
    FR.mkdir(parents=True, exist_ok=True)
    for p in FR.glob("*.png"):
        p.unlink()
    d = np.load(DEMO / "demo_studies.npz")
    meta = json.loads((DEMO / "demo_meta.json").read_text())
    T = meta["targets"]
    frames, durations = [], []

    fig = canvas("Demo: three knee MRI studies the model never saw")
    fig.text(0.04, 0.70, "Picked by a fixed rule on the expert labels, before any prediction was viewed:", fontsize=20)
    for i, s in enumerate(meta["studies"]):
        fig.text(0.07, 0.61 - i * 0.075, f"Study {i + 1}: {s['rule']}", fontsize=20, color="#0369A1")
    fig.text(0.04, 0.30, "For each study:  DICOM series in  →  12 probabilities vs the radiologists  →  "
             "the slices the model looked at", fontsize=20, fontweight="bold")
    save(fig, "title", frames); durations.append(6)

    for i, s in enumerate(meta["studies"]):
        st, pres, cs = d[f"s{i}_stack"], d[f"s{i}_present"], d[f"s{i}_centers"]
        prob, gold, att, slot_of = d[f"s{i}_prob"], d[f"s{i}_gold"], d[f"s{i}_attn"], d[f"s{i}_slot_of"]

        # 1. input
        fig = canvas(f"Study {i + 1} · input", f"{s['n_series']} DICOM series, {s['n_dicom']} slice files  →  "
                     f"decoded into 5 anatomical slots in {s['decode_s']:.1f} s  (right knees mirrored, 140 mm field)")
        for k in range(5):
            ax = fig.add_axes([0.04 + k * 0.188, 0.22, 0.175, 0.56])
            if pres[k]:
                ax.imshow(st[SLOT_OFF[k] + SLOT_LEN[k] // 2], cmap="gray")
            else:
                ax.imshow(np.zeros((336, 336)), cmap="gray", vmin=0, vmax=1)
                ax.text(168, 168, "not acquired", color="white", ha="center", va="center", fontsize=16)
            ax.set_title(SLOT_LABEL[k], fontsize=17)
            ax.axis("off")
        save(fig, f"s{i}_input", frames); durations.append(6)

        # 2. predictions vs expert labels
        fig = canvas(f"Study {i + 1} · 12 predictions vs the radiologists",
                     f"{s['n_windows']} windows of 3 adjacent slices → CoAtNet-2 → attention MIL · both models: "
                     f"{s['model_s']:.2f} s on one {meta['gpu'].split(' ')[1]} GPU" + (" (incl. GPU warm-up)" if i == 0 else ""))
        ax = fig.add_axes([0.20, 0.12, 0.62, 0.70])
        y = np.arange(12)[::-1]
        cols = [PAL["green"] if g else PAL["grey_fill"] for g in gold]
        ax.barh(y, prob, color=cols, edgecolor=[PAL["green"] if g else PAL["grey"] for g in gold], height=0.7)
        for yy, p_, g in zip(y, prob, gold):
            ax.text(p_ + 0.01, yy, f"{p_:.2f}", va="center", fontsize=15)
            ax.text(1.13, yy, "expert: positive" if g else "expert: negative", va="center", fontsize=15,
                    color=PAL["green"] if g else PAL["grey"], fontweight="bold" if g else None)
        ax.set_yticks(y, T, fontsize=16)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Predicted probability (the metric is AUC: what matters is the ranking across studies)",
                      fontsize=14)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        save(fig, f"s{i}_pred", frames); durations.append(9)

        # 3. evidence: top-4 windows for the rule-chosen finding
        pos = np.where(gold == 1)[0]
        f = pos[np.argmax(prob[pos])] if len(pos) > 1 else (pos[0] if len(pos) else int(np.argmax(prob)))
        top = np.argsort(att[:, f])[::-1][:4]
        share = att[top, f].sum()
        verdict = ("expert-positive" if gold[f] else "expert-negative") + f", predicted {prob[f]:.2f}"
        if gold[f] and prob[f] < 0.5:
            verdict += "  →  a miss"
        fig = canvas(f"Study {i + 1} · where the model looked for {T[f]}",
                     f"{T[f]}: {verdict} · the 4 most-attended of {len(cs)} windows carry {share:.0%} of the "
                     f"attention")
        for j, w in enumerate(top):
            ax = fig.add_axes([0.04 + j * 0.235, 0.17, 0.215, 0.62])
            ax.imshow(st[cs[w]], cmap="gray")
            k = int(slot_of[w])
            ax.set_title(f"#{j + 1}  {SLOT_LABEL[k]}, slice {cs[w] - SLOT_OFF[k] + 1}/{SLOT_LEN[k]}\n"
                         f"attention {att[w, f]:.2f}", fontsize=15)
            ax.axis("off")
        save(fig, f"s{i}_attn", frames); durations.append(8)

    fig = canvas("In the competition")
    fig.text(0.04, 0.66, "The same code ran offline in a Kaggle notebook on the hidden test set:", fontsize=22)
    fig.text(0.07, 0.56, "• about 1,300 studies, about 20 minutes on 2 × T4 GPUs (this 2-checkpoint model)",
             fontsize=22)
    fig.text(0.07, 0.48, "• main submission: 10 checkpoints, about 100 minutes, private AUC 0.912", fontsize=22)
    fig.text(0.04, 0.32, "Attention gives a radiologist a handful of slices to check instead of the whole exam.",
             fontsize=22, fontweight="bold", color="#0369A1")
    save(fig, "end", frames); durations.append(6)

    lst = DEMO / "frames.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\nduration {t}\n" for p, t in zip(frames, durations))
                   + f"file '{frames[-1].as_posix()}'\n")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf",
                    "fps=30,format=yuv420p", "-c:v", "libx264", "-tune", "stillimage", str(OUT)], check=True)
    print(f"wrote {OUT.name}: {sum(durations)} s, {len(frames)} frames")


if __name__ == "__main__":
    main()
