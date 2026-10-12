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

"""Build the course progress presentation (5 slides + 1 backup, 16:9, speaker notes) from docs/figures/.

Framing: a progress update ("so far", "current best", "what is left"). Every statement is true as of the build date;
nothing claims an experiment is running or a result is pending when it is not. Numbers are the ones in
docs/CV_PROGRESS_PRESENTATION.md (sources listed there).

    python scripts/build_progress_deck.py        # -> docs/presentation/progress_presentation.pptx
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs/figures"
OUT = ROOT / "docs/presentation/progress_presentation.pptx"

INK, GREY, LIGHT = RGBColor(0x11, 0x18, 0x27), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xF3, 0xF4, 0xF6)
BLUE, VERM, GREEN = RGBColor(0x00, 0x72, 0xB2), RGBColor(0xD5, 0x5E, 0x00), RGBColor(0x00, 0x9E, 0x73)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"
W, H = 13.333, 7.5


def tb(slide, x, y, w, h, paras, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
       spacing=1.1):
    """Text box. paras: list of str or (str, dict(size=, color=, bold=, bullet=)) items."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    first = True
    for p in paras:
        text, o = (p, {}) if isinstance(p, str) else p
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = align
        para.line_spacing = spacing
        para.space_after = Pt(o.get("after", 6))
        bullet = o.get("bullet", False)
        run = para.add_run()
        run.text = ("•  " + text) if bullet else text
        f = run.font
        f.name, f.size = FONT, Pt(o.get("size", size))
        f.bold, f.color.rgb = o.get("bold", bold), o.get("color", color)
    return box


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    return s


def base(prs, title, kicker, notes, n):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(slide, 0, 0, W, 0.12, BLUE)
    tb(slide, 0.55, 0.28, 12.2, 0.35, [kicker.upper()], size=12, color=GREY, bold=True)
    tb(slide, 0.55, 0.58, 12.2, 0.9, [title], size=30, bold=True, anchor=MSO_ANCHOR.TOP)
    tb(slide, 0.55, 7.08, 9, 0.3, ["Progress update · RSNA Knee Abnormality Detection 2026 · Husnain Rasool"], size=10,
       color=GREY)
    tb(slide, 12.2, 7.08, 0.6, 0.3, [str(n)], size=10, color=GREY, align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def pic(slide, name, x, y, max_w, max_h):
    """Place a figure scaled to fit inside (max_w x max_h) inches, centred in that box."""
    path = FIG / name
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (max_w - w) / 2), Inches(y + (max_h - h) / 2), Inches(w), Inches(h))


def callout(slide, x, y, w, h, head, lines, color, size=14):
    rect(slide, x, y, w, h, LIGHT)
    rect(slide, x, y, 0.07, h, color)
    tb(slide, x + 0.2, y + 0.1, w - 0.3, 0.4, [head], size=size + 1, color=color, bold=True)
    tb(slide, x + 0.2, y + 0.55, w - 0.3, h - 0.6, [(t, dict(size=size, after=7)) for t in lines], size=size)


def main() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # ------------------------------------------------------------------ 1
    s = base(prs, "Reading 12 knee-MRI findings from only 58 real labels", "Where the project started", (
        "I predict twelve knee-MRI findings, scored by macro AUC. The catch is the labels: of 4,407 training studies, only "
        "58 have expert image-review labels. Every study has a radiology report, but there are none at test time, and the "
        "reports agree with the experts only about 82% of the time, because they over-call mild disease. So I built a label "
        "engine. Two open language models read each report against a rubric of the experts' thresholds. I calibrate on the "
        "58 real labels, let the image model clean the labels once, and stop there, because a second round made things "
        "worse. Better labels alone were worth about plus 0.014 AUC."), 1)
    for i, (num, lab, col) in enumerate([("4,407", "training studies", BLUE), ("58", "with expert image-review labels", VERM),
                                         ("0", "reports available at test time", INK)]):
        y = 1.75 + i * 1.65
        rect(s, 0.55, y, 4.6, 1.45, LIGHT)
        rect(s, 0.55, y, 0.08, 1.45, col)
        tb(s, 0.85, y + 0.12, 4.2, 0.8, [num], size=44, color=col, bold=True)
        tb(s, 0.85, y + 0.9, 4.2, 0.45, [lab], size=16, color=GREY)
    tb(s, 5.6, 1.7, 7.2, 0.4, ["Label engine (done)"], size=20, color=BLUE, bold=True)
    tb(s, 5.6, 2.2, 7.2, 4.6, [
        ("Two open LLMs (Qwen2.5-72B, Llama-3.3-70B) read each report against a rubric of the experts' thresholds", dict(bullet=True)),
        ("Calibrated on the 58 real labels with leave-one-out, so the labeler is never scored on its own tuning data", dict(bullet=True)),
        ("One noisy-student round: blend with the image model's own out-of-fold predictions → v3 labels", dict(bullet=True)),
        ("A second round regressed, so I stopped", dict(bullet=True)),
        ("Reports and experts agree only ~82%: reports over-call mild disease", dict(bullet=True, color=GREY)),
        ("Better labels alone: +0.014 AUC at a fixed architecture", dict(bullet=True, bold=True)),
    ], size=18)

    # ------------------------------------------------------------------ 2
    s = base(prs, "So far: a first model reached 0.893, then every tweak stalled", "The first model family", (
        "My first image model squeezes each series into one token, then mixes six slots. Its public score was 0.893. One "
        "correction: that notebook picks its configuration from a time budget, and on Kaggle's GPUs the three-model ensemble "
        "would need about 21 hours, so it almost certainly ran only its smallest model, one ViT-S. I can't confirm that. "
        "Then I tried every lever I had: resolution, more slices, a better head, a report-text loss, bigger ensembles, each "
        "against a bar I set in advance. Nothing cleared it. So I concluded the labels were the ceiling. That was wrong."), 2)
    pic(s, "ablation_path_and_nulls_progress.png", 0.3, 1.5, 9.35, 5.5)
    callout(s, 9.75, 1.7, 3.1, 2.45, "What I tried", [
        "Resolution, slice count, head init, a report-text loss, bigger ensembles",
        "Each judged against a bar set in advance",
        "All inside noise"], BLUE)
    callout(s, 9.75, 4.35, 3.1, 2.45, "Caveat I owe you", [
        "The 0.893 was most likely ONE small model (exp005): the 3-model notebook falls back on a time budget",
        "Inferred, not confirmed"], VERM)

    # ------------------------------------------------------------------ 3
    s = base(prs, "A different formulation moved the score to 0.929", "Diagnosis and the change", (
        "Instead of another ablation, I looked outside my own pipeline. The DINOv2 family stalled near 0.89 everywhere, "
        "even in a public twenty-model ensemble, and one very different design didn't. Mine compresses each series to one "
        "vector; the other keeps every window of three adjacent slices and lets each finding attend over all of them. To be "
        "clear about what's mine: that window design follows a public notebook, which I cite. The label engine, the labels, "
        "the folds, the laterality fix, the re-implementation and all the training are mine, with no third-party weights or "
        "labels. It scored 0.929. I changed four things at once, so I can't say which one mattered."), 3)
    pic(s, "formulation_before_after.png", 0.4, 1.55, 8.7, 5.3)
    callout(s, 9.3, 1.7, 3.5, 2.55, "What's mine vs borrowed", [
        "Borrowed: the window-MIL design, from a public Kaggle kernel (cited)",
        "Mine: label engine, labels, folds, laterality fix, re-implementation, all training",
        "No third-party weights or labels"], GREEN)
    callout(s, 9.3, 4.45, 3.5, 2.3, "Honest caveat", [
        "Four things changed together: backbone, windows, wider field of view, laterality fix",
        "No ablation separates them"], VERM)

    # ------------------------------------------------------------------ 4
    s = base(prs, "Where the model looks: each finding pools over the windows it needs", "Looking inside the model", (
        "Because each finding pools over windows, I can see where it looks. I chose these three studies by a rule fixed "
        "before I viewed any map: the most confident gold-positive case per finding. So this is a best case, not a hit rate. "
        "The ACL pools mostly from sagittal fat-suppressed slices, the medial meniscus from coronal, the Baker's cyst from "
        "axial. Across all 58 gold studies, the ACL puts 62% of its attention on sagittal fat-suppressed slices, against 30% "
        "if it were uniform. That shows where the pooling draws from. It's not proof of localization, and no radiologist has "
        "checked these windows."), 4)
    pic(s, "attention_windows_gold58_slide.png", 0.4, 1.5, 7.9, 5.5)
    tb(s, 8.6, 1.7, 4.3, 0.4, ["How to read it"], size=20, color=BLUE, bold=True)
    tb(s, 8.6, 2.2, 4.3, 4.6, [
        ("Four most-attended windows per finding, plus attention over all windows", dict(bullet=True)),
        ("ACL → sagittal, medial meniscus → coronal, Baker's cyst → axial", dict(bullet=True)),
        ("Across all 58 gold studies: ACL puts 62% of its attention on sagittal FS slices (30% if uniform)", dict(bullet=True, bold=True)),
        ("Studies picked by a fixed rule before any map was viewed: a best case, not a hit rate", dict(bullet=True, color=VERM)),
        ("Shows where pooling draws from, not proof of localisation; no radiologist has checked", dict(bullet=True, color=GREY)),
    ], size=16)

    # ------------------------------------------------------------------ 5
    s = base(prs, "Where things stand, and what is left", "Status", (
        "Where things stand: my current best submissions are ten checkpoints at public 0.929, and a two-fold version at 0.925 "
        "in about twenty minutes. The leaderboard beat my validation score every time, because I validate against noisy "
        "report labels. Three lessons. One: six null results showed my model family was saturated, not that the problem was. "
        "Two: measure runtime on the real hardware. I projected fifteen minutes and measured fifty-three. Three: a runtime "
        "fallback can silently change which model you're scoring, which is why my 0.893 is probably one small model. Log the "
        "choice. What's left: the private leaderboard on October 22, which I don't know yet, a demo in late November, "
        "releasing code and weights after the close, and the final report on December 5."), 5)
    rows = [("", "OOF", "Gold-58", "Public LB", "Runtime (2×T4)"),
            ("Slot pipeline (most likely exp005 alone)", "0.8631", "0.8903", "0.893", "3 models ~21 h"),
            ("10 checkpoints (current best, accuracy)", "0.8840", "0.9111", "0.929", "~100 min"),
            ("2 folds (current best, efficiency)", "n/a", "0.9078", "0.925", "~20.5 min")]
    gt = s.shapes.add_table(len(rows), 5, Inches(0.55), Inches(1.65), Inches(12.2), Inches(1.9))
    tbl = gt.table
    for j, wd in enumerate([4.9, 1.4, 1.5, 1.7, 2.7]):
        tbl.columns[j].width = Inches(wd)
    for i, r in enumerate(rows):
        for j, val in enumerate(r):
            c = tbl.cell(i, j)
            c.text = val
            c.fill.solid()
            c.fill.fore_color.rgb = BLUE if i == 0 else (LIGHT if i % 2 == 0 else WHITE)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                for run in p.runs:
                    run.font.name, run.font.size = FONT, Pt(16)
                    run.font.bold = (i == 0) or (i > 1 and j == 3)
                    run.font.color.rgb = WHITE if i == 0 else INK
    tb(s, 0.55, 3.65, 12.2, 0.4, ["Gold-58 is n = 58 (95% interval about ±0.03). OOF is scored against noisy report labels; the leaderboard beat it every time."],
       size=12, color=GREY)
    callout(s, 0.55, 4.2, 6.5, 2.65, "Three lessons so far", [
        "Null ablations show a family is saturated, not where the ceiling is",
        "Measure runtime on the target hardware: I projected 15 min, measured 53",
        "A runtime fallback can silently change which model is scored: log it"], BLUE, size=15)
    callout(s, 7.3, 4.2, 5.45, 2.65, "What is left", [
        "Oct 22: private leaderboard (rank not yet known)",
        "Nov 23–27: demo",
        "After the close: release code + weights",
        "Dec 5: final report"], VERM, size=17)

    # ------------------------------------------------------------------ backup
    s = base(prs, "Backup: per-label ROC on the 58 image-review studies", "Backup (for questions)", (
        "Backup slide, not presented. Gold-58, exp021: strongest are medial meniscus and Baker's at 0.986, ACL 0.979, effusion "
        "0.978; weakest are PF OA 0.824, lateral OA 0.828 and lateral meniscus 0.834. With n = 58 the intervals are wide. ACL is "
        "the clearest case of noisy proxy labels: out-of-fold 0.857 but 0.979 on the 58."), 6)
    pic(s, "roc_gold58_exp018_exp021.png", 0.4, 1.5, 7.6, 5.5)
    tb(s, 8.4, 1.7, 4.4, 4.8, [
        ("Strongest: medial meniscus, Baker's 0.986; ACL 0.979; effusion 0.978", dict(bullet=True)),
        ("Weakest: PF OA 0.824, lateral OA 0.828, lateral meniscus 0.834", dict(bullet=True)),
        ("n = 58: wide intervals, shown on the figure", dict(bullet=True, color=GREY)),
        ("ACL: out-of-fold 0.857 but gold-58 0.979 → noisy proxy labels", dict(bullet=True, bold=True)),
    ], size=16)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print("wrote", OUT, "-", len(prs.slides), "slides")


if __name__ == "__main__":
    main()
