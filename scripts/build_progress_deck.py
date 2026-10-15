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

"""Build the course progress presentation: exactly the five slides the instructor specified (max 5 minutes), 16:9,
with speaker notes. Slide order and required content:

  1 Introduction   project title, expected output / application, a figure of the system
  2 Dataset        name + URL, size (samples AND storage), input format
  3 Baseline model GitHub / code link, pretrained-model URL, baseline performance
  4 Method         existing implementation vs my modifications, evaluation metrics, preliminary results
  5 Plan           next steps and planned improvements

Every fact on the slides is sourced in docs/CV_PROGRESS_PRESENTATION.md. Nothing claims an experiment is running or a
result is pending when it is not.

    python scripts/build_progress_deck.py        # -> docs/presentation/progress_presentation.pptx
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs/figures"
OUT = ROOT / "docs/presentation/progress_presentation.pptx"

INK, GREY, LIGHT = RGBColor(0x11, 0x18, 0x27), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xF3, 0xF4, 0xF6)
BLUE, VERM, GREEN = RGBColor(0x00, 0x72, 0xB2), RGBColor(0xD5, 0x5E, 0x00), RGBColor(0x00, 0x9E, 0x73)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"
W, H = 13.333, 7.5

URL_COMP = "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection"
URL_INFER = "https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model"
URL_TRAIN = "https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model"
URL_WEIGHTS = "https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense"
URL_TIMM_HF = "https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
URL_TIMM_GH = "https://github.com/huggingface/pytorch-image-models"
URL_MINE = "https://github.com/HRDevelops/rsna-knee"


def tb(slide, x, y, w, h, paras, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.05):
    """Text box. paras: list of str or (str, opts) with opts: size, color, bold, bullet, after, link, font."""
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
        para.space_after = Pt(o.get("after", 5))
        run = para.add_run()
        run.text = ("•  " + text) if o.get("bullet") else text
        f = run.font
        f.name, f.size = o.get("font", FONT), Pt(o.get("size", size))
        f.bold, f.color.rgb = o.get("bold", bold), o.get("color", color)
        if o.get("link"):
            run.hyperlink.address = o["link"]
    return box


def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def card(slide, x, y, w, h, head, paras, color, size=13):
    rect(slide, x, y, w, h, LIGHT)
    rect(slide, x, y, 0.07, h, color)
    tb(slide, x + 0.2, y + 0.08, w - 0.3, 0.4, [head], size=15, color=color, bold=True)
    tb(slide, x + 0.2, y + 0.5, w - 0.3, h - 0.55, paras, size=size)


def pic(slide, name, x, y, max_w, max_h):
    path = FIG / name
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (max_w - w) / 2), Inches(y + (max_h - h) / 2), Inches(w), Inches(h))


def base(prs, n, kicker, title, notes, title_size=28):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(slide, 0, 0, W, 0.12, BLUE)
    tb(slide, 0.55, 0.26, 12.2, 0.32, [f"{n}  ·  {kicker}".upper()], size=12, color=GREY, bold=True)
    tb(slide, 0.55, 0.55, 12.2, 0.8, [title], size=title_size, bold=True)
    tb(slide, 0.55, 7.1, 9.5, 0.3, ["Progress presentation · RSNA Knee Abnormality Detection 2026 · Husnain Rasool"], size=10, color=GREY)
    tb(slide, 12.2, 7.1, 0.6, 0.3, [f"{n}/5"], size=10, color=GREY, align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def main() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # ======================================================================== 1  Introduction
    s = base(prs, 1, "Introduction", "Weak Supervision and Window-Level Attention for Knee-MRI Abnormality Detection", (
        "Hi, I'm Husnain. My project is on the RSNA knee MRI challenge on Kaggle. Given a knee MRI study, the system outputs twelve probabilities, one per finding: ACL, MCL, meniscus tears, arthritis, effusion and others. It uses images only and is scored by the average ROC-AUC. The application is research decision support, not a medical device. The figure shows the whole system. At the top, language models turn radiology reports into soft training labels, during training only. At the bottom is the image-only model: each study becomes windows of three adjacent slices, a CoAtNet encodes them, and each finding attends over all the windows."), title_size=24)
    tb(s, 0.55, 1.4, 12.2, 0.4, ["RSNA Knee Abnormality Detection (Kaggle 2026)"], size=17, color=GREY)
    card(s, 0.55, 2.0, 4.1, 2.25, "Expected output", [
        ("For each knee-MRI study: 12 probabilities (ACL, MCL, medial / lateral meniscus, 3 OA compartments, effusion, "
         "synovitis, Baker's cyst, contusion, fracture)", dict(size=13)),
        ("From images only; scored by macro ROC-AUC", dict(size=13, bold=True))], BLUE)
    card(s, 0.55, 4.45, 4.1, 2.35, "Application", [
        ("Research decision support for knee-MRI reading; not a medical device", dict(size=13)),
        ("Delivered as a Kaggle code competition: a notebook (GPU ≤ 9 h, no internet) in a main track and an "
         "efficiency track", dict(size=13))], GREEN)
    pic(s, "system_overview.png", 4.85, 1.85, 8.1, 5.1)

    # ======================================================================== 2  Dataset
    s = base(prs, 2, "Dataset", "RSNA Knee Abnormality Detection: 4,407 studies, only 58 with real labels", (
        "The data is the RSNA Knee Abnormality Detection competition on Kaggle: 4,407 training studies, 24,371 series, about 819 thousand DICOM slices, roughly 531 gigabytes raw. The test set is hidden, about 1,300 studies. The key problem: only 58 studies have expert labels. Every study has a free-text report, but reports don't exist at test time. I preprocess each study into a 64 by 336 by 336 tensor, 32 gigabytes in total, then into up to 54 windows of three slices at 384 pixels. That is what the model sees."))
    card(s, 0.55, 1.45, 7.9, 1.4, "Name & URL", [
        ("RSNA Knee Abnormality Detection (Kaggle competition, 2026)", dict(size=14, bold=True)),
        (URL_COMP, dict(size=13, color=BLUE, link=URL_COMP))], BLUE)
    card(s, 0.55, 2.98, 7.9, 1.65, "Size", [
        ("4,407 training studies (samples) · 24,371 series · 819,078 DICOM slices", dict(size=14, bold=True)),
        ("Storage: ≈ 531 GB raw; my preprocessed cache 31.8 GB (4,407 × 7.2 MB)", dict(size=13)),
        ("Test set: ~1,300 hidden studies (images only)", dict(size=13, color=GREY))], VERM)
    card(s, 0.55, 4.76, 7.9, 2.25, "Input format", [
        ("Raw: one DICOM file per slice; a study = 3–14 series; 11–320 slices per series (median 30); mostly 512×512 (221 sizes)",
         dict(size=12)),
        ("Cache per study: (64, 336, 336) uint8", dict(size=13, bold=True, font="Consolas")),
        ("Model input per study: (K ≤ 54, 3, 384, 384)", dict(size=13, bold=True, font="Consolas")),
        ("Target: 12 values per study", dict(size=13, bold=True, font="Consolas"))], GREEN)
    for i, (num, lab, col) in enumerate([("4,407", "studies, each with a free-text report", BLUE),
                                         ("58", "with expert image-review labels", VERM),
                                         ("0", "reports available at test time", INK)]):
        y = 1.45 + i * 1.86
        rect(s, 8.75, y, 4.05, 1.7, LIGHT)
        rect(s, 8.75, y, 0.08, 1.7, col)
        tb(s, 9.05, y + 0.1, 3.6, 0.85, [num], size=44, color=col, bold=True)
        tb(s, 9.05, y + 1.0, 3.6, 0.55, [lab], size=15, color=GREY)

    # ======================================================================== 3  Baseline model
    s = base(prs, 3, "Baseline model", "Baseline: a public CoAtNet-2 window-attention notebook (0.924 public)", (
        "The baseline is a public Kaggle notebook by Dread Development: a single CoAtNet-2 with attention over windows of three adjacent slices. There is no GitHub repository; it is a Kaggle notebook with a separate training notebook, and its weights are a Kaggle dataset. All the links are here. I did not use those weights; my backbone starts from public timm ImageNet weights, not knee-trained. The baseline reports 0.924 on the public leaderboard, which I have not re-run. Its 0.9167 on the 58 gold studies is optimistic, because its training keeps the best epochs by AUC on those same 58. My own first model scored 0.893. My code is on GitHub, private until October 22."))
    card(s, 0.55, 1.5, 8.0, 2.9, "Code / GitHub repository", [
        ("Baseline: no GitHub repo; two public Kaggle notebooks (Dread Development)", dict(size=13, bold=True)),
        ("Inference: " + URL_INFER, dict(size=12, color=BLUE, link=URL_INFER)),
        ("Training: " + URL_TRAIN, dict(size=12, color=BLUE, link=URL_TRAIN)),
        ("Backbone library (timm): " + URL_TIMM_GH, dict(size=12, color=BLUE, link=URL_TIMM_GH)),
        ("My repository: " + URL_MINE + "  (private until the Oct 22 close)", dict(size=12, color=GREY))], BLUE)
    card(s, 0.55, 4.55, 8.0, 2.4, "Pretrained model URL", [
        ("Baseline's trained weights (Kaggle dataset; I did not use them):", dict(size=13, bold=True)),
        (URL_WEIGHTS, dict(size=12, color=BLUE, link=URL_WEIGHTS)),
        ("My backbone init: timm CoAtNet-2, ImageNet-12k → 1k (not trained on knees):", dict(size=13, bold=True)),
        (URL_TIMM_HF, dict(size=12, color=BLUE, link=URL_TIMM_HF))], GREEN)
    card(s, 8.8, 1.5, 4.0, 5.45, "Baseline performance", [
        ("Public baseline (author-reported, not re-run by me)", dict(size=13, bold=True)),
        ("Public leaderboard: 0.924", dict(size=22, bold=True, color=VERM, after=4)),
        ("Gold-58: 0.9167, optimistic: epochs were chosen on those 58 studies", dict(size=12, color=GREY, after=14)),
        ("My first model (DINOv2 slots)", dict(size=13, bold=True)),
        ("Public leaderboard: 0.893", dict(size=22, bold=True, color=BLUE, after=4)),
        ("Metric: macro ROC-AUC over the 12 findings", dict(size=12, color=GREY))], VERM)

    # ======================================================================== 4  Method
    s = base(prs, 4, "Method", "Existing implementation, my modifications, and preliminary results", (
        "The existing implementation fills five slots with 64 slices, crops 140 millimetres, builds three-slice windows, and applies per-finding attention; its labels also come from a language model reading reports. My modifications: my own label engine, with expert thresholds, two language models, and calibration on the 58; my own folds, grouped by report cluster, never choosing epochs on the gold set; windows that never cross slots, with explicit laterality handling; and a ten-checkpoint ensemble plus a fast two-fold version. I did not audit whether the baseline handles laterality. Evaluation is macro AUC, out-of-fold, on the 58 gold studies and on the public leaderboard, plus attention maps. Preliminary: 0.929 versus 0.924, a small difference I am not claiming as an improvement. Private results come October 22."))
    card(s, 0.55, 1.45, 4.0, 3.45, "Existing implementation", [
        ("5 slots, 64 slices; 140 mm crop → 336 px", dict(bullet=True)),
        ("Windows of 3 adjacent slices as RGB → CoAtNet-2 (384 px)", dict(bullet=True)),
        ("Per-finding attention over all windows", dict(bullet=True)),
        ("Labels: an LLM reads each report → soft labels (4,349 studies)", dict(bullet=True)),
        ("Epochs chosen on the 58 gold studies", dict(bullet=True))], GREY, size=13)
    card(s, 4.72, 1.45, 4.3, 3.45, "My modifications", [
        ("Own label engine: expert-threshold rubric, 2 LLMs, leave-one-out calibration, noisy-student v3", dict(bullet=True)),
        ("Own folds grouped by report cluster; epochs chosen without the gold set", dict(bullet=True)),
        ("Windows never cross slot boundaries; laterality normalisation (side table → DICOM tag → geometry)",
         dict(bullet=True)),
        ("10-checkpoint ensemble; 2-fold fast variant", dict(bullet=True)),
        ("Pre-registered gates; runtime measured on Kaggle T4s", dict(bullet=True))], GREEN, size=13)
    card(s, 9.19, 1.45, 3.6, 3.45, "Evaluation metrics", [
        ("Quantitative", dict(bold=True, size=12, color=BLUE, after=2)),
        ("Macro ROC-AUC (mean of 12 per-finding AUCs): out-of-fold (4,349 studies), gold-58, public leaderboard",
         dict(bullet=True)),
        ("Efficiency score + runtime on 2×T4", dict(bullet=True, after=8)),
        ("Qualitative", dict(bold=True, size=12, color=BLUE, after=2)),
        ("Attention maps: top windows per finding; per-finding ROC curves", dict(bullet=True))], BLUE, size=13)
    rows = [("Preliminary results", "OOF AUC", "Gold-58 AUC", "Public LB", "Runtime (2×T4)"),
            ("Public baseline (author-reported)", "n/a", "0.9167 (optimistic)", "0.924", "n/a"),
            ("Mine: 5 folds", "0.8816", "0.9091", "0.929", "~53 min"),
            ("Mine: 10 checkpoints (best)", "0.8840", "0.9111", "0.929", "~100 min"),
            ("Mine: 2 folds (fast)", "n/a", "0.9078", "0.925", "~20.5 min")]
    gt = s.shapes.add_table(len(rows), 5, Inches(0.55), Inches(5.05), Inches(12.25), Inches(1.7))
    tbl = gt.table
    for j, wd in enumerate([4.2, 1.7, 2.45, 1.7, 2.2]):
        tbl.columns[j].width = Inches(wd)
    for i, r in enumerate(rows):
        tbl.rows[i].height = Inches(0.34)
        for j, val in enumerate(r):
            c = tbl.cell(i, j)
            c.text = val
            c.fill.solid()
            c.fill.fore_color.rgb = BLUE if i == 0 else (LIGHT if i % 2 == 0 else WHITE)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_top = c.margin_bottom = Inches(0.02)
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                for run in p.runs:
                    run.font.name, run.font.size = FONT, Pt(13)
                    run.font.bold = (i == 0) or (i == 3)
                    run.font.color.rgb = WHITE if i == 0 else INK
    tb(s, 0.55, 6.78, 12.25, 0.3, ["Public-leaderboard differences of ~0.005 are not claimed as an improvement. Gold-58 is n = 58 "
                                   "(95% interval about ±0.03). Private leaderboard: Oct 22."], size=11, color=GREY)

    # ======================================================================== 5  Plan
    s = base(prs, 5, "Plan", "Next steps and planned improvements", (
        "Next steps. October 22: the competition closes and the private leaderboard shows whether these scores hold. Then I release code and weights, give the demo in late November, and submit the report on December 5. Planned improvements are analyses: ablate the four things I changed together to find which one mattered; ask a radiologist to review the attention windows, which are only a best-case illustration now; and trace two studies where deployed predictions differ from training, probably a laterality rule. Thank you."))
    tb(s, 0.55, 1.5, 6.0, 0.4, ["Next steps"], size=20, color=BLUE, bold=True)
    steps = [("Oct 22", "Competition closes: private leaderboard shows whether 0.929 holds", VERM),
             ("After the close", "Release code, weights and a write-up", GREEN),
             ("Nov 23–27", "Demo", BLUE),
             ("Dec 5", "Final report", INK)]
    for i, (when, what, col) in enumerate(steps):
        y = 2.05 + i * 1.18
        rect(s, 0.55, y, 1.85, 0.95, col)
        tb(s, 0.55, y, 1.85, 0.95, [when], size=17, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        rect(s, 2.4, y, 4.2, 0.95, LIGHT)
        tb(s, 2.55, y, 3.95, 0.95, [what], size=15, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, 7.0, 1.5, 5.8, 0.4, ["Planned improvements"], size=20, color=BLUE, bold=True)
    imps = [("Isolate what mattered", "Ablate the four changes I made together (backbone, windows, field of view, laterality)"),
            ("Clinical check of the attention", "Ask a radiologist to review the attention windows; today they are a best-case illustration"),
            ("Trace a prediction mismatch", "2 of 58 studies differ between training and deployment (suspected laterality rule); confirm the cause")]
    for i, (head, body) in enumerate(imps):
        y = 2.05 + i * 1.6
        rect(s, 7.0, y, 5.8, 1.42, LIGHT)
        rect(s, 7.0, y, 0.07, 1.42, BLUE)
        tb(s, 7.2, y + 0.08, 5.5, 0.4, [f"{i + 1}.  {head}"], size=16, color=BLUE, bold=True)
        tb(s, 7.2, y + 0.55, 5.5, 0.85, [body], size=14)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print("wrote", OUT, "-", len(prs.slides), "slides")


if __name__ == "__main__":
    main()
