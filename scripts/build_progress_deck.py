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

"""Course progress presentation (max 5 minutes), 16:9, speaker notes in the file.

The instructor's structure, in order: 1 Introduction, 2 Dataset, 3 Baseline model, 4 Method, 5 Plan. One extra slide,
"Everything I trained", sits between Baseline and Method so the experiments get room. The baseline is MY first model (the
DINOv2 slot model), not the public notebook; the public notebook appears on the Method slide as the design I started from.

Design: plain white slides, one idea each, real MRI and real charts, little text. Facts are sourced in
docs/CV_PROGRESS_PRESENTATION.md.

    python scripts/fig_deck_assets.py && python scripts/build_progress_deck.py
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
OUT = ROOT / "docs/presentation/progress_presentation_v2.pptx"

INK, GREY, RULE = RGBColor(0x1F, 0x29, 0x37), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xD1, 0xD5, 0xDB)
BLUE, VERM = RGBColor(0x00, 0x72, 0xB2), RGBColor(0xD5, 0x5E, 0x00)
BODY, TITLE, MONO = "Calibri", "Georgia", "Consolas"
W, H = 13.333, 7.5

URL_COMP = "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection"
URL_REPO = "https://github.com/HRDevelops/rsna-knee"
URL_DINO_B = "https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m"
URL_DINO_S = "https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m"
URL_PUBLIC_NB = "https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model"


def text(slide, x, y, w, h, paras, size=18, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.05):
    """paras: str or (str, opts) with opts: size, color, bold, font, after, link, italic."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, p in enumerate(paras):
        t, o = (p, {}) if isinstance(p, str) else p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment, para.line_spacing, para.space_after = align, spacing, Pt(o.get("after", 4))
        r = para.add_run()
        r.text = t
        f = r.font
        f.name, f.size = o.get("font", font), Pt(o.get("size", size))
        f.bold, f.italic, f.color.rgb = o.get("bold", bold), o.get("italic", False), o.get("color", color)
        if o.get("link"):
            r.hyperlink.address = o["link"]
    return box


def hline(slide, x, y, w):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.012))
    s.fill.solid()
    s.fill.fore_color.rgb = RULE
    s.line.fill.background()
    s.shadow.inherit = False


def pic(slide, name, x, y, max_w, max_h, center=False):
    path = FIG / name
    iw, ih = Image.open(path).size
    k = min(max_w / iw, max_h / ih)
    w, h = iw * k, ih * k
    ox = (max_w - w) / 2 if center else 0
    slide.shapes.add_picture(str(path), Inches(x + ox), Inches(y), Inches(w), Inches(h))
    return w, h


def new_slide(prs, n, title, notes, size=30):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, 0.6, 0.38, 12.1, 1.0, [title], size=size, font=TITLE, anchor=MSO_ANCHOR.TOP, spacing=0.95)
    text(s, 12.0, 7.05, 0.8, 0.3, [f"{n}"], size=11, color=GREY, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def main() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # ================================================================== 1  Introduction
    s = new_slide(prs, 1, "Weak Supervision and Window-Level Attention for Knee-MRI Abnormality Detection", (
        "I'm working on the RSNA knee MRI challenge on Kaggle. A knee study is several image series in three planes. The system reads one study and outputs twelve probabilities, one per finding, like ACL tear, meniscus tears and effusion, from images only, scored by the average ROC-AUC. The use is research decision support, not a medical device. In the diagram, the training labels come from radiology reports, but only during training; at test time only the images are used."), size=27)
    text(s, 0.6, 1.45, 12.1, 0.4, ["RSNA Knee Abnormality Detection (Kaggle 2026)  \u00b7  Husnain Rasool"], size=16, color=GREY)
    pic(s, "deck_mri_strip.png", 0.6, 1.95, 4.6, 1.85)
    text(s, 0.6, 3.72, 4.6, 0.3, ["One knee, three planes"], size=12, color=GREY)
    text(s, 5.7, 2.0, 3.9, 0.3, ["Output"], size=13, color=GREY)
    text(s, 5.7, 2.3, 3.9, 1.6, ["12 probabilities per study, from images only: ACL, MCL, two menisci, three arthritis "
                                 "compartments, effusion, synovitis, Baker's cyst, contusion, fracture"], size=15)
    text(s, 9.9, 2.0, 2.9, 0.3, ["Use"], size=13, color=GREY)
    text(s, 9.9, 2.3, 2.9, 1.6, ["Research decision support for reading knee MRI; not a medical device"], size=15)
    pic(s, "system_overview_wide.png", 0.6, 4.15, 12.1, 2.85)

    # ================================================================== 2  Dataset
    s = new_slide(prs, 2, "Only 58 of the 4,407 studies have real labels", (
        "The data is the Kaggle competition set: 4,407 training studies, about 819 thousand DICOM slices, roughly 531 gigabytes. The test set is hidden, about 1,300 studies. The problem is the bar, drawn to scale: only 58 studies have expert labels. The rest only have a free-text report, and at test time there are no reports. I turn each study into a 64 by 336 by 336 tensor, then up to 54 windows of three slices."))
    rows = [("Dataset", [("RSNA Knee Abnormality Detection, Kaggle 2026", dict(size=20, bold=True)),
                         (URL_COMP, dict(size=14, color=BLUE, link=URL_COMP))]),
            ("Size", [("4,407 studies (samples)  ·  24,371 series  ·  819,078 DICOM slices", dict(size=20, bold=True)),
                      ("≈ 531 GB raw  ·  31.8 GB after my preprocessing  ·  hidden test set ≈ 1,300 studies",
                       dict(size=15))]),
            ("Input", [("raw: one DICOM file per slice (11–320 per series, mostly 512×512)", dict(size=14, font=MONO)),
                       ("per study: (64, 336, 336) uint8  →  windows (K ≤ 54, 3, 384, 384)  →  12 labels",
                        dict(size=14, font=MONO, bold=True))])]
    for i, (lab, paras) in enumerate(rows):
        y = 1.5 + i * 1.18
        text(s, 0.6, y + 0.04, 1.6, 0.4, [lab], size=14, color=GREY)
        text(s, 2.3, y, 10.4, 1.1, paras)
        hline(s, 0.6, y + 1.08, 12.15)
    pic(s, "deck_data_scale.png", 0.6, 5.3, 12.15, 1.6)

    # ================================================================== 3  Baseline
    s = new_slide(prs, 3, "Baseline: the first model I built, a DINOv2 slot model", (
        "My baseline is the first model I built, a DINOv2 slot model. Each series goes through a vision transformer and is squashed into one token; a small transformer mixes six slots and twelve queries read the result. It starts from public DINOv2 weights; the code is in my GitHub repository, private until the competition closes. The best version scored 0.868 out-of-fold. My first submission scored 0.893 publicly, though I believe it was the small model alone."))
    pic(s, "deck_lane_before.png", 0.6, 1.55, 9.3, 2.4)
    text(s, 10.15, 1.6, 2.7, 0.4, ["Public leaderboard"], size=14, color=GREY)
    text(s, 10.15, 1.95, 2.7, 0.9, ["0.893"], size=50, color=VERM, bold=True)
    text(s, 10.15, 3.0, 2.7, 0.8, ["my first submission (most likely the small model alone)"], size=13, color=GREY)
    hline(s, 0.6, 4.3, 12.15)
    cols = [("Code", [("github.com/HRDevelops/rsna-knee", dict(size=16, color=BLUE, link=URL_REPO)),
                      ("private until the competition closes (Oct 22)", dict(size=13, color=GREY))]),
            ("Pretrained model", [("DINOv2 ViT-B/14 (Hugging Face)", dict(size=16, color=BLUE, link=URL_DINO_B)),
                                  ("DINOv2 ViT-S/14 (Hugging Face)", dict(size=16, color=BLUE, link=URL_DINO_S)),
                                  ("trained on generic images, not on knees", dict(size=13, color=GREY))]),
            ("Performance", [("Out-of-fold macro-AUC: 0.863 (ViT-S), 0.868 (ViT-B)", dict(size=16)),
                             ("Public leaderboard: 0.893", dict(size=16, bold=True)),
                             ("Several versions: see the next slide", dict(size=13, color=GREY))])]
    for i, (head, paras) in enumerate(cols):
        x = 0.6 + i * 4.1
        text(s, x, 4.5, 3.9, 0.35, [head], size=14, color=GREY)
        text(s, x, 4.9, 3.9, 1.9, paras)

    # ================================================================== 4  Everything I trained
    s = new_slide(prs, 4, "Everything I trained, on the same five folds", (
        "Every model I trained, on the same folds. Blue is the DINOv2 slot family. The biggest steps were better labels, 0.849 to 0.863, and a bigger backbone, about 0.005. A newer backbone was worse, a 3D network was too slow, and the tweaks on the right gave nothing beyond noise. Orange is the window-attention family, 0.882 and 0.884. The public leaderboard moved more than this proxy: 0.893 to 0.929."))
    pic(s, "deck_models_ladder.png", 0.45, 1.35, 8.3, 5.4)
    text(s, 9.1, 1.5, 3.7, 0.4, ["Tried on top, no gain"], size=18, bold=True)
    tried = ["448 px instead of 384", "32 slices instead of 24", "ACL-aware head initialisation", "a report-text auxiliary loss",
             "mirror-image averaging at test", "EMA weights; denser windows", "CoAtNet-3: no pretrained weights that size",
             "3D ResNet-50: too slow, stopped at 40%"]
    text(s, 9.1, 2.0, 3.7, 3.3, [(t, dict(size=15, after=6)) for t in tried])
    text(s, 9.1, 5.3, 3.7, 1.5, [("What did help", dict(size=18, bold=True, color=BLUE)),
                                 ("better labels +0.014", dict(size=15)), ("bigger backbone +0.005", dict(size=15)),
                                 ("CoAtNet over DINOv2 in windows +0.011", dict(size=15))])

    # ================================================================== 5  Method
    s = new_slide(prs, 5, "Method: window attention, with my own labels and folds", (
        "The public notebook I started from keeps every window of three adjacent slices and lets each finding attend over them. I rebuilt it and changed four things. Labels: Qwen and Llama read each report; a small calibrator per finding is fit on the 58 expert labels, always scoring a study with a calibrator that never saw it; then one noisy-student round. Folds are grouped by report cluster, and epochs are never chosen on the gold set. Windows never cross slots, and laterality is normalized. Result: 0.929 against 0.924 for the public notebook, a small difference."))
    pic(s, "deck_lane_after.png", 0.6, 1.45, 8.1, 2.1)
    text(s, 0.6, 3.4, 7.7, 0.35, ["The design I started from: a public Kaggle notebook (author-reported 0.924)"], size=12, color=GREY)
    text(s, 0.6, 3.95, 7.7, 0.4, ["What I changed"], size=18, bold=True, color=BLUE)
    changes = ["Labels: Qwen2.5-72B and Llama-3.3-70B read the reports; calibrated per finding, scored leave-one-out on the 58; "
               "one noisy-student round",
               "Folds grouped by report cluster; epochs never chosen on the 58 gold studies",
               "Windows never cross slots; laterality normalised (right knees mirrored)",
               "Ten checkpoints (2 seeds × 5 folds); a 2-fold model that runs in 20 minutes"]
    text(s, 0.6, 4.4, 7.7, 2.6, [(f"{i + 1}.  {t}", dict(size=15, after=7)) for i, t in enumerate(changes)])
    text(s, 8.9, 1.5, 3.9, 0.4, ["Scored by"], size=14, color=GREY)
    text(s, 8.9, 1.85, 3.9, 1.5, [("Macro ROC-AUC: the mean of the 12 per-finding AUCs", dict(size=15)),
                                  ("out-of-fold  ·  58 gold studies  ·  public leaderboard", dict(size=13, color=GREY)),
                                  ("plus attention maps, to look inside", dict(size=13, color=GREY))])
    hline(s, 8.9, 3.6, 3.9)
    text(s, 8.9, 3.75, 3.9, 0.35, ["Public leaderboard"], size=14, color=GREY)
    for i, (lab, val, col) in enumerate([("my baseline (slots)", "0.893", INK), ("public notebook*", "0.924", INK),
                                         ("mine, 10 checkpoints", "0.929", VERM), ("mine, 2 folds (fast)", "0.925", INK)]):
        y = 4.2 + i * 0.55
        text(s, 8.9, y, 2.6, 0.45, [lab], size=16)
        text(s, 11.4, y - 0.04, 1.4, 0.5, [val], size=22, color=col, bold=True, align=PP_ALIGN.RIGHT)
    text(s, 8.9, 6.45, 3.9, 0.55, ["* author-reported, not re-run by me. A 0.005 gap is not claimed as an improvement. "
                                   "Private score: Oct 22."], size=11, color=GREY)

    # ================================================================== 6  Plan
    s = new_slide(prs, 6, "Next: finish the competition, then answer three open questions", (
        "Next steps: on October 22 the private leaderboard shows whether 0.929 holds; then the demo in late November and the report on December 5. I also want to answer three questions with experiments. What caused the jump? The backbone is measured; I will undo the field of view and the laterality fix, five folds each, about 17 GPU-hours. Is the attention real? Delete the top-attended windows and compare with random ones. Does it survive missing data? Drop one plane at test time."))
    text(s, 0.6, 1.4, 12.1, 0.45, [("Oct 22  private leaderboard     →     Nov 23–27  demo     →     Dec 5  final report",
                                    dict(size=18, bold=True, color=BLUE))])
    for x, head in ((0.6, "Question"), (4.7, "Test"), (9.9, "Cost")):
        text(s, x, 2.05, 3.0, 0.3, [head], size=13, color=GREY)
    hline(s, 0.6, 2.4, 12.15)
    plan = [("What caused the big jump in score?",
             "Undo one change at a time (115 mm field of view; no laterality fix), all 5 folds. The backbone is already "
             "measured: DINOv2-B inside windows scores 0.871, CoAtNet-2 0.882.",
             "≈ 17 GPU-hours\n(2 variants × 5 folds × ~100 min)"),
            ("Is the attention doing real work?",
             "Delete the top-attended windows and measure the AUC drop; compare with deleting the same number of random windows.",
             "inference only,\nunder 1 GPU-hour"),
            ("Does it survive missing data?",
             "Remove one plane (sagittal, coronal or axial) from each study and measure the AUC loss; some real studies lack a plane.",
             "inference only,\n~1 GPU-hour")]
    for i, (q, t, c) in enumerate(plan):
        y = 2.55 + i * 1.35
        text(s, 0.6, y, 3.9, 1.2, [q], size=19, bold=True, font=TITLE)
        text(s, 4.7, y, 4.9, 1.25, [t], size=15)
        text(s, 9.9, y, 2.9, 1.2, [c], size=15, color=BLUE)
        hline(s, 0.6, y + 1.25, 12.15)
    text(s, 0.6, 6.65, 12.1, 0.4, ["Also: trace the 2 of 58 studies whose deployed predictions differ from training (suspected laterality "
                                   "rule); after the close, release the code and weights."], size=13, color=GREY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print("wrote", OUT, "-", len(prs.slides), "slides")


if __name__ == "__main__":
    main()
