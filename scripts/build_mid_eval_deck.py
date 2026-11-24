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

"""Build the Course Mid-Evaluation Presentation Deck.

Strict adherence to instructor's 5-slide structure and <= 5-minute duration limit:
  Slide 1: Introduction (Project title, expected output / application, system figure)
  Slide 2: Dataset (Name & direct URL, size in samples & storage, input format)
  Slide 3: Baseline Model (GitHub link, pretrained model URL, baseline performance)
  Slide 4: Method (Existing implementation vs our modifications, metrics, preliminary results)
  Slide 5: Plan (Next steps & planned improvements grounded in empirical ablations)

Speaker notes included on every slide, calibrated for a 4.5-minute delivery (~635 words total).
Outputs:
  docs/presentation/mid_evaluation_presentation.pptx
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
OUT = ROOT / "docs/presentation/mid_evaluation_presentation.pptx"

# Cohesive modern palette
NAVY = RGBColor(0x0F, 0x17, 0x2A)       # Slate 900
SLATE = RGBColor(0x33, 0x41, 0x55)      # Slate 700
MUTED = RGBColor(0x64, 0x74, 0x8B)      # Slate 500
CARD_BG = RGBColor(0xF8, 0xFA, 0xFC)    # Slate 50
CARD_BORDER = RGBColor(0xE2, 0xE8, 0xF0)# Slate 200
BLUE = RGBColor(0x02, 0x84, 0xC7)       # Sky 600
DARK_BLUE = RGBColor(0x03, 0x69, 0xA1)  # Sky 700
EMERALD = RGBColor(0x05, 0x96, 0x69)    # Emerald 600
AMBER = RGBColor(0xD9, 0x77, 0x06)      # Amber 600
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"
FONT_CODE = "Consolas"
W, H = 13.333, 7.5

URL_COMP = "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection"
URL_INFER = "https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model"
URL_TRAIN = "https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model"
URL_WEIGHTS = "https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense"
URL_TIMM_HF = "https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
URL_TIMM_GH = "https://github.com/huggingface/pytorch-image-models"
URL_MINE = "https://github.com/HRDevelops/rsna-knee"


def tb(slide, x, y, w, h, paras, size=14, color=NAVY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.05, font=FONT_BODY):
    """Add a wrapped text box with styled paragraphs and runs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    first = True
    for p in paras:
        text, o = (p, {}) if isinstance(p, str) else p
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = align
        para.line_spacing = spacing
        para.space_after = Pt(o.get("after", 4))
        run = para.add_run()
        prefix = "•  " if o.get("bullet") else ""
        run.text = prefix + text
        f = run.font
        f.name = o.get("font", font)
        f.size = Pt(o.get("size", size))
        f.bold = o.get("bold", bold)
        f.color.rgb = o.get("color", color)
        if o.get("link"):
            run.hyperlink.address = o["link"]
    return box


def rect(slide, x, y, w, h, fill, line=None, width=1.0):
    """Draw a rectangle."""
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(width)
    s.shadow.inherit = False
    return s


def card(slide, x, y, w, h, head, paras, accent_color=BLUE, size=12):
    """Structured card with background, subtle outline, accent stripe, header, and body."""
    rect(slide, x, y, w, h, CARD_BG, line=CARD_BORDER, width=0.8)
    rect(slide, x, y, 0.08, h, accent_color)
    tb(slide, x + 0.18, y + 0.08, w - 0.26, 0.36, [head], size=14, color=accent_color, bold=True, font=FONT_HEADING)
    tb(slide, x + 0.18, y + 0.44, w - 0.26, h - 0.50, paras, size=size, color=SLATE)


def pic(slide, name, x, y, max_w, max_h):
    """Insert an image scaled proportionally within bounds."""
    path = FIG / name
    if not path.exists():
        return
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (max_w - w) / 2), Inches(y + (max_h - h) / 2), Inches(w), Inches(h))


def base_slide(prs, num, kicker, title, notes, title_size=24):
    """Create a standardized slide with header banner, footer, and presenter notes."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    # Top subtle colored bar
    rect(slide, 0, 0, W, 0.08, BLUE)
    # Kicker & Slide Title
    tb(slide, 0.55, 0.22, 12.2, 0.28, [f"SLIDE {num} OF 5  ·  {kicker.upper()}"], size=11, color=MUTED, bold=True, font=FONT_HEADING)
    tb(slide, 0.55, 0.48, 12.2, 0.72, [title], size=title_size, color=NAVY, bold=True, font=FONT_HEADING)
    # Footer
    rect(slide, 0.55, 7.02, 12.2, 0.015, CARD_BORDER)
    tb(slide, 0.55, 7.08, 9.5, 0.32, ["Computer Vision Course Project · Mid-Evaluation · Husnain Rasool"], size=9.5, color=MUTED)
    tb(slide, 11.5, 7.08, 1.25, 0.32, [f"{num} / 5"], size=9.5, color=MUTED, align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def build_deck():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # =========================================================================
    # SLIDE 1: INTRODUCTION
    # =========================================================================
    notes_1 = (
        "[0:00 - 0:50 | Target: 110 words]\n"
        "Good morning, Professor. My semester computer vision project addresses the RSNA Knee Abnormality Detection "
        "challenge. Our objective is to predict twelve binary knee-MRI findings—including ACL, MCL, meniscal tears, "
        "and osteoarthritis compartments—from 3D DICOM volumes without human intervention. The expected output is a "
        "calibrated vector of 12 probabilities evaluated by macro ROC-AUC. In clinical workflows, this serves as an "
        "automated triage and second-reader decision support tool. As illustrated in the system diagram, the pipeline "
        "has two components: during training, a dual-LLM label engine extracts soft supervisory signals from free-text "
        "radiology reports; at inference time, an image-only CoAtNet-2 model processes adjacent slice windows with "
        "multi-instance attention to predict all findings directly from DICOM images."
    )
    s1 = base_slide(prs, 1, "Introduction",
                    "Weak Supervision & Window-Level Attention for Knee-MRI Abnormality Detection",
                    notes_1, title_size=23)

    tb(s1, 0.55, 1.25, 12.2, 0.35, ["RSNA Knee Abnormality Detection (Kaggle 2026) · Multilabel 3D Computer Vision"],
       size=14, color=MUTED, bold=True)

    card(s1, 0.55, 1.70, 4.3, 2.50, "Expected Output", [
        ("12 Abnormality Probabilities per Knee-MRI Study:", dict(size=12, bold=True, color=NAVY)),
        ("• Ligament Tears: ACL, MCL", dict(size=11.5)),
        ("• Meniscal Pathology: Medial Meniscus, Lateral Meniscus", dict(size=11.5)),
        ("• Osteoarthritis (OA): Medial OA, Lateral OA, PF OA", dict(size=11.5)),
        ("• Fluid & Joint: Joint Effusion, Synovitis, Baker's Cyst", dict(size=11.5)),
        ("• Bone Damage: Bone Contusion, Fracture", dict(size=11.5)),
        ("Evaluation Metric: Macro-Averaged ROC-AUC across all 12 findings.", dict(size=11.5, bold=True, color=DARK_BLUE, after=0)),
    ], BLUE)

    card(s1, 0.55, 4.35, 4.3, 2.55, "Clinical Application & Deployment", [
        ("Research Clinical Decision Support:", dict(size=12, bold=True, color=NAVY)),
        ("• Triage and second-reader assistance for radiologists interpreting complex multi-sequence knee MRIs.", dict(size=11.5)),
        ("• Helps prevent missed subtle tears and prioritizes acute cases.", dict(size=11.5)),
        ("Kaggle Code Competition Constraints:", dict(size=12, bold=True, color=NAVY, after=2)),
        ("• Offline execution: Internet disabled, strict 9-hour GPU cap.", dict(size=11.5)),
        ("• Dual submission targets: Main Leaderboard & Efficiency Track.", dict(size=11.5, bold=True, color=EMERALD, after=0)),
    ], EMERALD)

    pic(s1, "system_overview.png", 5.05, 1.70, 7.70, 5.20)

    # =========================================================================
    # SLIDE 2: DATASET
    # =========================================================================
    notes_2 = (
        "[0:50 - 1:45 | Target: 120 words]\n"
        "Turning to the dataset: we use the official RSNA Knee Abnormality Detection benchmark. The training corpus "
        "contains 4,407 patient studies, over 24,000 series, and approximately 819,000 DICOM slices, totaling 531 "
        "gigabytes uncompressed. The hidden test set comprises roughly 1,300 studies. Raw inputs are variable-slice "
        "DICOM series across axial, coronal, and sagittal planes. We preprocess and normalize each study into a unified "
        "64-slice tensor across five anatomical slots, taking 31.8 gigabytes. The central technical challenge is weak "
        "supervision: while all 4,407 studies have radiology reports, only 58 have expert image-review labels. Crucially, "
        "reports do not exist at test time, and naive reports disagree with expert image ground-truth in about 18% of cases "
        "due to reporting mild, sub-threshold findings as positive."
    )
    s2 = base_slide(prs, 2, "Dataset",
                    "RSNA Knee MRI Dataset: Multi-Sequence Volumetric Imaging with Weak Supervision",
                    notes_2, title_size=23)

    card(s2, 0.55, 1.35, 8.1, 1.40, "Name & Direct Access URL", [
        ("Dataset Name: RSNA Knee Abnormality Detection (Kaggle, 2026)", dict(size=13, bold=True, color=NAVY)),
        ("Direct Access URL: " + URL_COMP, dict(size=12, color=BLUE, link=URL_COMP)),
        ("Multi-institutional cohort across 16–19 international clinical sites with 12+ reporting languages.", dict(size=11, color=MUTED)),
    ], BLUE)

    card(s2, 0.55, 2.90, 8.1, 1.65, "Dataset Scale & Storage Footprint", [
        ("Training Cohort: 4,407 studies · 24,371 series · 819,078 DICOM slices", dict(size=12.5, bold=True, color=NAVY)),
        ("Hidden Evaluation Test Set: ~1,300 studies (evaluated server-side with internet disabled)", dict(size=11.5)),
        ("Storage Size: ~531 GB raw DICOM files · Preprocessed Tensor Cache: 31.8 GB (7.2 MB/study)", dict(size=12, bold=True, color=AMBER)),
        ("Cache format: float-normalized uint8 tensors, yielding a 16.7× storage reduction for rapid training.", dict(size=11, color=MUTED)),
    ], AMBER)

    card(s2, 0.55, 4.70, 8.1, 2.20, "Input Formats & Tensor Dimensions", [
        ("Raw DICOM Input:", dict(size=12, bold=True, color=NAVY)),
        ("  Variable series per study (3–14, median 5); 11–320 slices/series; 221 distinct resolutions (mostly 512×512).", dict(size=11)),
        ("Preprocessed 5-Slot Cache Tensor: (N=4407, 64, 336, 336) uint8", dict(size=12, bold=True, font=FONT_CODE, color=DARK_BLUE)),
        ("  Slots: Sag-FS (18), Sag (14), Cor-FS (12), Cor (8), Axial (12); resampled at 140 mm FOV.", dict(size=11)),
        ("Model Input Format: (Batch, K ≤ 54, 3, 384, 384) float32", dict(size=12, bold=True, font=FONT_CODE, color=DARK_BLUE)),
        ("  Adjacent 3-slice RGB windows; Output Target: (Batch, 12) continuous probability vector.", dict(size=11)),
    ], DARK_BLUE)

    # 3 High-Impact KPI stat blocks on the right
    stats = [
        ("4,407", "Training Studies", "All have free-text reports in 12+ languages; 0 reports at test time.", BLUE),
        ("58", "Gold Ground-Truth", "Expert image-review labels (2 MSK radiologists + adjudicator).", AMBER),
        ("~18%", "Report vs. Gold Gap", "Systematic discordance: reports over-call mild/trace findings.", EMERALD),
    ]
    for i, (val, title_text, desc_text, col) in enumerate(stats):
        y_pos = 1.35 + i * 1.80
        rect(s2, 8.85, y_pos, 3.90, 1.65, CARD_BG, line=CARD_BORDER, width=0.8)
        rect(s2, 8.85, y_pos, 0.08, 1.65, col)
        tb(s2, 9.10, y_pos + 0.06, 3.50, 0.70, [val], size=36, color=col, bold=True, font=FONT_HEADING)
        tb(s2, 9.10, y_pos + 0.74, 3.50, 0.30, [title_text], size=13, color=NAVY, bold=True, font=FONT_HEADING)
        tb(s2, 9.10, y_pos + 1.06, 3.50, 0.50, [desc_text], size=11, color=SLATE)

    # =========================================================================
    # SLIDE 3: BASELINE MODEL
    # =========================================================================
    notes_3 = (
        "[1:45 - 2:40 | Target: 115 words]\n"
        "Moving to the baseline: the leading public approach was introduced by Dread Development via Kaggle kernels, "
        "employing a CoAtNet-2 architecture with window-level attention. It lacks a GitHub repo, but uses the open-source "
        "timm vision library. The author-reported baseline achieves 0.924 macro-AUC on the public leaderboard and 0.9167 "
        "on the 58 gold studies. However, critical analysis reveals that this 0.9167 is optimistic and in-sample: the author "
        "selected checkpoint epochs directly on those 58 gold studies. To ensure strict scientific validity, our own "
        "implementation trains from generic ImageNet weights, quarantines the 58 gold studies as a zero-leak tripwire, "
        "and selects epochs exclusively using out-of-fold validation. For reference, our initial DINOv2 slice-attention "
        "baseline reached 0.893 public AUC."
    )
    s3 = base_slide(prs, 3, "Baseline Model",
                    "Public Benchmark Analysis: Architecture, Sources, & Validation Integrity",
                    notes_3, title_size=23)

    card(s3, 0.55, 1.35, 7.9, 2.70, "Code Repositories & Implementation Links", [
        ("Public Baseline: Released as public Kaggle kernels by Dread Development (no GitHub repository)", dict(size=12.5, bold=True, color=NAVY)),
        ("• Inference Kernel: " + URL_INFER, dict(size=11, color=BLUE, link=URL_INFER)),
        ("• Training Kernel: " + URL_TRAIN, dict(size=11, color=BLUE, link=URL_TRAIN)),
        ("Core Backbone Library (timm): PyTorch Image Models (Hugging Face / GitHub)", dict(size=12, bold=True, color=NAVY, after=2)),
        ("• timm GitHub Repo: " + URL_TIMM_GH, dict(size=11, color=BLUE, link=URL_TIMM_GH)),
        ("Our Research Repository: Complete end-to-end framework with automated test suites", dict(size=12, bold=True, color=NAVY, after=2)),
        ("• GitHub Link: " + URL_MINE + " (active research repo; private until competition close)", dict(size=11, color=MUTED, link=URL_MINE)),
    ], BLUE)

    card(s3, 0.55, 4.20, 7.9, 2.70, "Pretrained Model Weights & Initializations", [
        ("Baseline Pretrained Weights (Kaggle Dataset):", dict(size=12.5, bold=True, color=NAVY)),
        ("• Author-released trained checkpoint: " + URL_WEIGHTS, dict(size=11, color=BLUE, link=URL_WEIGHTS)),
        ("• Note: We strictly DID NOT use these knee-trained weights to prevent data contamination.", dict(size=11, color=AMBER)),
        ("Our Model Initialization (Hugging Face / timm):", dict(size=12.5, bold=True, color=NAVY, after=2)),
        ("• CoAtNet-2 ImageNet-12k → 1k Pretrained Weights:", dict(size=11.5, bold=True)),
        ("  " + URL_TIMM_HF, dict(size=11, color=BLUE, link=URL_TIMM_HF)),
        ("• Generic vision features only; no external knee MRI data or third-party labels were utilized.", dict(size=11, color=MUTED)),
    ], DARK_BLUE)

    card(s3, 8.65, 1.35, 4.1, 5.55, "Baseline Performance & Audit", [
        ("Dread Development Baseline:", dict(size=13, bold=True, color=NAVY)),
        ("Public Leaderboard: 0.924", dict(size=24, bold=True, color=AMBER, font=FONT_HEADING)),
        ("Gold-58 AUC: 0.9167 (Optimistic / In-Sample)", dict(size=12, bold=True, color=AMBER)),
        ("• Critical Audit: The baseline selected checkpoint epochs directly on the 58 gold studies. This inflates reported test metrics via subtle data snooping.", dict(size=11, color=SLATE, after=12)),
        ("Our Initial Baseline (DINOv2):", dict(size=13, bold=True, color=NAVY)),
        ("Public Leaderboard: 0.893", dict(size=24, bold=True, color=BLUE, font=FONT_HEADING)),
        ("OOF AUC: 0.8631 · Gold-58 AUC: 0.8903", dict(size=12, bold=True, color=BLUE)),
        ("• 2.5D slice-attention with series-CLS compression. Slower inference (~59 s/study) on Kaggle T4 GPUs.", dict(size=11, color=SLATE, after=12)),
        ("Our Core Principle: Strict split hygiene.", dict(size=11.5, bold=True, color=EMERALD)),
        ("Gold-58 is exclusively used as an untouched zero-leak tripwire, never for epoch or model selection.", dict(size=11, color=SLATE)),
    ], AMBER)

    # =========================================================================
    # SLIDE 4: METHOD & RESULTS
    # =========================================================================
    notes_4 = (
        "[2:40 - 4:05 | Target: 165 words]\n"
        "Slide 4 presents our method, architectural modifications, and quantitative results. While the baseline used "
        "uncalibrated LLM labels and in-sample epoch selection, we introduced four fundamental improvements: First, a dual-LLM "
        "label engine combining Qwen-72B and Llama-70B with clinical severity thresholds, followed by leave-one-out calibration "
        "and noisy-student iteration that raised report label quality from 0.823 to 0.904 LOO-AUC. Second, report-template "
        "GroupKFold clustering across 3,732 clusters, completely eliminating site leakage. Third, slot-isolated windowing "
        "and 3D laterality normalization, correcting coronal mirroring and sagittal slice ordering for right knees across 99.4% "
        "of studies. Fourth, a dual-track submission strategy: a 10-checkpoint seed ensemble for the main leaderboard, and a "
        "2-fold model for the efficiency track. As shown in the table, our 10-checkpoint model achieves 0.884 OOF, 0.9111 on "
        "gold-58, and 0.929 on the public leaderboard. For efficiency, our 2-fold model matches 0.925 public AUC while running "
        "in only 20.5 minutes—hitting the physical DICOM decoding floor on Kaggle's dual T4 hardware."
    )
    s4 = base_slide(prs, 4, "Method & Preliminary Results",
                    "Engineering Rigor: Weak-Supervision Refinement, Architectural Tuning, & Results",
                    notes_4, title_size=23)

    card(s4, 0.55, 1.30, 3.8, 3.55, "Existing Baseline Implementation", [
        ("Architectural Pipeline:", dict(size=12, bold=True, color=NAVY)),
        ("• 5 fixed anatomical slots, 64 slices, 140 mm FOV cropped to 336 px.", dict(size=11)),
        ("• Windows of 3 adjacent slices stacked as RGB into CoAtNet-2 (384 px).", dict(size=11)),
        ("• Per-finding attention-MIL pooling across all study windows.", dict(size=11)),
        ("Weak Supervision & Validation:", dict(size=12, bold=True, color=NAVY, after=2)),
        ("• Single uncalibrated LLM to extract labels from free-text reports.", dict(size=11)),
        ("• Epoch checkpoint selection performed directly on the 58 gold studies.", dict(size=11, color=AMBER)),
        ("• No systematic study of laterality or template-cluster leakage.", dict(size=11)),
    ], MUTED, size=11)

    card(s4, 4.50, 1.30, 4.3, 3.55, "Our Technical Modifications", [
        ("1. Dual-LLM Severity Label Engine:", dict(size=11.5, bold=True, color=NAVY)),
        ("  Qwen-72B + Llama-70B with clinical severity thresholds; LOO calibration & noisy-student distillation (LOO-AUC 0.823 → 0.904).", dict(size=10.5)),
        ("2. Leak-Free GroupKFold Clustering:", dict(size=11.5, bold=True, color=NAVY)),
        ("  MinHash + LSH on report text (3,732 clusters). No template leakage; zero-leak validation; epochs chosen without gold.", dict(size=10.5)),
        ("3. Slot-Isolated Windowing & Laterality Fix:", dict(size=11.5, bold=True, color=NAVY)),
        ("  Windows never cross planes; coronal mirroring + sagittal slice-order reversal resolves 99.4% of right knees correctly.", dict(size=10.5)),
        ("4. Dual-Track Deployment Architecture:", dict(size=11.5, bold=True, color=NAVY)),
        ("  Main Track: 10-checkpoint seed ensemble; Efficiency Track: 2-fold model (~20.5 min on T4, hitting decode floor).", dict(size=10.5)),
    ], EMERALD, size=10.5)

    card(s4, 8.95, 1.30, 3.8, 3.55, "Evaluation Metrics", [
        ("Quantitative Metrics:", dict(size=12, bold=True, color=NAVY)),
        ("• Macro ROC-AUC: Unweighted arithmetic mean across all 12 target findings.", dict(size=11)),
        ("• Three Evaluation Splits: Out-of-Fold (4,349 studies), Gold-58 (unseen), Public LB (~1,300 test studies).", dict(size=11)),
        ("• Efficiency Score: Formula penalized by GPU runtime seconds on 2×T4.", dict(size=11)),
        ("Qualitative Metrics:", dict(size=12, bold=True, color=NAVY, after=2)),
        ("• Attention Localization Maps: Top-attended windows align with true clinical anatomy (e.g. Sag-FS for ACL).", dict(size=11)),
        ("• Per-Finding ROC Curves & AUC Confidence Intervals.", dict(size=11)),
    ], BLUE, size=11)

    # Preliminary Results Table
    table_rows = [
        ("Model / Pipeline Configuration", "OOF Macro", "Gold-58 AUC", "Public LB", "Runtime (2×T4)", "Role / Outcome"),
        ("Public Baseline (Dread Dev)", "—", "0.9167*", "0.924", "—", "Reference Baseline"),
        ("Our 1st Model (DINOv2 Slots, exp005)", "0.8631", "0.8903", "0.893", "~59 s / study", "Initial Exploration"),
        ("exp018: CoAtNet-2 Window-MIL (5 Folds)", "0.8816", "0.9091", "0.929", "~53 min", "Core Architecture"),
        ("exp021: 10-Checkpoint Ensemble (2 Seeds)", "0.8840", "0.9111", "0.929", "~100 min", "Main Track (Selected)"),
        ("exp018 2-Fold: Fast Variant (Folds {0,1})", "—", "0.9078", "0.925", "~20.5 min", "Efficiency Track (Selected)"),
    ]
    gt = s4.shapes.add_table(len(table_rows), 6, Inches(0.55), Inches(5.00), Inches(12.20), Inches(1.85))
    tbl = gt.table
    widths = [3.80, 1.35, 1.65, 1.35, 1.75, 2.30]
    for j, wd in enumerate(widths):
        tbl.columns[j].width = Inches(wd)
    for i, r in enumerate(table_rows):
        tbl.rows[i].height = Inches(0.30)
        is_hdr = (i == 0)
        is_sel = (i in (4, 5))
        for j, val in enumerate(r):
            c = tbl.cell(i, j)
            c.text = val
            c.fill.solid()
            if is_hdr:
                c.fill.fore_color.rgb = DARK_BLUE
            elif is_sel:
                c.fill.fore_color.rgb = RGBColor(0xEC, 0xFD, 0xF5) if i == 5 else RGBColor(0xF0, 0xF9, 0xFF)
            else:
                c.fill.fore_color.rgb = CARD_BG if i % 2 == 0 else WHITE
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.margin_left = c.margin_right = Inches(0.05)
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if (j == 0 or j == 5) else PP_ALIGN.CENTER
                for run in p.runs:
                    run.font.name = FONT_BODY
                    run.font.size = Pt(11 if is_hdr else 10.5)
                    run.font.bold = is_hdr or is_sel
                    run.font.color.rgb = WHITE if is_hdr else (DARK_BLUE if is_sel else NAVY)

    tb(s4, 0.55, 6.85, 12.20, 0.22,
       ["*Baseline 0.9167 is optimistic due to in-sample epoch selection on gold-58. Our models never train or select on gold-58."],
       size=9.5, color=MUTED)

    # =========================================================================
    # SLIDE 5: PLAN
    # =========================================================================
    notes_5 = (
        "[4:05 - 4:55 | Target: 115 words]\n"
        "To conclude, Slide 5 outlines our next steps and planned improvements, grounded directly in the experiments "
        "we have already completed. In earlier rounds, we rigorously evaluated and ruled out unproductive paths: scaling "
        "slice count to 32 or resolution to 448 pixels proved completely null, ConvNeXt backbones were 93% redundant, and "
        "multilingual report-text auxiliary losses failed to surpass our pre-registered bar. With these insights, our next steps "
        "focus on high-impact validation and deployment: First, conducting a clean single-variable ablation to isolate the "
        "individual contributions of the four bundled changes in the 0.929 model. Second, partnering with an MSK radiologist "
        "to clinically validate our attention maps against anatomical lesions. Third, deploying FP16 TensorRT compilation to "
        "enable full 5-fold ensembling under the 20-minute efficiency budget. Finally, open-sourcing our code and model weights "
        "upon competition close on October 22. Thank you, and I look forward to your questions."
    )
    s5 = base_slide(prs, 5, "Plan & Completed Experiments",
                    "Empirical Learnings, Scientific Ablations, & Next Steps",
                    notes_5, title_size=23)

    card(s5, 0.55, 1.30, 5.8, 2.70, "Completed Experiments & Empirical Nulls", [
        ("Evidence from Rigorous Pre-Registered Hypotheses:", dict(size=12, bold=True, color=NAVY)),
        ("• Backbone Explorations: DINOv3 ViT-S was significantly negative (−0.025 OOF); ConvNeXt-V2 was 93% redundant with ViTs; 3D ResNet-50 stalled at 0.693 AUC.", dict(size=10.5)),
        ("• Spatial & Resolution Scaling: 448 px (vs 384 px) and T=32 slices (vs 24) were strictly null (Δ ≈ −0.0001); dense 87 windows (vs 51) was also null.", dict(size=10.5)),
        ("• Auxiliary Multilingual Text Loss: BGE-M3 text regression (+0.0036 OOF) failed our pre-registered 2-SD bar (+0.0082) and did not hold on gold-58.", dict(size=10.5)),
        ("• Runtime Profiling: Proved CoAtNet-2 is model-bound on T4 (~9.7 min/fold), directly motivating the 2-fold efficiency breakthrough.", dict(size=10.5, bold=True, color=DARK_BLUE)),
    ], BLUE, size=10.5)

    card(s5, 6.55, 1.30, 6.2, 2.70, "Actionable Planned Improvements", [
        ("1. Single-Variable Ablation Study:", dict(size=11.5, bold=True, color=NAVY)),
        ("  Disentangle the 0.893 → 0.929 jump by isolating the individual gains of: CoAtNet backbone, Window-MIL pooling, 140 mm FOV, and laterality fix.", dict(size=10.5)),
        ("2. Clinical Expert Validation of Attention Maps:", dict(size=11.5, bold=True, color=NAVY)),
        ("  Engage an MSK radiologist to grade whether high-attention windows (top-4 per finding) align with true anatomical pathology rather than imaging artifacts.", dict(size=10.5)),
        ("3. TensorRT & GPU DICOM Acceleration:", dict(size=11.5, bold=True, color=NAVY)),
        ("  Compile CoAtNet-2 into FP16 TensorRT / torch.compile with nvJPEG2000 decoding, aiming to run all 5 folds within the 20.5-min efficiency envelope.", dict(size=10.5)),
        ("4. Reconcile 2/58 Deployment Outliers:", dict(size=11.5, bold=True, color=NAVY)),
        ("  Investigate the 2 studies showing minor divergence between training and Kaggle deployment, resolving edge cases in patient orientation tags.", dict(size=10.5)),
    ], EMERALD, size=10.5)

    tb(s5, 0.55, 4.15, 12.2, 0.35, ["Project Milestones & Delivery Roadmap"], size=16, color=NAVY, bold=True, font=FONT_HEADING)

    roadmap_steps = [
        ("Oct 22, 2026", "Competition Closes", "Private leaderboard reveal evaluates test generalization on ~1,300 hidden cases.", AMBER),
        ("Late Oct 2026", "Code & Weights Release", "Publish Apache-2.0 repository, calibrated models, and reproducible documentation.", BLUE),
        ("Nov 23–27", "Live Demonstration", "Course project live demo showcasing the model inference pipeline & attention overlays.", DARK_BLUE),
        ("Dec 05, 2026", "Final Research Report", "Submission of comprehensive conference-style paper with complete ablation tables.", EMERALD),
    ]
    step_w = 2.85
    step_gap = 0.26
    for i, (date_text, title_text, desc_text, col) in enumerate(roadmap_steps):
        x_pos = 0.55 + i * (step_w + step_gap)
        y_pos = 4.60
        rect(s5, x_pos, y_pos, step_w, 2.30, CARD_BG, line=CARD_BORDER, width=0.8)
        rect(s5, x_pos, y_pos, step_w, 0.40, col)
        tb(s5, x_pos + 0.10, y_pos + 0.05, step_w - 0.20, 0.30, [date_text], size=12, color=WHITE, bold=True, align=PP_ALIGN.CENTER, font=FONT_HEADING)
        tb(s5, x_pos + 0.12, y_pos + 0.50, step_w - 0.24, 0.35, [title_text], size=12.5, color=NAVY, bold=True, font=FONT_HEADING)
        tb(s5, x_pos + 0.12, y_pos + 0.90, step_w - 0.24, 1.30, [desc_text], size=11, color=SLATE)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Successfully generated {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build_deck()
