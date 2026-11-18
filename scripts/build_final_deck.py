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

"""Build the final course presentation (12 min + 3 min Q&A, 24 / 26 Nov 2026).

Follows the instructor's structure: 1 Introduction, 2 Challenges, 3 Related work, 4 Methodology and results,
5 Demonstration. Speaker notes on every slide (about 1,500 words in total, about 12 minutes).
Same visual style as scripts/build_mid_eval_deck.py. Result figures: scripts/fig_final_deck.py.

    python scripts/build_final_deck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mid_eval_deck import (AMBER, BLUE, CARD_BORDER, DARK_BLUE, EMERALD, FONT_HEADING, H, MUTED,  # noqa: E402
                                 NAVY, SLATE, URL_COMP, URL_MINE, URL_TIMM_HF, WHITE, W, card, pic, rect, tb)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/presentation/Team10_FinalPresentation_KneeMRIAbnormalityDetection.pptx"
N_SLIDES = 12
RED = AMBER.__class__(0xB9, 0x1C, 0x1C)


def base(prs, num, section, title, notes, title_size=24):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, 0.08, BLUE)
    tb(s, 0.55, 0.22, 12.2, 0.28, [section.upper()], size=11, color=MUTED, bold=True, font=FONT_HEADING)
    tb(s, 0.55, 0.48, 12.2, 0.72, [title], size=title_size, color=NAVY, bold=True, font=FONT_HEADING)
    rect(s, 0.55, 7.02, 12.2, 0.015, CARD_BORDER)
    tb(s, 0.55, 7.08, 9.5, 0.32, ["Team 10 · Computer Vision Course · Final Presentation · Husnain Rasool"],
       size=9.5, color=MUTED)
    tb(s, 11.5, 7.08, 1.25, 0.32, [f"{num} / {N_SLIDES}"], size=9.5, color=MUTED, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def table(slide, rows, x, y, w, h, widths, size=10.5, highlight=None):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for j, cw in enumerate(widths):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.text = v
            p = c.text_frame.paragraphs[0]
            p.font.size = Pt(size)
            p.font.name = "Segoe UI"
            p.font.bold = i == 0 or (highlight is not None and i == highlight)
            p.font.color.rgb = WHITE if i == 0 else NAVY
            c.fill.solid()
            c.fill.fore_color.rgb = DARK_BLUE if i == 0 else (
                AMBER.__class__(0xFE, 0xF3, 0xC7) if i == highlight else
                (AMBER.__class__(0xF8, 0xFA, 0xFC) if i % 2 else WHITE))
            c.margin_left = c.margin_right = Inches(0.05)
            c.margin_top = c.margin_bottom = Inches(0.02)
    return t


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # ---------------------------------------------------------------- 1. Introduction
    s = base(prs, 1, "1 · Introduction", "Detecting 12 Knee-MRI Abnormalities When Only 58 Studies Are Labelled", (
        "[0:00-1:00] Our project is the RSNA 2026 Knee Abnormality Detection challenge. The input is one knee MRI "
        "exam: three to fourteen DICOM series in sagittal, coronal and axial planes. The output is twelve "
        "probabilities: ligament tears, meniscal tears, osteoarthritis in three compartments, effusion, synovitis, "
        "Baker's cyst, contusion and fracture. The score is the mean ROC-AUC over the twelve findings. Why it "
        "matters: knee MRI is one of the most common MSK exams, and a reliable second reader helps triage and "
        "reduces missed findings. What makes it a research problem is the supervision: the training set has 4,407 "
        "exams but only 58 carry expert labels. Every exam has a free-text report, but reports are not available "
        "at test time, so the final model must work from images alone."), title_size=23)
    tb(s, 0.55, 1.2, 12.2, 0.35, ["RSNA Knee Abnormality Detection · Kaggle 2026 · " + URL_COMP],
       size=12.5, color=BLUE, bold=True)
    card(s, 0.55, 1.7, 4.2, 2.45, "Input", [
        ("One knee MRI study (DICOM)", dict(size=12.5, bold=True, color=NAVY)),
        ("• 3–14 series per study (median 5)", dict(size=11.5)),
        ("• Sagittal / coronal / axial; fat-suppressed or not", dict(size=11.5)),
        ("• 11–320 slices per series (median 30)", dict(size=11.5)),
        ("• No report at test time: images only", dict(size=11.5, bold=True, color=AMBER)),
    ], BLUE)
    card(s, 0.55, 4.3, 4.2, 2.6, "Output and metric", [
        ("12 probabilities per study", dict(size=12.5, bold=True, color=NAVY)),
        ("• ACL, MCL · medial / lateral meniscus", dict(size=11.5)),
        ("• Medial, lateral, patellofemoral OA", dict(size=11.5)),
        ("• Effusion, synovitis, Baker's cyst", dict(size=11.5)),
        ("• Bone contusion, fracture", dict(size=11.5)),
        ("Metric: macro ROC-AUC over the 12 findings", dict(size=11.5, bold=True, color=DARK_BLUE)),
    ], EMERALD)
    pic(s, "deck_mri_strip.png", 4.95, 1.7, 7.8, 2.9)
    card(s, 4.95, 4.75, 7.8, 2.15, "Motivation", [
        ("• Knee MRI is a high-volume exam; an automatic second reader can triage and reduce missed findings.",
         dict(size=12)),
        ("• Research question: how far can image-only models go when training labels come from free-text reports"
         " in 12+ languages, and only 58 studies carry expert image labels?", dict(size=12, bold=True, color=NAVY)),
    ], AMBER)

    # ---------------------------------------------------------------- 2. Challenges
    s = base(prs, 2, "2 · Challenges", "Five Obstacles and How We Handled Each", (
        "[1:00-2:15] Five obstacles shaped the project. First and largest: weak supervision. Fifty-eight expert "
        "labels for 4,407 studies. We built a label engine: two open 70-billion-parameter LLMs read every report "
        "with a severity rubric, calibrated on the 58. Second: reports and expert image reads disagree, about 18 "
        "percent of cells. Third: heterogeneous DICOM, 531 gigabytes in four compression formats, different series "
        "per study. We decoded once into a cache and built fixed anatomical slots. Fourth: laterality. Medial and "
        "lateral are mirrored between left and right knees, so we detect the side and mirror right knees; this "
        "resolved 99.4 percent. Fifth: the Kaggle limits: no internet, nine hours of GPU, and a crash means no "
        "score. We kept split hygiene strict: folds grouped by report template, and the 58 gold studies never used "
        "for training or checkpoint selection."))
    rows = [("Challenge", "Why it is hard", "What we did"),
            ("Weak supervision", "58 expert-labelled studies of 4,407; reports absent at test time",
             "LLM label engine (Qwen2.5-72B + Llama-3.3-70B, vLLM) → soft labels, LOO-calibrated on gold-58"),
            ("Report ≠ image truth", "~18% of cells disagree; experts apply severity thresholds",
             "Severity rubric per finding; noisy-student round with image-model predictions (labels v3)"),
            ("Heterogeneous DICOM", "531 GB; JPEG-lossless / JPEG-2000; 3–14 series; 221 matrix sizes",
             "Decode once to a cache; 5 anatomical slots; fixed physical field of view"),
            ("Laterality", "Medial / lateral flip between left and right knees",
             "Side from tags, geometry and reports; mirror right knees (99.4% of studies resolved)"),
            ("Leakage and limits", "Template-shared reports; 9 h GPU, offline, crash = no score",
             "GroupKFold on 3,681 report clusters; gold-58 never used for selection; wall-clock guards")]
    table(s, rows, 0.55, 1.35, 12.2, 4.6, [2.3, 4.4, 5.5], size=11.5)
    tb(s, 0.55, 6.15, 12.2, 0.8, [("Every number on later slides uses this split: 5 folds by report cluster; gold-58 held out as "
                                    "a tripwire, never for model selection.", dict(size=12.5, bold=True, color=EMERALD))])

    # ---------------------------------------------------------------- 3. Related work (literature)
    s = base(prs, 3, "3 · Related work", "Prior Work: Multi-Plane Knee MRI and Learning From Weak Labels", (
        "[2:15-3:15] Related work comes from three directions. Knee MRI: MRNet from Stanford trained one CNN per "
        "plane and combined them, for three labels. Qiu and colleagues in Nature Communications 2024 used co-plane "
        "attention across sequences for twelve knee abnormalities, the closest published work to this task. Weak "
        "labels: CheXpert showed report-mined labels with uncertainty can train strong image models, and Noisy "
        "Student showed self-training with a teacher's predictions. Architecture: attention-based multiple-instance "
        "learning pools many instances into one bag label. Past RSNA spine challenges were won by 2.5D encoders "
        "with a sequence model across slices. We combine these: report-mined soft labels, a noisy-student round, "
        "and 2.5D window-level attention MIL."))
    card(s, 0.55, 1.35, 6.0, 2.7, "Knee MRI", [
        ("MRNet (Bien et al., PLoS Med 2018)", dict(size=12.5, bold=True, color=NAVY)),
        ("One CNN per plane, combined by logistic regression; 1,370 exams; 3 labels (abnormal, ACL, meniscus).",
         dict(size=11.5)),
        ("Co-plane attention (Qiu et al., Nat Commun 2024)", dict(size=12.5, bold=True, color=NAVY)),
        ("Attention across MRI sequences and planes for 12 knee abnormalities: the closest published task.",
         dict(size=11.5)),
    ], BLUE)
    card(s, 6.75, 1.35, 6.0, 2.7, "Learning from weak labels", [
        ("CheXpert (Irvin et al., AAAI 2019)", dict(size=12.5, bold=True, color=NAVY)),
        ("Labels mined from radiology reports, with explicit uncertainty; strong image models trained on them.",
         dict(size=11.5)),
        ("Noisy Student (Xie et al., CVPR 2020)", dict(size=12.5, bold=True, color=NAVY)),
        ("A teacher's predictions become soft targets for a student; we use one round to repair report labels.",
         dict(size=11.5)),
    ], EMERALD)
    card(s, 0.55, 4.2, 12.2, 2.7, "Architectures we build on", [
        ("Attention-based MIL (Ilse et al., ICML 2018): many instances (here: 3-slice windows) → one bag label, with "
         "learned attention weights that are also interpretable.", dict(size=12)),
        ("CoAtNet (Dai et al., NeurIPS 2021) from timm; DINOv2 (Oquab et al., 2023) for our first slot model.",
         dict(size=12)),
        ("RSNA 2023 / 2024 challenge pattern: a 2.5D slice encoder plus a sequence model across slices.",
         dict(size=12)),
        ("Our combination: report-mined soft labels → noisy-student repair → 2.5D CoAtNet-2 window encoder → "
         "per-finding attention MIL.", dict(size=12, bold=True, color=NAVY)),
    ], AMBER)

    # ---------------------------------------------------------------- 4. Related work (top solutions)
    s = base(prs, 4, "3 · Related work", "What the Winning Teams Did (Published Writeups, Oct–Nov 2026)", (
        "[3:15-4:15] After the competition, the top teams published their solutions; we read the top ten. The "
        "pattern is clear. Every top team added image-based supervision: most used a vision-language model that read "
        "the training MRIs together with the report; the sixth-place team used image models trained on external "
        "expert grades. Some added external knee MRI from the "
        "Osteoarthritis Initiative with expert MOAKS grades. The architectures are close to ours: the fourth-place "
        "team used the same CoAtNet-2 window MIL. Image-read labels scored 0.936 to 0.956 against the 58 "
        "gold studies; our report-only labels scored about 0.90. So the field converged on one lesson: in this "
        "task, the labels decide the rank."))
    rows = [("Rank", "Private", "Training labels", "Model", "Extra data"),
            ("1st", "top", "LLM agents read images + report; ROI and lesion boxes", "ConvNeXt-T box pooling", "—"),
            ("2nd", "0.960", "Report LLM + VLM (all slices) + teacher mix", "6 × 2.5D MIL", "fastMRI, OAI"),
            ("3rd", "0.959", "VLM image + report, finding-specific readers", "36-model 2.5D ensemble", "OAI"),
            ("4th", "0.958", "VLM reads rendered MRI pages", "CoAtNet-2 window MIL (= ours)", "—"),
            ("6th", "0.958", "Report LLMs + rules + MOAKS second reader", "Knee-frame tiles, 55 models", "OAI, fastMRI"),
            ("7th", "0.957", "VLM: 3 reads + teacher", "Label-query 2.5D, 4 models", "—"),
            ("10th", "0.957", "LLM/VLM blend + OAI mapping", "Multi-backbone + CoAtNet MIL", "OAI"),
            ("Ours", "0.912", "Report-only LLMs + noisy student", "CoAtNet-2 window MIL", "—")]
    table(s, rows, 0.55, 1.35, 12.2, 4.3, [0.9, 1.0, 4.5, 3.6, 2.2], size=11, highlight=8)
    tb(s, 0.55, 5.85, 12.2, 1.1, [
        ("Lesson: every top team added image-based supervision: a VLM reading the training MRIs (1st–4th, 7th, 10th) or "
         "image models trained on external expert grades (6th). Image-read labels scored 0.936–0.956 on gold-58; ours about 0.90.", dict(size=12.5, bold=True,
                                                                                            color=NAVY)),
        ("1st place: final private score not stated in the writeup; the top private score was 0.960.",
         dict(size=10, color=MUTED))])

    # ---------------------------------------------------------------- 5. Data and tools
    s = base(prs, 5, "4 · Methodology", "Data Structure, Formats and Tools", (
        "[4:15-5:00] The data: 4,407 training studies, 24,371 series, about 819,000 DICOM slices, 531 gigabytes. "
        "The hidden test set has about 1,300 studies. We decode each study once into a cache: five anatomical "
        "slots, sagittal fat-suppressed and not, coronal fat-suppressed and not, and axial; 64 slices at a 140 mm "
        "field of view. The model input is up to 54 windows of three adjacent slices, each a 384 by 384 RGB image; "
        "the output is a 12-vector. Tools: pydicom and pylibjpeg for decoding, timm for the CoAtNet-2 backbone, "
        "vLLM to run Qwen and Llama on the university H100 cluster, and SimpleITK for the registration audit."))
    card(s, 0.55, 1.35, 6.0, 5.55, "Data and tensor formats", [
        ("Training: 4,407 studies · 24,371 series · ~819k slices · 531 GB", dict(size=12, bold=True, color=NAVY)),
        ("Hidden test: ~1,300 studies, scored on Kaggle (offline, ≤ 9 h GPU)", dict(size=11.5)),
        ("Labels: 12 binary findings; 58 expert studies; 4,407 reports", dict(size=11.5)),
        ("Cache (decoded once):", dict(size=12, bold=True, color=NAVY)),
        ("(4407, 64, 336, 336) uint8 · 5 slots: Sag-FS 18 · Sag 14 · Cor-FS 12 · Cor 8 · Ax 12", dict(size=11,
                                                                                                    font="Consolas")),
        ("Model input:", dict(size=12, bold=True, color=NAVY)),
        ("(B, K ≤ 54, 3, 384, 384) float · 3 adjacent slices per window", dict(size=11, font="Consolas")),
        ("Model output:", dict(size=12, bold=True, color=NAVY)),
        ("(B, 12) probabilities", dict(size=11, font="Consolas")),
        ("Research cache v2 (post-competition): all 24,371 series, every native slice, exact geometry, 113 GB.",
         dict(size=11, color=MUTED)),
    ], BLUE)
    card(s, 6.75, 1.35, 6.0, 5.55, "Repositories, models and tools", [
        ("Backbone: CoAtNet-2 (timm), ImageNet-12k → 1k weights", dict(size=12, bold=True, color=NAVY)),
        (URL_TIMM_HF, dict(size=10, color=BLUE, link=URL_TIMM_HF)),
        ("Label engine: Qwen2.5-72B-Instruct + Llama-3.3-70B via vLLM", dict(size=12, bold=True, color=NAVY)),
        ("DICOM: pydicom + pylibjpeg / GDCM (JPEG lossless, JPEG 2000)", dict(size=11.5)),
        ("Registration audit: SimpleITK (Powell, mutual information)", dict(size=11.5)),
        ("Compute: NU HPC (SLURM, H100) for training and labels; RTX 4090 locally; Kaggle 2×T4 for inference",
         dict(size=11.5)),
        ("Baseline we started from: public CoAtNet-2 window-MIL kernel (public LB 0.924)", dict(size=11.5)),
        ("Our code: " + URL_MINE, dict(size=11, color=BLUE, link=URL_MINE)),
        ("Apache-2.0; tests for decoding, geometry and data loaders", dict(size=11, color=MUTED)),
    ], EMERALD)

    # ---------------------------------------------------------------- 6. Pipeline
    s = base(prs, 6, "4 · Methodology", "Our System: Report-Mined Soft Labels → 2.5D Window-Level Attention MIL", (
        "[5:00-6:00] The system has two lanes. Top lane, training only: two LLMs read each report with a "
        "severity rubric and return a probability per finding. We calibrate per finding on the 58 gold studies "
        "with leave-one-out, then do one noisy-student round: we blend in out-of-fold image-model predictions. "
        "This raised label agreement on gold from 0.823 to 0.904 AUC. Bottom lane, training and test: DICOM to "
        "five anatomical slots, right knees mirrored, three-slice windows, a shared CoAtNet-2 encoder, and one "
        "attention head per finding that pools over all windows of the study. Attention tells us which windows "
        "drove each prediction. Final submission: ten checkpoints, two seeds by five folds."))
    pic(s, "system_overview_wide.png", 0.55, 1.3, 12.2, 4.3)
    card(s, 0.55, 5.7, 6.0, 1.25, "Labels (training only)", [
        ("Report LLMs → per-finding LOO calibration → noisy-student blend with image OOF.  "
         "Gold-58 agreement: 0.823 → 0.904 AUC.", dict(size=11.5)),
    ], AMBER)
    card(s, 6.75, 5.7, 6.0, 1.25, "Image model (train and test)", [
        ("5 slots → 3-slice windows → CoAtNet-2 → per-finding attention MIL → 12 probabilities. "
         "Final: 2 seeds × 5 folds.", dict(size=11.5)),
    ], BLUE)

    # ---------------------------------------------------------------- 7. Competition results
    s = base(prs, 7, "4 · Results", "Competition Results: 0.912 Private, Rank 3,970 of 6,425", (
        "[6:00-7:00] Results on three evaluation sets: out-of-fold over all training studies against our labels, "
        "the 58 gold studies against expert labels, and the Kaggle leaderboard against expert labels on about "
        "1,300 hidden studies. The window-MIL model moved us from 0.893 to 0.929 public. The final ten-checkpoint "
        "ensemble scored 0.912 on the private leaderboard, rank 3,970 of 6,425. The public notebooks plateaued "
        "at 0.942 and the winners reached 0.960. The right panel explains most of the gap: the fourth-place team "
        "trained the same architecture as ours; with report labels they got 0.925 on one fold, with image-read "
        "labels 0.952. The architecture was not our bottleneck; the labels were."))
    rows = [("Model", "OOF", "Gold-58", "Public", "Private", "Runtime"),
            ("DINOv2-S slot model (exp005)", "0.863", "0.890", "0.893", "—", "~59 s / study"),
            ("CoAtNet-2 window MIL, 5 folds (exp018)", "0.882", "0.909", "0.929", "—", "~53 min"),
            ("10 checkpoints, 2 seeds (exp021) · main", "0.884", "0.911", "0.929", "0.912", "~100 min"),
            ("2 folds (efficiency track)", "—", "0.908", "0.925", "—", "~20.5 min")]
    table(s, rows, 0.55, 1.35, 12.2, 2.0, [4.4, 1.3, 1.4, 1.4, 1.4, 2.3], size=11, highlight=3)
    pic(s, "final/fig1_leaderboard_labels.png", 0.55, 3.5, 12.2, 3.45)

    # ---------------------------------------------------------------- 8. Why: label noise
    s = base(prs, 8, "4 · Results", "Why We Trail the Winners: Our Labels Miss Positives", (
        "[7:00-7:50] Why are our labels the problem? We compared our training labels with the 58 expert studies, "
        "finding by finding. We expected reports to over-call mild disease. The opposite happened: errors are "
        "mostly missed positives. For MCL, lateral OA and contusion, our labels miss 42 to 55 percent of the "
        "expert positives. A missed finding is often simply not mentioned in the report: mild disease that a "
        "report has no reason to describe. The second-place team found the same: experts marked 240 positives "
        "where reports stated 154. A label made from text alone cannot recover what the text omits; a model that "
        "reads the images can. That is what every winning team did."))
    pic(s, "final/fig4_label_noise.png", 0.55, 1.3, 12.2, 3.75)
    card(s, 0.55, 5.2, 6.0, 1.75, "Finding", [
        ("Errors are mostly missed positives (η⁻ ≫ η⁺ for 10 of 12 findings), the opposite of our design "
         "assumption that reports over-call.", dict(size=11.5)),
    ], RED)
    card(s, 6.75, 5.2, 6.0, 1.75, "Consistent with the winners", [
        ("2nd place: experts mark 240 positives where reports state 154 (PF OA: 21 vs 2). The fix is an "
         "image-reading labeller.", dict(size=11.5)),
    ], EMERALD)

    # ---------------------------------------------------------------- 9. Post-competition: ladder + ACL
    s = base(prs, 9, "4 · Results", "After the Competition (1): Does Using All the Data Help?", (
        "[7:50-8:50] After the deadline we asked research questions with pre-registered rules: we wrote the "
        "decision rule before running. First: our cache used only about a third of the slices. We built cache "
        "version 2 with every series and every native slice, and added the changes one at a time: a ladder from "
        "B0 to B5. Panel a: macro AUC stays flat within seed noise. Panel b: ACL drops, but the two B0 seeds alone "
        "differ by 0.022. We tested the obvious explanation, slice spacing inside each window, with a "
        "pre-registered test: wider spacing made ACL worse, not better. Conclusion: more data did not help this "
        "pipeline, and the ACL drop is most likely seed noise."))
    pic(s, "final/fig2_ladder_acl.png", 0.55, 1.3, 12.2, 3.95)
    tb(s, 0.55, 5.4, 12.2, 1.5, [
        ("• Using all 24,371 series and every native slice: macro-AUC 0.9557 → 0.9568 (within seed noise).",
         dict(size=12.5)),
        ("• Pre-registered rule: adopt a triplet spacing only if the ACL 95% CI is above 0 and macro falls by no more "
         "than 0.002. Neither arm qualified; ±10.5 mm: ACL −0.022 [−0.038, −0.007].", dict(size=12.5)),
        ("• Fold 0, seeds 42 / 43, paired cluster bootstrap (1,000 reps over 672 report clusters).",
         dict(size=11, color=MUTED))])

    # ---------------------------------------------------------------- 10. Geometry experiments
    s = base(prs, 10, "4 · Results", "After the Competition (2): Does Cross-Plane Geometry Help?", (
        "[8:50-9:55] Second research question, our main innovation attempt. MRI planes are acquired separately, "
        "but DICOM headers tell us exactly where each slice sits in 3D. We added a geometry bias to cross-plane "
        "attention: tokens from different planes attend more to each other where their slice footprints "
        "physically overlap. We pre-registered two causal controls. Donor swap: give each study another study's "
        "geometry; if correct geometry matters, AUC should drop. Beta-off: remove the learned bias at test time. "
        "Panel a: all effects sit inside seed noise; the donor swap changes nothing. Panel b: with local features "
        "the model does learn strong locality, but correct geometry adds no information. By our pre-registered "
        "rule, we stopped the geometry line. Note: other teams used geometry as an input representation, which "
        "we did not test."))
    pic(s, "final/fig3_geometry_controls.png", 0.55, 1.3, 12.2, 4.05)
    tb(s, 0.55, 5.5, 12.2, 1.45, [
        ("• Mahalanobis footprint bias in cross-plane attention, β initialised at 0; tested on final-stage and on "
         "local (stride-8) tokens.", dict(size=12.5)),
        ("• Pre-registered STOP rule fired twice: donor geometry (median 26 mm off) gives the same AUC (Δ ≤ 0.00002).",
         dict(size=12.5)),
        ("• Narrow claim only: this attention bias adds nothing here. Geometry as an input frame (used by the 1st, 6th "
         "and 7th places) was not tested.", dict(size=11.5, color=MUTED))])

    # ---------------------------------------------------------------- 11. Demonstration
    s = base(prs, 11, "5 · Demonstration", "Demo: One Study In, Twelve Probabilities and Their Evidence Out", (
        "[9:55-11:05] Demonstration. Here is the offline Kaggle submission notebook running the model on a test "
        "study: it decodes the DICOM series, builds the five slots, runs the ten checkpoints and writes twelve "
        "probabilities. [Show the notebook or the recorded clip.] The figure shows what the model looked at, for "
        "three gold studies chosen by a fixed rule before any map was viewed, so this is a best-case illustration. "
        "For ACL the attention sits on the central sagittal fat-suppressed windows; for the medial meniscus on "
        "coronal fat-suppressed windows; for Baker's cyst on axial windows. The top four windows carry 31 to 48 "
        "percent of the attention, so a radiologist can check a handful of slices instead of the whole exam."))
    pic(s, "attention_windows_gold58_slide.png", 0.55, 1.3, 7.9, 5.6)
    card(s, 8.65, 1.3, 4.1, 5.6, "Live demo (about 1 minute)", [
        ("1. Input: one study folder of DICOM series", dict(size=12)),
        ("2. Decode → 5 slots → right knee mirrored", dict(size=12)),
        ("3. 10 checkpoints → 12 probabilities", dict(size=12)),
        ("4. Evidence: top-attended windows per finding", dict(size=12)),
        ("Runs offline in the Kaggle notebook: about 4.5 s per study on one T4 GPU for 5 folds.",
         dict(size=11.5, color=MUTED)),
        ("Left: three gold-58 studies chosen by a fixed rule (best case, not a hit rate). The top 4 of 54 windows "
         "carry 31–48% of the attention.", dict(size=11.5, bold=True, color=NAVY)),
    ], BLUE)

    # ---------------------------------------------------------------- 12. Conclusions
    s = base(prs, 12, "Conclusions", "What We Learned", (
        "[11:05-12:00] To conclude. We built a complete, image-only system that runs offline within the Kaggle "
        "limits and scored 0.912 private. The winners reached 0.960, and the published solutions plus our own "
        "label analysis show the main reason: labels made from report text miss positives that only the images "
        "show. After the competition we ran pre-registered experiments: more data and cross-plane geometry "
        "attention did not help this pipeline, and we can say why with causal controls. If we started again, we "
        "would first build an image-reading labeller and evaluate against expert labels, and only then change "
        "the architecture. Thank you; happy to take questions."))
    card(s, 0.55, 1.35, 6.0, 2.6, "Results", [
        ("• Complete image-only system: offline, within the 9 h limit; private 0.912.", dict(size=12.5)),
        ("• Gap to the winners (0.960) is explained mainly by labels, not architecture.", dict(size=12.5)),
        ("• Two pre-registered negative results with causal controls: all data (ladder) and cross-plane geometry "
         "attention.", dict(size=12.5)),
    ], BLUE)
    card(s, 6.75, 1.35, 6.0, 2.6, "Lessons", [
        ("• In weak supervision, measure label quality first: our labels missed 22–55% of gold positives on 7 "
         "findings.", dict(size=12.5)),
        ("• Evaluate against expert labels: fold-validation against noisy labels can hide real gains.",
         dict(size=12.5)),
        ("• Write the decision rule before running: it kept us from chasing noise.", dict(size=12.5)),
    ], EMERALD)
    card(s, 0.55, 4.1, 12.2, 2.8, "Next steps", [
        ("• Image-reading labeller (vision-language model over the MRI slices, report as a hint), validated on "
         "gold-58.", dict(size=12.5)),
        ("• Re-run the geometry comparison on a competitive baseline, scored by Kaggle late submission against expert "
         "labels, across 5 folds; add geometry-as-input (resampling planes into a common knee frame).",
         dict(size=12.5)),
        ("• Only then a short negative-results paper (MELBA or a MICCAI 2027 workshop), with full AI-use disclosure.",
         dict(size=12.5)),
    ], AMBER)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    words = sum(len(sl.notes_slide.notes_text_frame.text.split()) for sl in prs.slides)
    print(f"wrote {OUT.name}: {len(prs.slides)} slides, {words} words of notes (~{words / 130:.1f} min at 130 wpm)")


if __name__ == "__main__":
    build()
