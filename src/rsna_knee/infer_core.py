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

"""Image-only inference: DICOM -> 6-slot volumes -> 5-fold ensemble -> submission.csv.

Shared by scripts/infer.py (HPC/local rehearsal against the tiny public sample test set) and
notebooks/submit-exp002.ipynb (the real Kaggle submission — same functions, just pointed at
/kaggle/input paths and run with internet off, no report data, no cache, no final_side.csv).

Key differences from training (src/rsna_knee/dataset.py) — all deliberate, all because test time
has strictly less information than train time (CLAUDE.md §2: "Reports are NOT provided at test
time"):
  - Side resolution is DICOM-tag-only (no report-extracted side, no final_side.csv — that table
    was built by cross-checking against reports). Per-study side is the majority DICOM Laterality
    tag across that study's series; if tag-less or conflicting, the CLAUDE.md §12 risk-register
    mitigation applies: run BOTH orientations and average (with the paired-label swap), rather
    than guessing a side.
  - DICOMs are decoded live via src/rsna_knee/preprocess.py (identical math to the cache-building
    pipeline) instead of reading a prebuilt cache — there is no cache at test time.
  - Every per-study step is wrapped so one bad study degrades to the label-prior row instead of
    crashing the whole run (CLAUDE.md §9: "a crash = no score").
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import (
    LATERALITY_SWAP_PAIRS, NUM_SLOTS, SLOT_NAMES, TARGET_COLUMNS, _patch_size_for_backbone,
    slot_index,
)
from .preprocess import decode_series_to_uint8

_DINOV2_PATCH = 14


def nearest_patch_multiple(size: int, patch: int = _DINOV2_PATCH) -> int:
    """patch must match the deployed backbone's actual patch size (see
    dataset._patch_size_for_backbone) — a mismatch silently produces an out_size the backbone's
    patch_embed rejects (AssertionError), exactly the exp004/DINOv3 bug found 2026-09-08."""
    return max(patch, round(size / patch) * patch)


# Label-prior fallback (mean probability over the full labels_v2_final.parquet, 4407 studies) —
# used ONLY when a study yields literally zero usable series (every series failed to decode), so
# the submission still has a plausible row instead of a crash or a blank one.
LABEL_PRIORS = {
    "ACL": 0.1881, "MCL": 0.0915, "Medial Meniscus": 0.3872, "Lateral Meniscus": 0.2336,
    "Medial OA": 0.2430, "Lateral OA": 0.1457, "PF OA": 0.3909, "Effusion": 0.3826,
    "Synovitis": 0.3894, "Baker's": 0.1912, "Contusion": 0.2086, "Fracture": 0.1321,
}


def normalize_tag_side(v: str | None) -> str | None:
    v = str(v or "").strip().upper()
    if v in ("R", "RIGHT"):
        return "R"
    if v in ("L", "LEFT"):
        return "L"
    if v == "B":
        return "BILATERAL"
    return None


def resolve_study_side(series_sides: list[str | None]) -> str | None:
    """Majority DICOM tag across a study's series. None (ambiguous/missing) triggers the
    dual-flip fallback in run_study_inference — never guessed."""
    values = sorted(set(v for v in series_sides if v in ("R", "L")))
    if len(values) == 1:
        return values[0]
    return None


class WallClockGuard:
    """CLAUDE.md §9: degrade gracefully on a time budget rather than risk a Kaggle timeout."""

    def __init__(self, budget_seconds: float):
        self.t0 = time.time()
        self.budget = budget_seconds

    def elapsed(self) -> float:
        return time.time() - self.t0

    def fraction_used(self) -> float:
        return self.elapsed() / self.budget

    def over(self, margin: float = 0.0) -> bool:
        return self.elapsed() > self.budget * (1 - margin)


def decode_study_slots(
    study_uid: str, series_rows: list[dict], series_root: Path, T: int, out_size: int,
) -> tuple[dict[int, torch.Tensor], list[str]]:
    """series_rows: list of {"series_uid","Anatomical_Plane","Fluid_Sensitive"} for this study.
    Returns {slot_idx: [T,out_size,out_size] float32 in [0,1]} for slots with a successfully
    decoded series (longest series wins ties, matching training's RSNAKneeDataset.slot_map), plus
    the raw per-series tag_laterality strings actually decoded (for resolve_study_side)."""
    candidates: dict[int, tuple[str, int]] = {}  # slot_idx -> (series_uid, n_slices)
    decoded: dict[str, tuple[np.ndarray, str]] = {}  # series_uid -> (uint8 vol, tag_side)

    for row in series_rows:
        idx = slot_index(row["Anatomical_Plane"], row["Fluid_Sensitive"])
        if idx is None:
            continue
        series_uid = row["series_uid"]
        try:
            vol, meta = decode_series_to_uint8(series_root / series_uid)
        except Exception as e:  # noqa: BLE001 - one bad series must never kill the study
            print(f"  [WARN] {study_uid}/{series_uid} decode failed: {type(e).__name__}: {e}")
            continue
        decoded[series_uid] = (vol, normalize_tag_side(meta.get("tag_laterality")))
        n_slices = vol.shape[0]
        if idx not in candidates or n_slices > candidates[idx][1]:
            candidates[idx] = (series_uid, n_slices)

    slots: dict[int, torch.Tensor] = {}
    for idx, (series_uid, _) in candidates.items():
        vol, _ = decoded[series_uid]
        n = vol.shape[0]
        take = np.clip(np.linspace(0, n - 1, T).round().astype(int), 0, n - 1)
        sampled = vol[take].astype(np.float32) / 255.0  # [T,384,384]
        t = torch.from_numpy(sampled).unsqueeze(1)  # [T,1,384,384]
        t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
        slots[idx] = t.squeeze(1)  # [T,out_size,out_size]

    tag_sides = [side for _, side in decoded.values()]
    return slots, tag_sides


def build_volumes(slots: dict[int, torch.Tensor], T: int, out_size: int, flip: bool) -> tuple[torch.Tensor, torch.Tensor]:
    volumes = torch.zeros(NUM_SLOTS, T, out_size, out_size, dtype=torch.float32)
    mask = torch.zeros(NUM_SLOTS, dtype=torch.bool)
    for idx, t in slots.items():
        volumes[idx] = torch.flip(t, dims=[-1]) if flip else t
        mask[idx] = True
    return volumes, mask


@torch.no_grad()
def _run_orientation(
    slots: dict[int, torch.Tensor], flip: bool, models: list[torch.nn.Module],
    device: str, T: int, out_size: int, autocast_dtype=torch.bfloat16,
) -> np.ndarray:
    """One orientation, fold-ensembled — no label swap applied (caller's responsibility, since
    whether a swap is needed depends on WHY this orientation was chosen, not just that flip=True)."""
    volumes, mask = build_volumes(slots, T, out_size, flip)
    volumes, mask = volumes.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
    fold_probs = []
    for model in models:
        # autocast only on CUDA — CPU autocast+bf16 hits dtype-mismatch errors in some ops
        # (MultiheadAttention) that never surface on the GPU path Kaggle actually runs on; the
        # CPU path only exists for local/no-GPU rehearsal, where plain fp32 is fine.
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(volumes, mask)
        else:
            logits = model(volumes, mask)
        fold_probs.append(torch.sigmoid(logits).float().cpu().numpy()[0])
    return np.mean(fold_probs, axis=0)


def _swap_paired_labels(probs: np.ndarray) -> np.ndarray:
    probs = probs.copy()
    for i, j in LATERALITY_SWAP_PAIRS:
        probs[i], probs[j] = probs[j], probs[i]
    return probs


def run_study_inference(
    slots: dict[int, torch.Tensor], tag_sides: list[str | None], models: list[torch.nn.Module],
    device: str, T: int, out_size: int, autocast_dtype=torch.bfloat16,
) -> np.ndarray:
    """Returns a [12] probability vector, ensembled over all folds and (if side is unresolved)
    over both orientations. side is RIGHT -> flip (mirror to canonical left-side convention,
    matching training); side LEFT or unresolved-default-pass -> no flip."""
    side = resolve_study_side(tag_sides)
    orientations = [side == "R"] if side is not None else [False, True]  # unresolved: both

    all_probs = []
    for flip in orientations:
        probs = _run_orientation(slots, flip, models, device, T, out_size, autocast_dtype)
        if side is None and flip:
            # This pass was mirrored relative to the "canonical" no-flip pass above, so swap the
            # paired medial/lateral targets back before averaging the two orientations together
            # (exact analogue of the training-time laterality_swap_prob label swap).
            probs = _swap_paired_labels(probs)
        all_probs.append(probs)

    return np.mean(all_probs, axis=0)


def run_study_inference_tta(
    slots: dict[int, torch.Tensor], tag_sides: list[str | None], models: list[torch.nn.Module],
    device: str, T: int, out_size: int, autocast_dtype=torch.bfloat16,
) -> np.ndarray:
    """Accuracy-seeking laterality-swap TTA (2026-09-09 instruction, distinct from
    run_study_inference's side-UNKNOWN fallback above): always runs BOTH orientations regardless
    of whether side is resolved, and ALWAYS swaps the mirrored pass's paired labels back (every
    flipped pass needs it here, not just the side-unknown case) before averaging."""
    side = resolve_study_side(tag_sides)
    canonical_flip = side == "R" if side is not None else False
    normal = _run_orientation(slots, canonical_flip, models, device, T, out_size, autocast_dtype)
    mirrored = _swap_paired_labels(
        _run_orientation(slots, not canonical_flip, models, device, T, out_size, autocast_dtype)
    )
    return (normal + mirrored) / 2


def prior_row() -> np.ndarray:
    return np.array([LABEL_PRIORS[c] for c in TARGET_COLUMNS], dtype=np.float32)


def load_ensemble_recipe(path) -> dict:
    """Parses a frozen ensemble_recipe_vN.yaml (see ensemble_recipe_v1.yaml for the format and
    2026-09-10 provenance) into {"members": {name: {backbone, kaggle_dataset, ...}},
    "recipe": {label: (members_list, method)}} — the submission notebook builds its member/recipe
    structures from THIS, not a hardcoded list, so a new recipe version is a new file, not a
    notebook edit."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    members = {m["name"]: m for m in data["members"]}
    recipe = {label: (spec["members"], spec["method"]) for label, spec in data["recipe"].items()}
    return {"members": members, "recipe": recipe, "tta_enabled": data["tta"]["enabled"], "raw": data}


def empirical_percentile(raw_score: float, sorted_reference: np.ndarray) -> float:
    """Converts a single new prediction into the percentile rank it would occupy within a stored
    reference distribution (that member's own OOF scores for this label). "rank-mean" ensembling
    (CLAUDE.md §11) is only defined relative to a population — there's no population of 1 to rank
    against at single-study inference time, so the offline OOF-evaluation notion of "rank" doesn't
    transfer directly to a live prediction. This is the standard fix: treat the stored OOF scores
    as an empirical CDF and look up where the new score falls, exactly the same operation
    `pd.Series.rank(pct=True)` does internally, just against a FIXED reference instead of the
    batch itself."""
    return float(np.searchsorted(sorted_reference, raw_score) / len(sorted_reference))


def ensemble_blend(
    member_probs: dict[str, np.ndarray], recipe: dict[str, tuple[list[str], str]],
    rank_references: dict[str, dict[str, np.ndarray]] | None,
) -> np.ndarray:
    """member_probs: {member_name: [12] probability vector, already fold-ensembled + orientation-
    resolved for that member}. recipe: {label: (members_for_this_label, method)} — for the
    accepted GLOBAL recipe (2026-09-09) every label lists the same 3 members, but this stays
    generic in case a per-label recipe is ever used instead. rank_references: {member_name:
    {label: sorted 1D array of that member's OOF scores for this label}} — required only for
    labels using "rank-mean"; None-safe (falls back to prob-mean with a warning) if missing."""
    out = np.zeros(len(TARGET_COLUMNS), dtype=np.float32)
    for i, label in enumerate(TARGET_COLUMNS):
        members, method = recipe[label]
        if method == "rank-mean" and rank_references is not None:
            vals = [
                empirical_percentile(member_probs[m][i], rank_references[m][label])
                for m in members
            ]
        else:
            vals = [member_probs[m][i] for m in members]
        out[i] = float(np.mean(vals))
    return out
