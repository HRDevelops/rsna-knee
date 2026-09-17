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

"""DEAD — exp011 (MRNet auxiliary pretraining) DROPPED, 2026-09-17. Do not run this script or
resurrect this plan without re-confirming the license problem below is actually resolved (it
wasn't — this was dropped specifically because of it, not superseded by something better).
kept in git for the record only (the license research below took real effort to establish, and
KneeMRI is now being evaluated as a licensing-clean replacement — see the corresponding
docs/EXPERIMENTS.md entry).

--- ORIGINAL STATUS, PREP ONLY — not run yet, exp011 (MRNet auxiliary pretraining), 2026-09-17.

============================================================================
LICENSE STATUS: NOT CONFIRMED ELIGIBLE. Do not download or use MRNet data
under this project until this is resolved by the user directly (see below).
============================================================================

Checked 2026-09-17 (stanfordmlgroup.github.io/competitions/mrnet/ and search results for the
dataset's terms of use):
  "Permission is granted to view and use the MRNet Dataset without charge for PERSONAL,
   NON-COMMERCIAL RESEARCH PURPOSES ONLY. Any commercial use, sale, or other monetization is
   prohibited... By registering for downloads... you are agreeing to a Research Use Agreement...
   you may NOT share the link to download the dataset with others."

Two separate problems this raises for THIS project specifically, neither resolved by this script:

1. Non-commercial-only license vs. a cash-prize Kaggle competition. Whether training on a
   non-commercial-research-only dataset and then competing for prize money counts as violating
   that restriction is a real legal question, not a technical one — this script does not resolve
   it, and neither should an LLM. The user needs to make this call, possibly after reading the
   actual Research Use Agreement text in full (gate is behind Stanford AIMI/Redivis registration,
   not reproduced here) and/or the RSNA competition's own external-data rules.

2. "May not share the download link" + "personal use only" is in tension with CLAUDE.md's
   winners'-obligation requirement to PUBLICLY release weights. If an MRNet-pretrained encoder's
   weights get baked into a publicly-released final model, that is arguably redistributing a
   derivative of gated, non-shareable data. EXECUTION_PLAN.md Phase 4.5's own bar ("license =
   'freely & publicly available' per rules") is plausibly NOT met by a dataset that requires
   signing a per-person Research Use Agreement before download, regardless of the non-commercial
   question above.

RECOMMENDATION (not a decision this script makes): do not download MRNet under this project's
Kaggle account/identity until the user has personally confirmed both (1) against the actual RSNA
competition rules' external-data clause and (2) against the real MRNet Research Use Agreement text.
This script only prepares the CODE PATH so that IF eligibility is confirmed, prep is a config change
and a `python scripts/prep_mrnet.py --mrnet-root <path>` away, not a redesign.

============================================================================
What this script does once (if) MRNet is locally available and eligibility is confirmed:
============================================================================

MRNet ships pre-extracted per-exam numpy stacks (NOT raw DICOM), one file per exam per plane
(axial/coronal/sagittal, single proton-density-weighted-ish sequence per plane — no
Fluid_Sensitive/Fat_Suppression split like our 6-slot scheme), each [n_slices, 256, 256] uint8 (per
the original MRNet paper). Labels: `train-{abnormal,acl,meniscus}.csv`, one binary label per exam
per file, 1,130 train / 120 valid exams (public splits; a 120-exam test set is hidden).

Slot mapping (best-effort, since MRNet has no FS/nonFS distinction): each MRNet plane maps to BOTH
of our corresponding FS/nonFS slots (duplicate the same volume into both) rather than leaving one
half of each plane pair empty — an approximation, not a claim that MRNet's sequence is truly
equivalent to either. Document this explicitly if the pretraining pass is ever actually run.

Pretraining task: reuse `rsna_knee.model.SlotModel` with `num_targets=3` (abnormal, ACL, meniscus)
instead of 12, train on MRNet's 1,130/120 split, then load the resulting backbone (not the 3-target
head) as an initialization for a normal 12-target exp011 config on our own data — a plain
`state_dict` filter (drop `study_head.*` / final classifier keys, keep everything upstream of it).

Usage (once eligibility is confirmed and MRNET_ROOT points at a local extraction):
    python scripts/prep_mrnet.py --mrnet-root /path/to/MRNet-v1.0 --out artifacts/mrnet_cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MRNET_PLANES = ["axial", "coronal", "sagittal"]
MRNET_LABEL_TASKS = ["abnormal", "acl", "meniscus"]

# Our slot indices these planes fill BOTH sides of (FS, nonFS) — see slot_index() in dataset.py.
# MRNet has no fluid-sensitive/non-fluid-sensitive distinction; duplicating into both halves of the
# pair is the least-wrong approximation available, not a validated equivalence.
PLANE_TO_SLOT_PAIR = {
    "sagittal": (0, 1),   # Sagittal_FS, Sagittal_nonFS
    "coronal": (2, 3),    # Coronal_FS, Coronal_nonFS
    "axial": (4, 5),       # Axial_FS, Axial_nonFS
}


def _load_split_labels(mrnet_root: Path, split: str) -> pd.DataFrame:
    """MRNet ships one CSV per task per split, header-less, columns [exam_id, label]."""
    frames = []
    for task in MRNET_LABEL_TASKS:
        csv_path = mrnet_root / f"{split}-{task}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found — expected MRNet's standard release layout "
                f"(train-abnormal.csv, train-acl.csv, train-meniscus.csv, and *-valid-* equivalents)."
            )
        df = pd.read_csv(csv_path, header=None, names=["exam_id", task])
        df["exam_id"] = df["exam_id"].astype(str).str.zfill(4)  # MRNet ids are zero-padded
        frames.append(df.set_index("exam_id"))
    merged = pd.concat(frames, axis=1).reset_index()
    return merged


def build_mrnet_cache(mrnet_root: Path, split: str, out_dir: Path) -> pd.DataFrame:
    """Converts one MRNet split into this project's per-study/per-slot .npy cache layout
    (data/cache/<study_uid>/<series_uid>.npy equivalent) so RSNAKneeDataset can read it with no
    code changes beyond a labels/folds parquet pointing at these paths. study_uid is prefixed
    `mrnet_<split>_` to guarantee no collision with real RSNA StudyInstanceUIDs.

    NOT RUN — see module docstring. Calling this before eligibility is confirmed is a bug, not a
    feature; there is no guard here preventing it because the intent is for this to only be
    invoked manually after that decision is made, not accidentally by another script.
    """
    labels_df = _load_split_labels(mrnet_root, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for exam_id in labels_df["exam_id"]:
        study_uid = f"mrnet_{split}_{exam_id}"
        study_dir = out_dir / study_uid
        study_dir.mkdir(exist_ok=True)
        for plane in MRNET_PLANES:
            src = mrnet_root / split / plane / f"{exam_id}.npy"
            if not src.exists():
                print(f"  [WARN] missing {src}, skipping this plane for {study_uid}", file=sys.stderr)
                continue
            vol = np.load(src)  # [n_slices, 256, 256] per the MRNet release format
            # Cast/resize to match this project's cache dtype convention (uint8, no spatial
            # resample here — MRNet's native 256px is smaller than our 384px cache target; an
            # upsample at this stage would be exactly the invalid-resolution-test problem flagged
            # in exp010's config comment, so deliberately left at native resolution and only
            # resized at model-input time via the same F.interpolate path everything else uses).
            vol_u8 = vol.astype(np.uint8) if vol.dtype != np.uint8 else vol
            series_uid = f"{study_uid}_{plane}"
            np.save(study_dir / f"{series_uid}.npy", vol_u8)
            rows.append({"study_uid": study_uid, "series_uid": series_uid, "plane": plane,
                         "n_slices": vol_u8.shape[0]})
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(out_dir / f"{split}_manifest.parquet", index=False)
    labels_out = labels_df.copy()
    labels_out["study_uid"] = "mrnet_" + split + "_" + labels_out["exam_id"]
    labels_out.to_parquet(out_dir / f"{split}_labels.parquet", index=False)
    print(f"{split}: {len(labels_df)} exams, {len(manifest)} series written to {out_dir}")
    return manifest


def main() -> None:
    raise SystemExit(
        "exp011/MRNet was DROPPED 2026-09-17 (license: personal/non-commercial-research-only, "
        "no-link-sharing Research Use Agreement -- incompatible with prize eligibility and the "
        "winners'-obligation public-weights requirement). This script is kept for the record, "
        "not for use. See docs/EXPERIMENTS.md's 2026-09-17 entries."
    )
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mrnet-root", type=Path, required=True,
                         help="Local path to an extracted MRNet-v1.0 release (train/, valid/, train-*.csv)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/mrnet_cache"))
    args = parser.parse_args()

    print(
        "REMINDER: this script assumes MRNet eligibility has already been personally confirmed "
        "by the user against the RSNA competition's external-data rules AND the MRNet Research "
        "Use Agreement — see this file's module docstring. It does not check this for you.",
        file=sys.stderr,
    )
    for split in ("train", "valid"):
        build_mrnet_cache(args.mrnet_root, split, args.out)


if __name__ == "__main__":
    main()
