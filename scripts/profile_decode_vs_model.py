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

"""One-off profiling: how much of the efficiency notebook's per-study cost (CLAUDE.md §8) is
DICOM decode vs. model inference, at T=16/320px (exp008's config). Informs the 2026-09-10
"pre-push fixes" instruction: pick the fold-count ladder rung from the efficiency-track FORMULA,
and add a decode fast-path if decode turns out to dominate.

Usage:
    python scripts/profile_decode_vs_model.py \
        --weights-dir $RSNA_ROOT/artifacts/exp008_dinov2s_slot_distilled \
        --raw $RSNA_ROOT/artifacts/synthetic_loadtest --n-studies 30
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.dataset import TARGET_COLUMNS, _patch_size_for_backbone  # noqa: E402
from rsna_knee.infer_core import decode_one_for_pool, nearest_patch_multiple, run_study_inference  # noqa: E402
from rsna_knee.model import SlotModel  # noqa: E402

BACKBONE = "vit_small_patch14_dinov2.lvd142m"
SLICES_PER_SERIES = 16
IMAGE_SIZE = 320


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--n-studies", type=int, default=30)
    parser.add_argument("--decode-workers", type=int, default=15)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_size = nearest_patch_multiple(IMAGE_SIZE, _patch_size_for_backbone(BACKBONE))
    print(f"device={device} out_size={out_size} gpu={torch.cuda.get_device_name(0) if device == 'cuda' else 'n/a'}",
          file=sys.stderr)

    models = []
    for fold_dir in sorted(args.weights_dir.glob("fold*")):
        ckpt = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
        m = SlotModel(BACKBONE, num_targets=len(TARGET_COLUMNS), pretrained=False).to(device)
        m.load_state_dict(ckpt["ema"])
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)}/5 folds.", file=sys.stderr)

    test_df = pd.read_csv(args.raw / "test.csv").iloc[:args.n_studies].reset_index(drop=True)
    series_df = pd.read_csv(args.raw / "test_series.csv").rename(
        columns={"StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid"}
    )
    series_root = args.raw / "test_series"
    uids = list(test_df["StudyInstanceUID"])

    tasks = [
        (uid, series_df[series_df["study_uid"] == uid].to_dict("records"), str(series_root),
         SLICES_PER_SERIES, out_size)
        for uid in uids
    ]

    t0 = time.time()
    decoded = {}
    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        futures = {pool.submit(decode_one_for_pool, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            uid, slots, tag_sides, err = fut.result()
            decoded[uid] = (slots, tag_sides, err)
    decode_elapsed = time.time() - t0

    n_ok = sum(1 for _, _, err in decoded.values() if err is None)
    print(f"Decode: {n_ok}/{len(uids)} ok, {decode_elapsed:.1f}s total, "
          f"{decode_elapsed / len(uids):.3f}s/study (wall, {args.decode_workers} threads)", file=sys.stderr)

    t0 = time.time()
    for uid in uids:
        slots, tag_sides, err = decoded[uid]
        if slots and err is None:
            run_study_inference(slots, tag_sides, models, device, SLICES_PER_SERIES, out_size, torch.float16)
    model_elapsed = time.time() - t0
    print(f"Model (5-fold): {model_elapsed:.1f}s total, {model_elapsed / len(uids):.3f}s/study", file=sys.stderr)

    total = decode_elapsed + model_elapsed
    print(f"\nSplit: decode {decode_elapsed / total:.1%}, model(5-fold) {model_elapsed / total:.1%} "
          f"(of {total:.1f}s combined, sequential-equivalent — decode itself ran threaded)", file=sys.stderr)
    print(f"Per-study combined: {total / len(uids):.3f}s/study at 5 folds", file=sys.stderr)


if __name__ == "__main__":
    main()
