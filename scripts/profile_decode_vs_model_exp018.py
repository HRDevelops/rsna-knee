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

"""exp018 pre-submission timing check: decode time vs 5-fold model time, measured SEPARATELY on real
studies on H100, so a T4 runtime can be projected the same way as the ensemble/efficiency notebooks
(docs/SUBMISSION_RUNBOOK.md: 3.5x model-time multiplier, decode unchanged — CPU-bound). The Kaggle
commit/dry-run environment always truncates test.csv to 3 placeholder rows (never the full ~1,300-study
hidden set), so this HPC measurement is the only way to check the full run against the wall-clock budget
before spending a submission.

    python scripts/profile_decode_vs_model_exp018.py --n-studies 300
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.wmil import (  # noqa: E402
    NUM_SLOTS, WindowMIL, decode_study, gather_windows, normalize_orientation, prepare_input,
    resolve_slot_sides,
)

EXP = "exp018_coatnet2_wmil_v3labels"
T4_MODEL_MULTIPLIER = 3.5  # docs/SUBMISSION_RUNBOOK.md, established 2026-09-11


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-studies", type=int, default=300)
    ap.add_argument("--decode-workers", type=int, default=8)
    ap.add_argument("--n-test-studies", type=int, default=1300, help="assumed hidden-test size, for the projection")
    ap.add_argument("--budget-hours", type=float, default=8.0)
    args = ap.parse_args()

    root = Path(os.environ["RSNA_ROOT"])
    art = root / "artifacts" / EXP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "n/a"
    print(f"device={device} gpu={gpu_name}", file=sys.stderr)

    raw = root / "data/raw"
    sdf_all = __import__("pandas").read_csv(raw / "train_series.csv").rename(columns={
        "StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid",
        "Anatomical_Plane": "plane", "Fluid_Sensitive": "fluid"})
    val = __import__("pandas").read_parquet(art / "fold0/oof.parquet")
    uids = list(val["study_uid"].iloc[:args.n_studies])
    print(f"{len(uids)} real fold-0 validation studies (never used for exp018 training or selection)", file=sys.stderr)

    models = []
    for k in range(5):
        p = art / f"fold{k}/best.pt"
        if not p.exists():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = WindowMIL(ck["arch"], pretrained=False, grad_checkpointing=False)
        m.load_state_dict(ck["model"], strict=True)
        models.append(m.to(device).eval())
    res = int(ck["res"])
    print(f"{len(models)} fold model(s) loaded, res={res}", file=sys.stderr)

    def decode_one(uid: str):
        rows = sdf_all[sdf_all["study_uid"] == uid][["series_uid", "plane", "fluid"]].to_dict("records")
        stack, present, metas = decode_study(raw / "train_series" / uid, rows)
        sides = resolve_slot_sides(metas, None)
        return uid, normalize_orientation(stack, metas, sides), present

    t0 = time.time()
    decoded = {}
    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        futs = {pool.submit(decode_one, u): u for u in uids}
        for fut in as_completed(futs):
            uid, stack, present = fut.result()
            decoded[uid] = (stack, present)
    decode_elapsed = time.time() - t0
    print(f"decode: {len(decoded)}/{len(uids)} ok, {decode_elapsed:.1f}s total, "
          f"{decode_elapsed / len(uids):.3f} s/study", file=sys.stderr)

    from rsna_knee.wmil import candidate_centers, predict_study
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    n_windows = []
    for uid in uids:
        stack, present = decoded[uid]
        st = torch.from_numpy(stack).to(device)
        n_windows.append(len(candidate_centers(present)))
        for m in models:
            logit = predict_study(m, st, present, res, torch.float16 if device == "cuda" else torch.bfloat16)
            if not torch.isfinite(logit).all():
                predict_study(m, st, present, res, torch.float32)
        del st
    torch.cuda.synchronize() if device == "cuda" else None
    model_elapsed = time.time() - t0
    print(f"model (all {len(models)} folds): {model_elapsed:.1f}s total, {model_elapsed / len(uids):.3f} s/study "
          f"({model_elapsed / len(uids) / len(models):.3f} s/study/fold); mean windows/study "
          f"{np.mean(n_windows):.1f}", file=sys.stderr)

    decode_per = decode_elapsed / len(uids)
    model_per_h100 = model_elapsed / len(uids)
    model_per_t4 = model_per_h100 * T4_MODEL_MULTIPLIER
    n_test = args.n_test_studies
    budget_s = args.budget_hours * 3600

    print(f"\n=== H100 measured ({len(uids)} real studies, {len(models)}-fold ensemble) ===")
    print(f"  decode: {decode_per:.3f} s/study")
    print(f"  model : {model_per_h100:.3f} s/study")
    print(f"  total : {decode_per + model_per_h100:.3f} s/study -> {(decode_per + model_per_h100) * n_test / 60:.1f} min for {n_test} studies")

    print(f"\n=== T4 projection (decode unchanged, model x{T4_MODEL_MULTIPLIER}; T4 x1 = 1 decode-consumer thread) ===")
    total_t4x1 = (decode_per + model_per_t4) * n_test
    print(f"  T4 x1: {decode_per + model_per_t4:.3f} s/study -> {total_t4x1 / 60:.1f} min for {n_test} studies "
          f"({'FITS' if total_t4x1 <= budget_s * 0.85 else 'EXCEEDS'} the {args.budget_hours:.0f}h x0.85 safety margin)")
    # T4 x2: infer_wmil.py is DATA-parallel across devices (each GPU independently consumes whole studies
    # from the shared decode queue), so throughput roughly doubles on the GPU side; decode is still shared
    # across all decode-worker threads regardless of GPU count, so it's the floor on how much x2 can help.
    per_study_2gpu = max(decode_per, (decode_per + model_per_t4) / 2)
    total_t4x2 = per_study_2gpu * n_test
    print(f"  T4 x2 (data-parallel, decode-floor projection): {per_study_2gpu:.3f} s/study -> "
          f"{total_t4x2 / 60:.1f} min for {n_test} studies "
          f"({'FITS' if total_t4x2 <= budget_s * 0.85 else 'EXCEEDS'} the {args.budget_hours:.0f}h x0.85 safety margin)")
    print(f"\nCaveat: the 3.5x multiplier is carried over from the ensemble/efficiency notebooks' DINOv2 ViT "
          f"profiling (2026-09-11) and has not been separately validated for CoAtNet-2's operator mix — "
          f"treat as a plausible range, not a guarantee, same as those notebooks' own documented caveat.")


if __name__ == "__main__":
    main()
