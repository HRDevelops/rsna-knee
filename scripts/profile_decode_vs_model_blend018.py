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

"""Probe #4 blend: decode time vs model time, measured SEPARATELY for BOTH pipelines on real studies
on H100, so a T4 runtime can be projected the same way as every other notebook in this project
(docs/SUBMISSION_RUNBOOK.md: model time x3.5, decode unchanged/CPU-bound, decode-worker-count rescaled
to Kaggle's real allocation). A naive single combined wall-clock number (as in rehearse_blend018.py)
cannot be split into "what gets 3.5x" and "what doesn't" -- this script decodes and predicts each
pipeline's stage separately to get that split honestly.

    python scripts/profile_decode_vs_model_blend018.py --n-studies 150
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.dataset import TARGET_COLUMNS, _patch_size_for_backbone  # noqa: E402
from rsna_knee.infer_core import decode_one_for_pool, nearest_patch_multiple, run_study_inference  # noqa: E402
from rsna_knee.model import SlotModel  # noqa: E402
from rsna_knee.wmil import WindowMIL, candidate_centers, decode_study, normalize_orientation, predict_study, resolve_slot_sides  # noqa: E402

OLD_T, OLD_IMG = 24, 384
T4_MODEL_MULTIPLIER = 3.5
PROFILED_DECODE_WORKERS = 8
KAGGLE_DECODE_WORKERS = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-studies", type=int, default=150)
    ap.add_argument("--decode-workers", type=int, default=PROFILED_DECODE_WORKERS)
    ap.add_argument("--n-test-studies", type=int, default=1300)
    ap.add_argument("--budget-hours", type=float, default=8.0)
    a = ap.parse_args()

    root = Path(os.environ["RSNA_ROOT"])
    art = root / "artifacts"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device=='cuda' else 'n/a'}", file=sys.stderr)

    raw = root / "data/raw"
    val = pd.read_parquet(art / "exp018_coatnet2_wmil_v3labels/fold0/oof.parquet")
    uids = list(val["study_uid"].iloc[:a.n_studies])
    sdf = pd.read_csv(raw / "train_series.csv")
    print(f"{len(uids)} real fold-0 validation studies (shared by both pipelines' inputs)", file=sys.stderr)

    recipe = yaml.safe_load((root / "code/src/rsna_knee/ensemble_recipe_v1.yaml").read_text())
    old_backbone0 = next(iter(recipe["members"]))["backbone"]
    old_out_size = nearest_patch_multiple(OLD_IMG, _patch_size_for_backbone(old_backbone0))
    stage = root / "kaggle_staging"
    old_models = {}
    for m in recipe["members"]:
        paths = sorted((stage / m["kaggle_dataset"]).glob("fold*_best.pt"))
        ms = []
        for p in paths:
            ck = torch.load(p, map_location=device, weights_only=False)
            mm = SlotModel(m["backbone"], num_targets=len(TARGET_COLUMNS), pretrained=False).to(device)
            mm.load_state_dict(ck["ema"])
            mm.eval()
            ms.append(mm)
        old_models[m["name"]] = ms
    print(f"old ensemble: {sum(len(v) for v in old_models.values())} model(s) loaded", file=sys.stderr)

    exp018_models = []
    for k in range(5):
        ck = torch.load(art / f"exp018_coatnet2_wmil_v3labels/fold{k}/best.pt", map_location="cpu", weights_only=False)
        m = WindowMIL(ck["arch"], pretrained=False, grad_checkpointing=False)
        m.load_state_dict(ck["model"], strict=True)
        exp018_models.append(m.to(device).eval())
    res018 = int(ck["res"])
    print(f"exp018: {len(exp018_models)} fold model(s) loaded, res={res018}", file=sys.stderr)

    def old_task(uid):
        rows = sdf[sdf["StudyInstanceUID"] == uid].rename(
            columns={"SeriesInstanceUID": "series_uid"}).to_dict("records")
        return decode_one_for_pool((uid, rows, str(raw / "train_series"), OLD_T, old_out_size))

    def new_task(uid):
        rows = sdf[sdf["StudyInstanceUID"] == uid].rename(
            columns={"SeriesInstanceUID": "series_uid", "Anatomical_Plane": "plane",
                     "Fluid_Sensitive": "fluid"}).to_dict("records")
        stack, present, metas = decode_study(raw / "train_series" / uid, rows)
        sides = resolve_slot_sides(metas, None)
        return uid, normalize_orientation(stack, metas, sides), present

    # --- decode, both pipelines, separately ---
    t0 = time.time()
    old_decoded = {}
    with ThreadPoolExecutor(max_workers=a.decode_workers) as pool:
        for fut in as_completed({pool.submit(old_task, u): u for u in uids}):
            uid, slots, tags, err = fut.result()
            if err is not None:
                print(f"  [WARN] {uid}: old-pipeline decode failed: {err}", file=sys.stderr)
            old_decoded[uid] = (slots, tags) if err is None else (None, None)
    old_decode_s = time.time() - t0
    print(f"old-pipeline decode: {old_decode_s:.1f}s total, {old_decode_s/len(uids):.3f} s/study", file=sys.stderr)

    t0 = time.time()
    new_decoded = {}
    with ThreadPoolExecutor(max_workers=a.decode_workers) as pool:
        for fut in as_completed({pool.submit(new_task, u): u for u in uids}):
            uid, stack, present = fut.result()
            new_decoded[uid] = (stack, present)
    new_decode_s = time.time() - t0
    print(f"exp018-pipeline decode: {new_decode_s:.1f}s total, {new_decode_s/len(uids):.3f} s/study", file=sys.stderr)

    # --- model, both pipelines, separately ---
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    for uid in uids:
        slots, tags = old_decoded.get(uid, (None, None))
        if slots is None:
            continue
        for name, ms in old_models.items():
            run_study_inference(slots, tags, ms, device, OLD_T, old_out_size, dtype)
    torch.cuda.synchronize() if device == "cuda" else None
    old_model_s = time.time() - t0
    print(f"old-pipeline model (3 members x5 folds): {old_model_s:.1f}s total, {old_model_s/len(uids):.3f} s/study",
          file=sys.stderr)

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    for uid in uids:
        stack, present = new_decoded.get(uid, (None, None))
        if stack is None:
            continue
        st = torch.from_numpy(stack).to(device)
        for m in exp018_models:
            logit = predict_study(m, st, present, res018, dtype)
            if not torch.isfinite(logit).all():
                predict_study(m, st, present, res018, torch.float32)
        del st
    torch.cuda.synchronize() if device == "cuda" else None
    new_model_s = time.time() - t0
    print(f"exp018-pipeline model (5 folds): {new_model_s:.1f}s total, {new_model_s/len(uids):.3f} s/study", file=sys.stderr)

    decode_h100 = (old_decode_s + new_decode_s) / len(uids)
    decode_kaggle = decode_h100 * (a.decode_workers / KAGGLE_DECODE_WORKERS)
    model_h100 = (old_model_s + new_model_s) / len(uids)
    model_t4 = model_h100 * T4_MODEL_MULTIPLIER
    n_test, budget_s = a.n_test_studies, a.budget_hours * 3600

    print(f"\n=== H100 measured ({len(uids)} real studies, combined pipeline) ===")
    print(f"  decode (both pipelines): {decode_h100:.3f} s/study")
    print(f"  model  (both pipelines): {model_h100:.3f} s/study")
    print(f"  raw combined wall (for comparison with rehearse_blend018.py): {decode_h100 + model_h100:.3f} s/study")

    print(f"\n=== T4 projection (decode rescaled {a.decode_workers}->{KAGGLE_DECODE_WORKERS} workers, model x{T4_MODEL_MULTIPLIER}) ===")
    for ngpu in (1, 2):
        per_study = max(decode_kaggle, model_t4 / ngpu)
        total = per_study * n_test
        print(f"  T4 x{ngpu}: {per_study:.3f} s/study -> {total/60:.1f} min for {n_test} studies "
              f"({'FITS' if total <= budget_s * 0.85 else 'EXCEEDS'} the {a.budget_hours:.0f}h x0.85 margin)")


if __name__ == "__main__":
    main()
