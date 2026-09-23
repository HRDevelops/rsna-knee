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

"""Probe #4 inference driver: exp018 (5-fold Window-MIL) blended with the deployed 3-member ensemble
in RANK space, at a fixed weight (default 0.75 exp018 / 0.25 ensemble — 2026-09-22 instruction: not
picked by OOF, see docs/EXPERIMENTS.md and artifacts/exp018_probe4_analysis.md for why).

Each study is decoded TWICE (the two pipelines use different physical crops/spacing/slot layouts —
115mm/384px/T=24/6-slot for the deployed ensemble, 140mm/336px/64-slice/5-slot for exp018 — there is
no shared intermediate representation to reuse between them). Both pipelines' predictions are
converted to a percentile against their own stored OOF reference distribution
(`infer_core.empirical_percentile`, the same empirical-CDF mechanism the 3-member ensemble already
uses internally for its rank-mean labels) before blending, so "rank-mean" is well-defined for a
single live study exactly as it already is for the deployed notebook.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna_knee.dataset import TARGET_COLUMNS, _patch_size_for_backbone
from rsna_knee.infer_core import (
    decode_one_for_pool, empirical_percentile, ensemble_blend, load_ensemble_recipe, nearest_patch_multiple,
    prior_row, run_study_inference,
)
from rsna_knee.model import SlotModel
from rsna_knee.wmil import (
    WindowMIL, decode_study, normalize_orientation, predict_study, resolve_slot_sides,
)

OLD_T, OLD_IMG = 24, 384


def _load_old_members(recipe: dict, weights_root: Path, devices: list[str]) -> dict:
    """{device: {member_name: [5 fold SlotModel]}}."""
    out: dict[str, dict[str, list]] = {d: {} for d in devices}
    for name, spec in recipe["members"].items():
        backbone = spec["backbone"]
        ckpt_dir = weights_root / spec["kaggle_dataset"]
        fold_paths = sorted(ckpt_dir.glob("fold*_best.pt")) or [
            d / "best.pt" for d in sorted(ckpt_dir.glob("fold*")) if (d / "best.pt").exists()]
        for d in devices:
            models = []
            for p in fold_paths:
                ck = torch.load(p, map_location=d, weights_only=False)
                m = SlotModel(backbone, num_targets=len(TARGET_COLUMNS), pretrained=False).to(d)
                m.load_state_dict(ck["ema"])
                m.eval()
                models.append(m)
            out[d][name] = models
    return out


def _load_exp018(weight_paths: list[Path], devices: list[str]) -> tuple[dict, int]:
    out: dict[str, list] = {d: [] for d in devices}
    for p in weight_paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        for d in devices:
            m = WindowMIL(ck["arch"], pretrained=False, grad_checkpointing=False)
            m.load_state_dict(ck["model"], strict=True)
            out[d].append(m.to(d).eval())
    return out, int(ck["res"])


def _decode_task(task: tuple) -> tuple:
    """Picklable top-level. Decodes BOTH pipelines for one study. Returns
    (uid, old_slots|None, old_tag_sides|None, new_stack|None, new_present|None, error|None)."""
    uid, series_rows, old_series_root, old_out_size, new_study_dir = task
    # decode_one_for_pool does the study_uid subdirectory join itself (series_root/study_uid/series_uid),
    # matching the deployed ensemble/efficiency notebooks exactly -- calling decode_study_slots directly
    # here (an earlier version of this function) skipped that join and looked for series straight under
    # series_root, which only ever finds "no .dcm files in series dir".
    _uid2, old_slots, old_tags, old_err = decode_one_for_pool((uid, series_rows, old_series_root, OLD_T, old_out_size))
    if old_err is not None or not old_slots:
        return uid, None, None, None, None, f"old-pipeline decode: {old_err or 'no slots decoded'}"
    try:
        new_stack, new_present, new_metas = decode_study(Path(new_study_dir), series_rows)
        new_sides = resolve_slot_sides(new_metas, None)
        new_stack = normalize_orientation(new_stack, new_metas, new_sides) if new_present.any() else new_stack
    except Exception as e:  # noqa: BLE001
        return uid, old_slots, old_tags, None, None, f"exp018-pipeline decode: {type(e).__name__}: {e}"
    return uid, old_slots, old_tags, new_stack, new_present, None


def run_blend018(
    test_csv: Path, test_series_csv: Path, old_series_root: Path, new_series_root: Path,
    recipe_path: Path, old_weights_root: Path, exp018_weight_paths: list[Path],
    exp018_rank_ref: Path, ensemble_output_rank_ref: Path, out_csv: Path,
    weight_exp018: float = 0.75, devices: list[str] | None = None, num_decode_workers: int = 3,
    budget_seconds: float = 8.0 * 3600, limit: int = 0, log=print,
) -> pd.DataFrame:
    t0 = time.time()
    devices = devices or (["cuda:%d" % i for i in range(torch.cuda.device_count())] or ["cpu"])
    dtype = torch.float16 if devices[0] != "cpu" else torch.bfloat16
    recipe = load_ensemble_recipe(recipe_path)
    old_backbone0 = next(iter(recipe["members"].values()))["backbone"]
    old_out_size = nearest_patch_multiple(OLD_IMG, _patch_size_for_backbone(old_backbone0))

    old_models = _load_old_members(recipe, old_weights_root, devices)
    old_refs = {name: {t: np.load(old_weights_root / spec["kaggle_dataset"] / "rank_reference.npz")[t]
                       for t in TARGET_COLUMNS}
                for name, spec in recipe["members"].items()}
    exp018_models, res018 = _load_exp018(exp018_weight_paths, devices)
    exp018_ref = np.load(exp018_rank_ref)
    ens_out_ref = np.load(ensemble_output_rank_ref)
    log(f"[blend018] loaded: {len(recipe['members'])} old member(s) x5 folds + {len(exp018_weight_paths)} "
        f"exp018 fold(s), x{len(devices)} device(s): {devices}, weight(exp018)={weight_exp018}, "
        f"loaded in {time.time() - t0:.0f}s")

    test = pd.read_csv(test_csv)
    # Both pipelines read the same series table but expect different key names: the deployed
    # ensemble's decode_study_slots wants the raw Anatomical_Plane/Fluid_Sensitive columns,
    # exp018's decode_study wants plane/fluid (rsna_knee.wmil.pick_slots) -- keep BOTH so one
    # series_rows record satisfies either pipeline.
    sdf = pd.read_csv(test_series_csv).rename(columns={
        "StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid"})
    sdf["plane"] = sdf["Anatomical_Plane"]
    sdf["fluid"] = sdf["Fluid_Sensitive"]
    studies = list(test["StudyInstanceUID"])
    if limit:
        studies = studies[:limit]
    tasks = [(u, sdf[sdf["study_uid"] == u].to_dict("records"), str(old_series_root),
              old_out_size, str(new_series_root / u)) for u in studies]

    results: dict[str, np.ndarray] = {}
    q: queue.Queue = queue.Queue(maxsize=6)
    errors: list[str] = []

    def gpu_worker(dev: str) -> None:
        while True:
            item = q.get()
            if item is None:
                return
            uid, old_slots, old_tags, new_stack, new_present, err = item
            try:
                if err is not None or old_slots is None or new_stack is None:
                    raise RuntimeError(err or "one pipeline produced nothing for this study")
                member_probs = {name: run_study_inference(old_slots, old_tags, old_models[dev][name], dev,
                                                           OLD_T, old_out_size, dtype)
                                for name in recipe["members"]}
                ens_out = ensemble_blend(member_probs, recipe["recipe"], old_refs)
                ens_pct = np.array([empirical_percentile(ens_out[i], ens_out_ref[c])
                                    for i, c in enumerate(TARGET_COLUMNS)])

                st = torch.from_numpy(new_stack).to(dev)
                probs018 = []
                for m in exp018_models[dev]:
                    logit = predict_study(m, st, new_present, res018, dtype)
                    if not torch.isfinite(logit).all():
                        logit = predict_study(m, st, new_present, res018, torch.float32)
                    probs018.append(torch.sigmoid(logit.float()).cpu().numpy())
                p018 = np.mean(probs018, axis=0)
                pct018 = np.array([empirical_percentile(p018[i], exp018_ref[c]) for i, c in enumerate(TARGET_COLUMNS)])
                del st

                results[uid] = weight_exp018 * pct018 + (1 - weight_exp018) * ens_pct
            except Exception as e:  # noqa: BLE001
                errors.append(f"{uid}: {type(e).__name__}: {e}")

    workers = [threading.Thread(target=gpu_worker, args=(d,), daemon=True) for d in devices]
    for w in workers:
        w.start()
    # Exit-safety (2026-09-23): see infer_wmil.run_exp018's identical fix and infer_core.prefetch_decode's
    # docstring for the mechanism -- an exception anywhere in the decode loop below must not skip the
    # sentinel+join code (it was a sequence of plain statements after `with pool:`, not a finally), and the
    # pool must be shut down with cancel_futures=True so we don't wait on decode tasks we're abandoning.
    pool = ThreadPoolExecutor(max_workers=num_decode_workers)
    try:
        inflight: deque = deque()
        it = iter(tasks)
        n_done = 0
        while True:
            while len(inflight) < num_decode_workers + 4:
                t = next(it, None)
                if t is None:
                    break
                inflight.append(pool.submit(_decode_task, t))
            if not inflight:
                break
            res = inflight.popleft().result()
            q.put(res)
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                if elapsed / n_done * len(studies) > budget_seconds:
                    log(f"[blend018] WARNING: projected total {elapsed / n_done * len(studies) / 60:.0f} min "
                        f"exceeds the budget at study {n_done}/{len(studies)} — no automatic degrade for this "
                        f"combined pipeline; consider a smaller fold count if this recurs")
                log(f"[blend018] {n_done}/{len(studies)} decoded, {len(results)} predicted, {elapsed / 60:.1f} min")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for _ in workers:
            q.put(None)
        for w in workers:
            w.join(timeout=30)

    prior = prior_row()
    rows = []
    for u in test["StudyInstanceUID"]:
        p = results.get(u, prior)
        rows.append({"StudyInstanceUID": u, **{c: float(v) for c, v in zip(TARGET_COLUMNS, p)}})
    sub = pd.DataFrame(rows)
    sub.to_csv(out_csv, index=False)
    log(f"[blend018] wrote {out_csv}: {len(sub)} rows, {len(results)} predicted, {len(sub) - len(results)} on prior, "
        f"{len(errors)} errors, {(time.time() - t0) / 60:.1f} min total")
    for e in errors[:10]:
        log(f"  [error] {e}")
    return sub
