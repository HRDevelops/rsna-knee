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

"""exp018 inference driver (Kaggle notebook and HPC rehearsal): DICOM study -> exp018 fold ensemble -> 12
probabilities. Uses exactly the training-time preprocessing (rsna_knee.wmil.decode_study /
normalize_orientation), so there is a single implementation of "what the model sees".

Memory/time safety (2026-09-14 lesson: an unbounded decode queue OOM'd the ensemble notebook): decoded
studies (7.2 MB each) flow through a BOUNDED queue (max 6 in flight) from a decode thread pool to one worker thread
per GPU; a failed decode/predict falls back to the label prior, never crashes; a wall-clock guard drops later
folds (never studies) if the projected runtime would overrun.
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

from rsna_knee.wmil import (
    DEPTH, IMG, NUM_SLOTS, TARGET_COLUMNS, WindowMIL, candidate_centers, decode_study, normalize_orientation,
    predict_study, resolve_slot_sides,
)

# mean probability over labels_v2_final (deployed pipeline's fallback for an undecodable study)
LABEL_PRIORS = {
    "ACL": 0.1881, "MCL": 0.0915, "Medial Meniscus": 0.3872, "Lateral Meniscus": 0.2336,
    "Medial OA": 0.2430, "Lateral OA": 0.1457, "PF OA": 0.3909, "Effusion": 0.3826,
    "Synovitis": 0.3894, "Baker's": 0.1912, "Contusion": 0.2086, "Fracture": 0.1321,
}


def _decode_task(task: tuple) -> tuple:
    """Top-level (picklable). task = (study_uid, series_rows, study_dir_str). Returns
    (uid, oriented uint8 stack [DEPTH,IMG,IMG] | None, present bool[5] | None, error | None)."""
    uid, rows, study_dir = task
    try:
        stack, present, metas = decode_study(Path(study_dir), rows)
        if not present.any():
            return uid, None, None, "no slot decoded"
        sides = resolve_slot_sides(metas, None)
        return uid, normalize_orientation(stack, metas, sides), present, None
    except Exception as e:  # noqa: BLE001
        return uid, None, None, f"{type(e).__name__}: {e}"


def load_fold_models(weight_paths: list[Path], devices: list[str], pretrained: bool = False) -> dict:
    """{device: [WindowMIL per fold]} — one full replica set per device."""
    models: dict[str, list] = {d: [] for d in devices}
    for p in weight_paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        for d in devices:
            m = WindowMIL(ck["arch"], pretrained=pretrained, grad_checkpointing=False)
            m.load_state_dict(ck["model"], strict=True)
            models[d].append(m.to(d).eval())
    res = int(ck["res"])
    return {"models": models, "res": res}


def run_exp018(
    test_csv: Path, test_series_csv: Path, series_root: Path, weight_paths: list[Path], out_csv: Path,
    devices: list[str] | None = None, num_decode_workers: int = 4, budget_seconds: float = 8.0 * 3600,
    limit: int = 0, log=print,
) -> pd.DataFrame:
    t0 = time.time()
    devices = devices or (["cuda:%d" % i for i in range(torch.cuda.device_count())] or ["cpu"])
    use_fp16 = devices[0] != "cpu"
    dtype = torch.float16 if use_fp16 else torch.bfloat16
    bundle = load_fold_models(weight_paths, devices)
    models, res = bundle["models"], bundle["res"]
    log(f"[exp018] {len(weight_paths)} fold model(s) x {len(devices)} device(s): {devices}, res {res}, "
        f"loaded in {time.time() - t0:.0f}s")

    test = pd.read_csv(test_csv)
    sdf = pd.read_csv(test_series_csv).rename(columns={
        "StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid",
        "Anatomical_Plane": "plane", "Fluid_Sensitive": "fluid"})
    studies = list(test["StudyInstanceUID"])
    if limit:
        studies = studies[:limit]
    tasks = [(u, sdf[sdf["study_uid"] == u][["series_uid", "plane", "fluid"]].to_dict("records"),
              str(series_root / u)) for u in studies]

    results: dict[str, np.ndarray] = {}
    q: queue.Queue = queue.Queue(maxsize=6)   # BOUNDED: backpressure on the decode pool
    errors: list[str] = []
    n_folds_used = {"n": len(weight_paths)}

    def gpu_worker(dev: str) -> None:
        my_models = models[dev]
        while True:
            item = q.get()
            if item is None:
                return
            uid, stack, present = item
            try:
                st = torch.from_numpy(stack).to(dev)
                # wall-clock guard: if we are behind schedule, use fewer folds for the remaining studies
                elapsed = time.time() - t0
                done = max(len(results), 1)
                projected = elapsed / done * len(studies)
                k = len(my_models)
                if projected > budget_seconds and k > 1:
                    k = max(1, int(k * budget_seconds / projected))
                    n_folds_used["n"] = min(n_folds_used["n"], k)
                probs = []
                for m in my_models[:k]:
                    logit = predict_study(m, st, present, res, dtype)
                    if not torch.isfinite(logit).all():  # fp16 overflow -> retry this fold in fp32
                        logit = predict_study(m, st, present, res, torch.float32)
                    probs.append(torch.sigmoid(logit.float()).cpu().numpy())
                results[uid] = np.mean(probs, axis=0)
                del st
            except Exception as e:  # noqa: BLE001
                errors.append(f"{uid}: {type(e).__name__}: {e}")
                torch.cuda.empty_cache() if dev != "cpu" else None

    workers = [threading.Thread(target=gpu_worker, args=(d,), daemon=True) for d in devices]
    for w in workers:
        w.start()
    # THREADS, not processes: a process pool deadlocks after CUDA init and cannot find its worker through the
    # notebook's custom package loader (2026-09-10 finding, submit-efficiency.ipynb). pydicom decompression and
    # scipy zoom release the GIL, so threads still parallelise decode. Bounded look-ahead: at most
    # num_decode_workers + 4 decoded stacks exist at once (7.2 MB each) besides the bounded GPU queue.
    #
    # Exit-safety (2026-09-23): everything from pool construction through the GPU workers' sentinel+join is
    # now one try/finally. The earlier version relied on this code running top-to-bottom unconditionally;
    # an exception anywhere in the decode loop would skip the `q.put(None)`/`w.join()` lines entirely (they
    # were plain statements AFTER the `with pool:` block, not inside a finally), and would leave `pool`'s
    # shutdown to the bare `with` statement's default `shutdown(wait=True)` -- no `cancel_futures`, so any
    # already-submitted-but-not-yet-started decode tasks would still be waited on instead of dropped. Both
    # are real hang/slow-exit risks (root-caused from a Kaggle commit that hung 9h after all visible work
    # finished; see infer_core.prefetch_decode's docstring for the mechanism), even though the GPU worker
    # threads themselves are daemon=True and can't alone block process exit.
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
            uid, stack, present, err = inflight.popleft().result()
            if err is not None:
                errors.append(f"{uid}: decode: {err}")
            else:
                q.put((uid, stack, present))
            n_done += 1
            if n_done % 50 == 0:
                log(f"[exp018] {n_done}/{len(studies)} studies decoded, {len(results)} predicted, "
                    f"{(time.time() - t0) / 60:.1f} min, folds in use {n_folds_used['n']}")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for _ in workers:
            q.put(None)
        # Untimed join (2026-09-23): the sentinels above guarantee each GPU worker returns once the queue
        # drains. A join(timeout=30) here (2ec038d) gave up on T4 while workers were still predicting,
        # silently writing label priors for queued studies and leaving daemon threads running CUDA work
        # at interpreter shutdown (blend018 v4 on Kaggle: 1/3 predicted, 2 on prior, 0 errors).
        for w in workers:
            w.join()

    prior = np.array([LABEL_PRIORS[c] for c in TARGET_COLUMNS])
    rows = []
    for u in test["StudyInstanceUID"]:
        p = results.get(u, prior)
        rows.append({"StudyInstanceUID": u, **{c: float(v) for c, v in zip(TARGET_COLUMNS, p)}})
    sub = pd.DataFrame(rows)
    sub.to_csv(out_csv, index=False)
    log(f"[exp018] wrote {out_csv}: {len(sub)} rows, {len(results)} predicted, {len(sub) - len(results)} on prior, "
        f"{len(errors)} errors, {(time.time() - t0) / 60:.1f} min total")
    for e in errors[:10]:
        log(f"  [error] {e}")
    return sub
