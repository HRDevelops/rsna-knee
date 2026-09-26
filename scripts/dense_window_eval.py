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

"""Dense-window inference test (main track, report only): the EXISTING exp018 checkpoints evaluated on a denser
slice layout, with no retraining. This is the public lineage's "more windows at eval, same model" idea
(`raptor_k_eval` 62 -> 94).

  --layout standard : exp018's own 5 slots / 64 slices -> <= 54 windows    (the CONTROL)
  --layout dense    : every slot x --scale (1.6)       -> [29,22,19,13,19] = 102 slices -> <= 92 windows

Both layouts go through the SAME inference path (DICOM decode -> tag/geometry side -> scored predict_study,
fp16 as deployed), so the only difference between the two runs is density. The library code is unmodified: the
layout lives in rsna_knee.wmil's module globals (SLOT_LEN / SLOT_OFF / DEPTH / MAX_WINDOWS), which
decode_study, candidate_centers, normalize_orientation and predict_study all read at call time, and this
process sets them before any decode. Each non-gold study is predicted by its own out-of-fold model (OOF);
gold-58 by the mean of all 5 folds.

    python scripts/dense_window_eval.py --layout dense --scale 1.6 --workers 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee import wmil  # noqa: E402

EXP = "exp018_coatnet2_wmil_v3labels"


def set_layout(scale: float) -> list[int]:
    base = [s[3] for s in wmil.SLOT_SPECS]
    lens = base if scale == 1.0 else [int(round(l * scale)) for l in base]
    wmil.SLOT_LEN = lens
    wmil.SLOT_OFF = [int(x) for x in np.concatenate([[0], np.cumsum(lens)[:-1]])]
    wmil.DEPTH = int(sum(lens))
    wmil.MAX_WINDOWS = sum(max(l - 2, 0) for l in lens)
    return lens


def decode_task(args):
    uid, rows, study_dir = args
    try:
        stack, present, metas = wmil.decode_study(Path(study_dir), rows)
        if not present.any():
            return uid, None, None, "no slot decoded"
        sides = wmil.resolve_slot_sides(metas, None)          # the inference-time side rule, as deployed
        return uid, wmil.normalize_orientation(stack, metas, sides), present, None
    except Exception as e:  # noqa: BLE001
        return uid, None, None, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", choices=["standard", "dense"], required=True)
    ap.add_argument("--scale", type=float, default=1.6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    root = Path(os.environ["RSNA_ROOT"])
    art, raw = root / "artifacts", root / "data/raw"
    lens = set_layout(1.0 if a.layout == "standard" else a.scale)
    print(f"layout={a.layout} slots={lens} depth={wmil.DEPTH} max_windows={wmil.MAX_WINDOWS}", flush=True)

    folds = pd.read_parquet(art / "folds.parquet")[["study_uid", "fold"]]
    folds = folds[folds.fold.isin([-1, 0, 1, 2, 3, 4])]
    if a.limit:
        folds = pd.concat([folds[folds.fold == -1], folds[folds.fold >= 0].head(a.limit)])
    sdf = pd.read_csv(raw / "train_series.csv").rename(columns={
        "StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid",
        "Anatomical_Plane": "plane", "Fluid_Sensitive": "fluid"})
    by = {u: g[["series_uid", "plane", "fluid"]].to_dict("records") for u, g in sdf.groupby("study_uid")}
    fold_of = dict(zip(folds.study_uid, folds.fold))
    tasks = [(u, by.get(u, []), str(raw / "train_series" / u)) for u in folds.study_uid]

    dev = "cuda"
    models = []
    for k in range(5):
        ck = torch.load(art / EXP / f"fold{k}/best.pt", map_location="cpu", weights_only=False)
        m = wmil.WindowMIL(ck["arch"], pretrained=False, grad_checkpointing=False)
        m.load_state_dict(ck["model"], strict=True)
        models.append(m.to(dev).eval())
    res = int(ck["res"])

    rows, errs, t0 = [], [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        it, inflight = iter(tasks), deque()
        while True:
            while len(inflight) < a.workers * 2:
                t = next(it, None)
                if t is None:
                    break
                inflight.append(pool.submit(decode_task, t))
            if not inflight:
                break
            uid, stack, present, err = inflight.popleft().result()
            if err is not None:
                errs.append(f"{uid}: {err}")
                continue
            f = fold_of[uid]
            use = models if f == -1 else [models[f]]           # gold-58: 5-fold mean; others: own OOF model
            st = torch.from_numpy(stack).to(dev)
            with torch.no_grad():
                probs = []
                for m in use:
                    lg = wmil.predict_study(m, st, present, res, torch.float16)
                    if not torch.isfinite(lg).all():
                        lg = wmil.predict_study(m, st, present, res, torch.float32)
                    probs.append(torch.sigmoid(lg.float()).cpu().numpy())
            p = np.mean(probs, axis=0)
            rows.append({"study_uid": uid, "fold": f, "n_windows": len(wmil.candidate_centers(present)),
                         **{c: float(v) for c, v in zip(wmil.TARGET_COLUMNS, p)}})
            if len(rows) % 500 == 0:
                print(f"  {len(rows)}/{len(tasks)} predicted, {(time.time() - t0) / 60:.1f} min", flush=True)
    out = art / f"dense_window_eval_{a.layout}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    nw = pd.DataFrame(rows)["n_windows"]
    print(f"wrote {out}: {len(rows)} studies, {len(errs)} decode errors, mean windows/study {nw.mean():.1f} "
          f"(max {nw.max()}), {(time.time() - t0) / 60:.1f} min", flush=True)
    for e in errs[:5]:
        print("  [err]", e)


if __name__ == "__main__":
    main()
