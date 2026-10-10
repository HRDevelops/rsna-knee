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

"""Extract Window-MIL attention for the gold-58 studies (inference only, existing checkpoints, no scored artifact
touched). For every gold-58 study: decode with the scored `rsna_knee.wmil` path (standard layout, the deployed
side rule), run the 10 exp021 checkpoints (exp018 seed 42 x 5 folds + seed 43 x 5 folds) with fp16 encode as
deployed, and recompute the per-finding attention a_{k,o} = softmax_k(w_o . tanh(W h_k)) from the same features. The
library is not modified: the attention is recomputed from `model.norm` / `model.att`, and the recomputed logits are
asserted equal to `model.head(...)`, so the attention shown is exactly the one that produced the prediction.

Windows are identical across checkpoints (same study, same candidate centres), so attention is averaged over the 10
checkpoints. Saves attention + predictions for all 58 studies, and the 64-slice stack (uint8) only for the studies
listed in --selection.

    python scripts/viz_attention.py --selection artifacts/viz_attention/attention_selection.csv
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

EXPS = ["exp018_coatnet2_wmil_v3labels", "exp021_coatnet2_wmil_seed43"]


def decode_task(args):
    uid, rows, study_dir = args
    try:
        stack, present, metas = wmil.decode_study(Path(study_dir), rows)
        if not present.any():
            return uid, None, None, "no slot decoded"
        sides = wmil.resolve_slot_sides(metas, None)
        return uid, wmil.normalize_orientation(stack, metas, sides), present, None
    except Exception as e:  # noqa: BLE001
        return uid, None, None, f"{type(e).__name__}: {e}"


@torch.no_grad()
def attention_and_logits(model, stack: torch.Tensor, centers: torch.Tensor, res: int):
    """-> attention [K, 12] (fp32), logits [12]. Same encode path as wmil.predict_study (fp16 autocast), then the
    head recomputed in fp32 with the attention exposed."""
    dev = stack.device
    feats = []
    for s in range(0, centers.shape[1], 64):
        w = wmil.gather_windows(stack[None], centers[:, s:s + 64])
        with torch.autocast(device_type=dev.type, dtype=torch.float16):
            feats.append(model.encode(wmil.prepare_input(w, res).view(1, -1, 3, res, res)).float())
    feats = torch.cat(feats, dim=1)
    with torch.autocast(device_type=dev.type, enabled=False):
        h = model.norm(feats)
        a = torch.softmax(model.att(h), dim=1)                       # [1, K, 12]
        logit = (torch.einsum("bkn,bkf->bnf", a, h) * model.cls_w).sum(-1) + model.cls_b
        ref = model.head(feats)
    assert torch.allclose(logit, ref, atol=1e-4), float((logit - ref).abs().max())
    return a[0].cpu().numpy(), logit[0].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = Path(os.environ["RSNA_ROOT"])
    art, raw = root / "artifacts", root / "data/raw"
    out = a.out or art / "viz_attention"
    out.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(a.selection)
    keep_stack = set(sel.study_uid)
    folds = pd.read_parquet(art / "folds.parquet")[["study_uid", "fold"]]
    gold = list(folds[folds.fold == -1].study_uid)
    if a.limit:
        gold = sorted(keep_stack) + [u for u in gold if u not in keep_stack][: max(a.limit - len(keep_stack), 0)]
    sdf = pd.read_csv(raw / "train_series.csv").rename(columns={
        "StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid",
        "Anatomical_Plane": "plane", "Fluid_Sensitive": "fluid"})
    by = {u: g[["series_uid", "plane", "fluid"]].to_dict("records") for u, g in sdf.groupby("study_uid")}
    tasks = [(u, by.get(u, []), str(raw / "train_series" / u)) for u in gold]

    dev = "cuda"
    models, names = [], []
    for e in EXPS:
        for k in range(5):
            ck = torch.load(art / e / f"fold{k}/best.pt", map_location="cpu", weights_only=False)
            m = wmil.WindowMIL(ck["arch"], pretrained=False, grad_checkpointing=False)
            m.load_state_dict(ck["model"], strict=True)
            models.append(m.to(dev).eval())
            names.append(f"{e.split('_')[0]}_fold{k}")
    res = int(ck["res"])
    print(f"loaded {len(models)} checkpoints, res {res}, {len(tasks)} studies", flush=True)

    store, errs, t0 = {}, [], time.time()
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
            st = torch.from_numpy(stack).to(dev)
            cs = wmil.candidate_centers(present)
            centers = torch.from_numpy(cs).to(dev)[None]
            att, lg = [], []
            for m in models:
                at, lo = attention_and_logits(m, st, centers, res)
                att.append(at)
                lg.append(lo)
            att, lg = np.stack(att), np.stack(lg)                    # [10, K, 12], [10, 12]
            slot_of = np.searchsorted(np.array(wmil.SLOT_OFF), cs, side="right") - 1
            store[f"{uid}|attn"] = att.astype(np.float32)
            store[f"{uid}|prob"] = (1 / (1 + np.exp(-lg))).astype(np.float32)
            store[f"{uid}|centers"] = cs
            store[f"{uid}|slot_of"] = slot_of
            store[f"{uid}|present"] = present
            if uid in keep_stack:
                store[f"{uid}|stack"] = stack
            n_done = sum(k.endswith("|attn") for k in store)
            if n_done % 10 == 0 or n_done == len(tasks):
                print(f"  {n_done}/{len(tasks)} studies, {(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez_compressed(out / "attention_gold58.npz", **store)
    pd.Series(names).to_csv(out / "checkpoint_order.csv", index=False, header=False)
    print(f"wrote {out / 'attention_gold58.npz'}: {sum(k.endswith('|attn') for k in store)} studies, "
          f"{len(errs)} decode errors, {(time.time() - t0) / 60:.1f} min", flush=True)
    for e in errs[:5]:
        print("  [err]", e)


if __name__ == "__main__":
    main()
