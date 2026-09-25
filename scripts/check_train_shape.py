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

"""Pre-launch trainability check for a Window-MIL backbone, at the REAL training and eval shapes.

The exp019 lesson (2026-09-24): a latency benchmark at the inference shape (one study per forward) cleared
CoAtNet-3, which then died at the first TRAINING step -- 8 studies x 12 windows = 96 images per forward
pushed a conv tensor past 2^31 elements (no 64-bit index path in F.conv2d). This script runs exactly what
train_wmil.py runs, on real cached studies, before any fold is launched:

  train: batch_studies x k_train windows, bf16 autocast, grad checkpointing, BCE + backward + AdamW step
  eval : 2 studies x ALL candidate windows (<= 54 each), no grad (train_wmil.py's eval_bs = 2)

For each backbone it reports pass/fail, the largest activation tensor seen (as a fraction of INT_MAX),
peak GPU memory, seconds per train step (-> projected epoch time) and seconds per eval study.

    python scripts/check_train_shape.py --arch vit_base_patch14_dinov2.lvd142m --res 378 --img-size 378
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.wmil import (  # noqa: E402
    MAX_WINDOWS, NUM_SLOTS, WindowMIL, candidate_centers, gather_windows, prepare_input,
)

INT_MAX = 2 ** 31 - 1


def load_studies(root: Path, n: int) -> tuple[list, list]:
    import pandas as pd
    man = pd.concat([pd.read_parquet(f) for f in sorted((root / "artifacts/cache_manifests_wmil").glob("shard_*.parquet"))],
                    ignore_index=True)
    man = man[man["ok"]]
    stacks, presents = [], []
    for _, r in man.iterrows():
        p = root / "data/cache_wmil" / f"{r['study_uid']}.npy"
        if p.exists():
            stacks.append(np.load(p))
            presents.append(np.array([bool(r[f"present_{i}"]) for i in range(NUM_SLOTS)]))
        if len(stacks) >= n:
            break
    return stacks, presents


def check(arch: str, res: int, img_size: int | None, bk: dict | None, stacks, presents, bs: int, k: int,
          steps: int) -> dict:
    dev = "cuda"
    out = {"arch": arch, "res": res, "img_size": img_size, "backbone_kwargs": bk}
    torch.cuda.reset_peak_memory_stats()
    biggest = {"n": 0}

    def hook(_m, _i, o):
        if torch.is_tensor(o):
            biggest["n"] = max(biggest["n"], o.numel())

    try:
        model = WindowMIL(arch, pretrained=False, grad_checkpointing=True, img_size=img_size,
                          backbone_kwargs=bk).to(dev)
    except Exception as e:  # noqa: BLE001
        return {**out, "ok": False, "stage": "construct", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    out["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    # ---- TRAIN shape: exactly train_wmil.py's step
    try:
        times = []
        for step in range(steps):
            idx = rng.choice(len(stacks), size=bs, replace=False)
            st = torch.from_numpy(np.stack([stacks[i] for i in idx])).to(dev)
            centers = []
            for i in idx:
                cs = candidate_centers(presents[i])
                centers.append(cs[rng.choice(len(cs), size=k, replace=len(cs) < k)])
            w = gather_windows(st, torch.from_numpy(np.stack(centers)).to(dev))
            y = torch.rand(bs, 12, device=dev)
            torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = prepare_input(w, res).view(bs, k, 3, res, res)
                logits = model(x)
            loss = lossf(logits.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
            torch.cuda.synchronize()
            if step > 0:  # step 0 = warm-up / autotune
                times.append(time.time() - t0)
        out["train_s_per_step"] = float(np.mean(times))
        out["train_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    except Exception as e:  # noqa: BLE001
        return {**out, "ok": False, "stage": f"train ({bs}x{k}={bs * k} images/forward)",
                "error": f"{type(e).__name__}: {str(e)[:200]}", "max_numel_frac": biggest["n"] / INT_MAX}

    # ---- EVAL shape: 2 studies x all candidate windows, padded to MAX_WINDOWS as train_wmil.predict does
    model.eval(); torch.cuda.reset_peak_memory_stats()
    try:
        ev = []
        with torch.no_grad():
            for j in range(0, min(len(stacks), 12), 2):
                st = torch.from_numpy(np.stack(stacks[j:j + 2])).to(dev)
                cen, msk = [], []
                for p in presents[j:j + 2]:
                    cs = candidate_centers(p)
                    c = np.full(MAX_WINDOWS, cs[0], dtype=np.int64); m = np.zeros(MAX_WINDOWS, dtype=bool)
                    c[:len(cs)], m[:len(cs)] = cs, True
                    cen.append(c); msk.append(m)
                w = gather_windows(st, torch.from_numpy(np.stack(cen)).to(dev))
                torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    f = model.encode(prepare_input(w, res).view(w.shape[0], w.shape[1], 3, res, res))
                model.head(f.float(), torch.from_numpy(np.stack(msk)).to(dev))
                torch.cuda.synchronize()
                if j > 0:
                    ev.append((time.time() - t0) / 2)
        out["eval_s_per_study"] = float(np.mean(ev))
        out["eval_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    except Exception as e:  # noqa: BLE001
        return {**out, "ok": False, "stage": f"eval (2x{MAX_WINDOWS} images/forward)",
                "error": f"{type(e).__name__}: {str(e)[:200]}", "max_numel_frac": biggest["n"] / INT_MAX}
    for h in hooks:
        h.remove()
    out["max_numel_frac"] = biggest["n"] / INT_MAX
    out["ok"] = True
    del model, opt
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", nargs="+", required=True,
                    help="arch:res[:img_size] entries, e.g. vit_base_patch14_dinov2.lvd142m:378:378")
    ap.add_argument("--batch-studies", type=int, default=8)
    ap.add_argument("--k-train", type=int, default=12)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--steps-per-epoch", type=int, default=433, help="train studies (~3470) / batch (8)")
    args = ap.parse_args()

    root = Path(os.environ["RSNA_ROOT"])
    stacks, presents = load_studies(root, 40)
    print(f"gpu={torch.cuda.get_device_name(0)} | {len(stacks)} real cached studies | train shape "
          f"{args.batch_studies}x{args.k_train} = {args.batch_studies * args.k_train} images/forward", flush=True)
    for spec in args.spec:
        parts = spec.split(":")
        arch, res = parts[0], int(parts[1])
        img_size = int(parts[2]) if len(parts) > 2 and parts[2] else None
        # ViTs must keep their pretrained final norm under avg pooling (see WindowMIL.__init__)
        bk = {"fc_norm": False} if arch.startswith("vit_") else None
        r = check(arch, res, img_size, bk, stacks, presents, args.batch_studies, args.k_train, args.steps)
        if not r["ok"]:
            print(f"FAIL {arch} res={res} img_size={img_size} at {r['stage']}: {r['error']}", flush=True)
            continue
        epoch_min = r["train_s_per_step"] * args.steps_per_epoch / 60
        print(f"PASS {arch} res={res} img_size={img_size}: {r['params_m']:.1f}M params | largest activation "
              f"{r['max_numel_frac']:.2f} x INT_MAX | train {r['train_s_per_step']:.2f} s/step, peak "
              f"{r['train_peak_gb']:.1f} GB -> ~{epoch_min:.1f} min/epoch train-only | eval "
              f"{r['eval_s_per_study']:.3f} s/study (1 model), peak {r['eval_peak_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
