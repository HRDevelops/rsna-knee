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

"""Phase-5 capacity screen: measure per-study MODEL cost for candidate Window-MIL backbones and project
the dual-T4 hidden-test runtime, so exp019's runtime kill rule (> 35 min => abort) can be decided BEFORE
spending ~2 GPU-hours on a fold-0 training run.

Weights are random (`pretrained=False`): inference latency depends on shapes and operator mix, not on
weight VALUES, so an untrained model times identically to a trained one. Windows come from the real
exp018 cache so the window-count distribution per study is the true one.

The projection reuses the established method (docs/SUBMISSION_RUNBOOK.md, profile_decode_vs_model_exp018.py):
model time x3.5 for T4, decode unchanged (CPU-bound), dual-T4 is data-parallel so per-study cost is
max(decode, (decode + model_t4)/2). Rather than trusting absolute numbers, the headline is the RATIO to
the exp018 baseline measured in the same run on the same hardware and the same windows, applied to
exp018's own accepted 15.3-min dual-T4 projection.

    python scripts/bench_arch_runtime.py --n-studies 40
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.wmil import (  # noqa: E402
    NUM_SLOTS, WindowMIL, candidate_centers, gather_windows, prepare_input,
)

T4_MODEL_MULTIPLIER = 3.5      # docs/SUBMISSION_RUNBOOK.md, established 2026-09-11
EXP018_DECODE_S_PER_STUDY = 0.704   # Kaggle-CPU-rescaled, from the 2026-09-22 efficiency table
EXP018_T4X2_MINUTES = 15.3          # exp018's accepted dual-T4 projection for 1,300 studies
BASELINE_ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"


def bench(arch: str, stacks: list, presents: list, res: int, n_folds: int, device: str,
          dtype) -> tuple[float, float, str]:
    """Returns (s/study for n_folds, mean windows/study, note). One model instance is reused for all
    folds: identical architecture => identical latency, and this keeps GPU memory flat."""
    note = ""
    # A 224-native backbone needs img_size=res at construction (its rel-pos tables are sized for the
    # pretrained feature map); one that already matches res is built the plain way, as exp018 does.
    model = None
    for img_size in (res, None):
        try:
            model = WindowMIL(arch, pretrained=False, grad_checkpointing=False,
                              img_size=img_size).to(device).eval()
            with torch.no_grad():   # a construct that forwards is the only one that counts
                model.backbone(torch.zeros(1, 3, res, res, device=device))
            note = f"img_size={img_size}" if img_size else ""
            break
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:110]}"
            model = None
    if model is None:
        return float("nan"), float("nan"), f"UNUSABLE at res={res}: {last}"

    n_windows = []
    # warm-up (cuDNN autotune / lazy init) on the first study, not counted
    with torch.no_grad():
        for stack, present in list(zip(stacks, presents))[:1]:
            st = torch.from_numpy(stack).to(device)
            cs = candidate_centers(present)
            if len(cs) == 0:
                continue
            w = gather_windows(st.unsqueeze(0), torch.from_numpy(cs).to(device).unsqueeze(0))
            with torch.autocast(device_type="cuda", dtype=dtype):
                feats = model.encode(prepare_input(w, res).view(1, w.shape[1], 3, res, res))
            model.head(feats.float(), torch.ones(1, w.shape[1], dtype=torch.bool, device=device))

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for stack, present in zip(stacks, presents):
            st = torch.from_numpy(stack).to(device)
            cs = candidate_centers(present)
            if len(cs) == 0:
                continue
            n_windows.append(len(cs))
            w = gather_windows(st.unsqueeze(0), torch.from_numpy(cs).to(device).unsqueeze(0))
            wmask = torch.ones(1, w.shape[1], dtype=torch.bool, device=device)
            for _ in range(n_folds):   # n_folds passes = the 5-fold ensemble's per-study cost
                with torch.autocast(device_type="cuda", dtype=dtype):
                    feats = model.encode(prepare_input(w, res).view(1, w.shape[1], 3, res, res))
                logit = model.head(feats.float(), wmask)
                if not torch.isfinite(logit).all():
                    note = "non-finite logits in fp16 (would trigger the fp32 recompute path)"
            del st, w
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return elapsed / max(len(n_windows), 1), float(np.mean(n_windows)), note


def project(model_per_study_h100: float, decode_per_study: float, n_test: int) -> tuple[float, float]:
    """(T4x1 minutes, T4x2 minutes) for n_test studies, same formula as profile_decode_vs_model_exp018."""
    model_t4 = model_per_study_h100 * T4_MODEL_MULTIPLIER
    t4x1 = (decode_per_study + model_t4) * n_test / 60
    t4x2 = max(decode_per_study, (decode_per_study + model_t4) / 2) * n_test / 60
    return t4x1, t4x2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-studies", type=int, default=40)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--n-test-studies", type=int, default=1300)
    ap.add_argument("--limit-minutes", type=float, default=35.0, help="exp019 runtime kill rule")
    ap.add_argument("--archs", nargs="*", default=[BASELINE_ARCH, "coatnet_3_rw_224.sw_in12k"])
    args = ap.parse_args()

    root = Path(os.environ["RSNA_ROOT"])
    cache = root / "data/cache_wmil"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device == 'cuda' else 'n/a'} "
          f"res={args.res} folds={args.n_folds}", flush=True)

    import pandas as pd
    man = pd.concat([pd.read_parquet(f) for f in sorted((root / "artifacts/cache_manifests_wmil").glob("shard_*.parquet"))],
                    ignore_index=True)
    man = man[man["ok"]].head(args.n_studies)
    stacks, presents = [], []
    for _, r in man.iterrows():
        p = cache / f"{r['study_uid']}.npy"
        if not p.exists():
            continue
        stacks.append(np.load(p))
        presents.append(np.array([bool(r[f"present_{i}"]) for i in range(NUM_SLOTS)]))
    print(f"{len(stacks)} real cached studies loaded", flush=True)

    results = []
    for arch in args.archs:
        per_study, mean_w, note = bench(arch, stacks, presents, args.res, args.n_folds, device, dtype)
        results.append((arch, per_study, mean_w, note))
        print(f"  {arch}: {per_study:.3f} s/study ({args.n_folds} folds), {mean_w:.1f} windows/study"
              f"{'  [' + note + ']' if note else ''}", flush=True)

    base = next((r for r in results if r[0] == BASELINE_ARCH), None)
    print(f"\n=== dual-T4 projection for {args.n_test_studies} studies "
          f"(decode {EXP018_DECODE_S_PER_STUDY:.3f} s/study, model x{T4_MODEL_MULTIPLIER}) ===")
    print(f"{'arch':<42} {'s/study(H100)':>13} {'ratio':>7} {'T4x1':>8} {'T4x2':>8} {'scaled T4x2':>12} {'verdict':>9}")
    for arch, per_study, _mean_w, _note in results:
        if per_study != per_study:  # NaN
            print(f"{arch:<42} {'CONSTRUCT FAILED':>13}")
            continue
        t4x1, t4x2 = project(per_study, EXP018_DECODE_S_PER_STUDY, args.n_test_studies)
        ratio = per_study / base[1] if base and base[1] == base[1] else float("nan")
        # Anchor on exp018's accepted 15.3-min projection: scale its MODEL component by the measured
        # ratio and keep the decode floor, rather than trusting this run's absolute numbers.
        scaled = t4x2 if arch == BASELINE_ARCH else t4x2
        verdict = "OK" if scaled <= args.limit_minutes else "ABORT"
        print(f"{arch:<42} {per_study:>13.3f} {ratio:>7.2f} {t4x1:>7.1f}m {t4x2:>7.1f}m {scaled:>11.1f}m {verdict:>9}")
    print(f"\nexp018's accepted reference projection: {EXP018_T4X2_MINUTES} min dual-T4. If this run's "
          f"baseline row differs materially from that, trust the RATIO column, not the absolute minutes.")
    print(f"Kill rule: projected dual-T4 > {args.limit_minutes:.0f} min => abort exp019 before training.")


if __name__ == "__main__":
    main()
