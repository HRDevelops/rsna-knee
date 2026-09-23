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

"""Shutdown-safety test for `infer_core.prefetch_decode`, used directly by `submit-ensemble.ipynb` and
`submit-efficiency.ipynb` (cell-8 in both, NOT wrapped in a driver function like exp018/blend018 --
their decode loop lives in the notebook cell itself: `for uid, slots, tags, err in
prefetch_decode(tasks, decode_fn, workers): if guard.over(...): break; ...`).

This deliberately reproduces the EARLY-BREAK scenario (the exact shape of that `if guard.over(): break`
line) rather than just letting the generator exhaust normally -- that early exit, abandoning the
generator while the feeder thread may still be mid-decode, is what the root-cause analysis identified
as the trigger (a small task list makes it easy for the feeder to still be "ahead" when the consumer
stops). Also forces `gc.collect()` after breaking, since on Kaggle's Python the generator's
`__del__`/`close()` (which is what actually reaches this function's `finally:` block) may not run
until garbage collection, exactly as it would at real interpreter shutdown.

    timeout 120 python scripts/test_prefetch_decode_shutdown.py --n-studies 3; echo EXIT=$?
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-studies", type=int, required=True)
    ap.add_argument("--decode-workers", type=int, default=3)
    ap.add_argument("--break-after", type=int, default=1,
                     help="consume this many items then break early, simulating the wall-clock guard")
    args = ap.parse_args()

    t_start = time.time()
    print(f"[test] start, n_studies={args.n_studies}, pid={os.getpid()}", flush=True)

    from rsna_knee.dataset import _patch_size_for_backbone  # noqa: E402
    from rsna_knee.infer_core import decode_one_for_pool, nearest_patch_multiple, prefetch_decode  # noqa: E402

    root = Path(os.environ["RSNA_ROOT"])
    art = root / "artifacts"
    raw = root / "data/raw"

    import pandas as pd  # noqa: E402

    val = pd.read_parquet(art / "exp018_coatnet2_wmil_v3labels/fold0/oof.parquet")
    uids = list(val["study_uid"].iloc[: args.n_studies])
    sdf = pd.read_csv(raw / "train_series.csv").rename(
        columns={"StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid"})
    out_size = nearest_patch_multiple(384, _patch_size_for_backbone("vit_base_patch14_dinov2.lvd142m"))
    tasks = [(u, sdf[sdf["study_uid"] == u].to_dict("records"), str(raw / "train_series"), 24, out_size)
             for u in uids]

    n_consumed = 0
    gen = prefetch_decode(tasks, decode_one_for_pool, args.decode_workers)
    for uid, slots, tags, err in gen:
        n_consumed += 1
        print(f"[test] consumed item {n_consumed}: {uid[-8:]}, err={err}", flush=True)
        if n_consumed >= min(args.break_after, args.n_studies):
            print(f"[test] breaking early after {n_consumed}/{args.n_studies} items (simulating "
                  f"WallClockGuard.over())", flush=True)
            break
    del gen
    gc.collect()  # force the abandoned generator's finalizer to run now, as it would eventually at exit

    print(f"[test] done at {time.time() - t_start:.1f}s, consumed {n_consumed}/{args.n_studies} -- "
          f"returning from main() now.", flush=True)


if __name__ == "__main__":
    main()
    print(f"[test] main() returned normally at {time.time():.0f} (wall clock); process should exit now.",
          flush=True)
