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

"""Standalone shutdown-safety test for the probe #4 blend driver (2026-09-23, root-cause: a Kaggle
commit wrote submission.csv + __results__.html, i.e. all cell code had already finished, then hung 9h
at kernel/interpreter teardown -- exactly the "N is small (dummy test uses N=3), a bounded-queue
producer blocks forever because the consumer stopped, so a ThreadPoolExecutor's non-daemon worker
threads never get shut down" failure mode documented in infer_core.prefetch_decode and
infer_wmil.run_exp018/infer_blend018.run_blend018's docstrings/comments).

This script does exactly what the Kaggle commit does (build weights, run the full blend pipeline on
N real studies, write a submission.csv) and then just RETURNS from main() -- no os._exit(), no
sys.exit() with a nonzero code, nothing that would mask a real hang. If the fix is correct, the
Python process exits on its own within seconds of the last print; if it isn't, this script (unlike a
Kaggle commit) will make that hang visible immediately under an external `timeout`:

    timeout 120 python scripts/test_blend_shutdown.py --n-studies 3; echo EXIT=$?

Exit code 0 within the timeout = pass. A `timeout`-induced kill (exit code 124) = the bug is still
there. Runs N=1, 2, 3 by default in one process-per-N (via --n-studies, called once per value from
the shell) so each run's shutdown is independently observable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-studies", type=int, required=True)
    ap.add_argument("--weight-exp018", type=float, default=0.75)
    ap.add_argument("--decode-workers", type=int, default=3)  # Kaggle-realistic, not the H100 profiling jobs' 8
    args = ap.parse_args()

    t_start = time.time()
    print(f"[test] start, n_studies={args.n_studies}, pid={os.getpid()}", flush=True)

    from rsna_knee.infer_blend018 import run_blend018  # noqa: E402

    root = Path(os.environ["RSNA_ROOT"])
    art = root / "artifacts"
    stage = root / "kaggle_staging"
    raw = root / "data/raw"

    import pandas as pd  # noqa: E402

    val = pd.read_parquet(art / "exp018_coatnet2_wmil_v3labels/fold0/oof.parquet")
    uids = list(val["study_uid"].iloc[: args.n_studies])
    tmp = art / f"_shutdown_test_n{args.n_studies}"
    tmp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"StudyInstanceUID": uids}).to_csv(tmp / "test.csv", index=False)
    sdf = pd.read_csv(raw / "train_series.csv")
    sdf[sdf["StudyInstanceUID"].isin(uids)].to_csv(tmp / "test_series.csv", index=False)

    sub = run_blend018(
        tmp / "test.csv", tmp / "test_series.csv", raw / "train_series", raw / "train_series",
        root / "code/src/rsna_knee/ensemble_recipe_v1.yaml", stage,
        [art / f"exp018_coatnet2_wmil_v3labels/fold{k}/best.pt" for k in range(5)],
        art / "exp018_coatnet2_wmil_v3labels/rank_reference.npz",
        art / "ensemble_v1_blend_output/rank_reference.npz",
        tmp / "submission.csv", weight_exp018=args.weight_exp018, num_decode_workers=args.decode_workers,
    )
    assert len(sub) == args.n_studies, f"expected {args.n_studies} rows, got {len(sub)}"
    # Row count alone is not enough: a timed worker join once filled rows with label priors on a slow GPU
    # (Kaggle T4) while passing here on H100. Every row must be a real prediction, not the prior.
    from rsna_knee.infer_core import prior_row  # noqa: E402
    n_prior = int((sub.iloc[:, 1:].to_numpy() == prior_row()).all(axis=1).sum())
    assert n_prior == 0, f"{n_prior}/{len(sub)} rows are label priors, not predictions"

    work_done_at = time.time() - t_start
    print(f"[test] all work finished at {work_done_at:.1f}s (submission.csv written, {len(sub)} rows) "
          f"-- returning from main() now. If the process does not exit within a few seconds of THIS "
          f"line, the shutdown-hang bug is still present.", flush=True)
    # Deliberately nothing else here: no os._exit(), no thread/pool handles kept alive, no sys.exit()
    # with a special code -- a plain return, exactly what proves (or disproves) clean interpreter exit.


if __name__ == "__main__":
    main()
    print(f"[test] main() returned normally at {time.time():.0f} (wall clock); process should exit now.",
          flush=True)
