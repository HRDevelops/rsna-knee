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

"""Shutdown-safety test for the exp018 solo driver (submit-exp018.ipynb, already scored 0.929 on
Kaggle -- this is a defense-in-depth check, not a sign anything went wrong there; the failure this
project actually hit was in the NEW probe #4 blend driver). Same structure as
scripts/test_blend_shutdown.py, calling rsna_knee.infer_wmil.run_exp018 instead.

    timeout 120 python scripts/test_exp018_shutdown.py --n-studies 3; echo EXIT=$?
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
    ap.add_argument("--decode-workers", type=int, default=3)
    args = ap.parse_args()

    t_start = time.time()
    print(f"[test] start, n_studies={args.n_studies}, pid={os.getpid()}", flush=True)

    from rsna_knee.infer_wmil import run_exp018  # noqa: E402

    root = Path(os.environ["RSNA_ROOT"])
    art = root / "artifacts"
    raw = root / "data/raw"

    import pandas as pd  # noqa: E402

    val = pd.read_parquet(art / "exp018_coatnet2_wmil_v3labels/fold0/oof.parquet")
    uids = list(val["study_uid"].iloc[: args.n_studies])
    tmp = art / f"_exp018_shutdown_test_n{args.n_studies}"
    tmp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"StudyInstanceUID": uids}).to_csv(tmp / "test.csv", index=False)
    sdf = pd.read_csv(raw / "train_series.csv")
    sdf[sdf["StudyInstanceUID"].isin(uids)].to_csv(tmp / "test_series.csv", index=False)

    sub = run_exp018(
        tmp / "test.csv", tmp / "test_series.csv", raw / "train_series",
        [art / f"exp018_coatnet2_wmil_v3labels/fold{k}/best.pt" for k in range(5)],
        tmp / "submission.csv", num_decode_workers=args.decode_workers,
    )
    assert len(sub) == args.n_studies, f"expected {args.n_studies} rows, got {len(sub)}"
    print(f"[test] all work finished at {time.time() - t_start:.1f}s ({len(sub)} rows) -- returning "
          f"from main() now.", flush=True)


if __name__ == "__main__":
    main()
    print(f"[test] main() returned normally at {time.time():.0f} (wall clock); process should exit now.",
          flush=True)
