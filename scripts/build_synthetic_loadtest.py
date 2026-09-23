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

"""Builds a synthetic ~1300-study test set from real train studies (CLAUDE.md §9 load-test
rehearsal) — symlinks train_series/<study>/<series>/*.dcm dirs into a synthetic test_series/ tree
(no data duplication) and writes test.csv/test_series.csv matching the real test-time schema
exactly (StudyInstanceUID only / StudyInstanceUID+SeriesInstanceUID+Fluid_Sensitive+
Fat_Suppression+Anatomical_Plane), stripped of report/label columns the same way the real hidden
test set is.

Usage:
    python scripts/build_synthetic_loadtest.py --raw data/raw --out artifacts/synthetic_loadtest --n-studies 1300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-studies", type=int, default=1300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    series_df = pd.read_csv(args.raw / "train_series.csv")
    all_studies = series_df["StudyInstanceUID"].unique()
    rng = pd.Series(all_studies).sample(n=min(args.n_studies, len(all_studies)), random_state=args.seed)
    chosen = set(rng.tolist())

    out_series_root = args.out / "test_series"
    out_series_root.mkdir(parents=True, exist_ok=True)

    subset = series_df[series_df["StudyInstanceUID"].isin(chosen)].copy()
    subset.to_csv(args.out / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": sorted(chosen)}).to_csv(args.out / "test.csv", index=False)

    linked = skipped = 0
    for study_uid in chosen:
        src = args.raw / "train_series" / study_uid
        dst = out_series_root / study_uid
        if dst.exists() or dst.is_symlink():
            linked += 1
            continue
        if not src.exists():
            skipped += 1
            continue
        dst.symlink_to(src.resolve(), target_is_directory=True)
        linked += 1

    print(f"{len(chosen)} studies selected, {linked} symlinked, {skipped} missing source dirs")
    print(f"Wrote {args.out / 'test.csv'} and {args.out / 'test_series.csv'}")


if __name__ == "__main__":
    main()
