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

"""Cache v2 build (docs/research/PLAN_v1_3.md s4.1, Phase 0B): every series of every study, every native slice,
uint8 [n, 384, 384] per SERIES at 0.4167 mm (160 mm field), exact affine + linear interpolation, no added prefilter
(rsna_knee.cache_v2). Output: OUT/<study>/<series>.npy plus one manifest row per series (plane, fluid flag, the full
geometry: IOP, per-slice IPP, PixelSpacing, the affine, rows/cols for validity; acquisition metadata; normalization
percentiles). Resumable per series (an existing .npy whose manifest row is present in a previous shard file is skipped
only with --resume). One failing series is logged, never fatal.

    python scripts/build_cache_v2.py --raw data/raw --out data/cache_v2 --manifest-out artifacts/cache_manifests_v2 \\
        --shard 0 --num-shards 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.cache_v2 import decode_series_v2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--split", default="train_series")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N studies of this shard")
    a = ap.parse_args()
    csv = a.raw / ("train_series.csv" if a.split == "train_series" else "test_series.csv")
    sdf = pd.read_csv(csv)
    studies = sorted(sdf.StudyInstanceUID.unique())
    mine = [s for i, s in enumerate(studies) if i % a.num_shards == a.shard]
    if a.limit:
        mine = mine[:a.limit]
    print(f"shard {a.shard}/{a.num_shards}: {len(mine)} studies", file=sys.stderr, flush=True)
    rows, t0 = [], time.time()
    a.manifest_out.mkdir(parents=True, exist_ok=True)
    for n, u in enumerate(mine):
        for s in sdf[sdf.StudyInstanceUID == u].itertuples():
            rec = dict(study_uid=u, series_uid=s.SeriesInstanceUID, plane=s.Anatomical_Plane,
                       fluid=int(s.Fluid_Sensitive), fat_sup=int(getattr(s, "Fat_Suppression", -1)), ok=False, error="")
            try:
                vol, meta = decode_series_v2(a.raw / a.split / u / s.SeriesInstanceUID)
                (a.out / u).mkdir(parents=True, exist_ok=True)
                np.save(a.out / u / f"{s.SeriesInstanceUID}.npy", vol)
                rec.update(ok=True, meta=json.dumps(meta), n_slices=meta["n_slices"],
                           n_bad_slices=int((~np.array(meta["slice_ok"])).sum()))
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"[:2000]
            rows.append(rec)
        if (n + 1) % 20 == 0 or n + 1 == len(mine):
            pd.DataFrame(rows).to_parquet(a.manifest_out / f"shard_{a.shard:03d}.parquet", index=False)
            print(f"  {n + 1}/{len(mine)} studies, {sum(r['ok'] for r in rows)}/{len(rows)} series ok, "
                  f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    pd.DataFrame(rows).to_parquet(a.manifest_out / f"shard_{a.shard:03d}.parquet", index=False)
    print(f"shard {a.shard} done: {sum(r['ok'] for r in rows)}/{len(rows)} series ok", file=sys.stderr)


if __name__ == "__main__":
    main()
