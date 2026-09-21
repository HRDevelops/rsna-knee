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

"""exp018 cache: one uint8 [64, 336, 336] slot-layout stack per STUDY (rsna_knee.wmil.decode_study), 140 mm
field of view, slices from the 6-94% span of each geometrically ordered series. ~7.2 MB/study (~32 GB for 4,407
studies). A per-shard manifest records, per slot, the series used and the geometry/side descriptors needed for
laterality normalisation at load time (so the side rule can change without rebuilding pixels).

Sharded for SLURM arrays like scripts/build_cache.py (index % num_shards == shard). Not resumable per study
(the manifest row needs the decode metadata; a full rebuild is ~10 minutes over 20 shards). One failing study
is logged, never fatal.

    python scripts/build_cache_wmil.py --raw data/raw --out data/cache_wmil \\
        --manifest-out artifacts/cache_manifests_wmil --shard 0 --num-shards 20
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.wmil import NUM_SLOTS, decode_study  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--split", default="train_series", help="train_series (training) or test_series")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N studies of this shard")
    args = ap.parse_args()

    series_csv = args.raw / ("train_series.csv" if args.split == "train_series" else "test_series.csv")
    sdf = pd.read_csv(series_csv).rename(columns={
        "StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid",
        "Anatomical_Plane": "plane", "Fluid_Sensitive": "fluid"})
    root = args.raw / args.split
    studies = sorted(sdf["study_uid"].unique())
    mine = [s for i, s in enumerate(studies) if i % args.num_shards == args.shard]
    if args.limit:
        mine = mine[:args.limit]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} of {len(studies)} studies", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, uid in enumerate(mine):
        out_path = args.out / f"{uid}.npy"
        rec = {"study_uid": uid, "ok": False, "error": ""}
        try:
            grp = sdf[sdf["study_uid"] == uid][["series_uid", "plane", "fluid"]].to_dict("records")
            stack, present, metas = decode_study(root / uid, grp)
            if not present.any():
                raise ValueError("no slot could be decoded")
            np.save(out_path, stack)
            rec["ok"] = True
            for i in range(NUM_SLOTS):
                m = metas[i]
                rec[f"present_{i}"] = bool(present[i])
                rec[f"series_{i}"] = m["series_uid"] if m else ""
                rec[f"n_native_{i}"] = int(m["n_native"]) if m else 0
                rec[f"row_x_{i}"] = float(m["row_x"]) if m else float("nan")
                rec[f"normal_x_{i}"] = float(m["normal_x"]) if m else float("nan")
                rec[f"cx_{i}"] = float(m["cx"]) if m else float("nan")
                rec[f"tag_{i}"] = str(m["tag_laterality"]) if m else ""
                rec[f"order_mismatch_{i}"] = bool(m["order_mismatch"]) if m else False
        except Exception as exc:  # noqa: BLE001 - never kill the shard for one study
            rec["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        rows.append(rec)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(mine)} studies, {sum(r['ok'] for r in rows)} ok", file=sys.stderr)

    args.manifest_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.manifest_out / f"shard_{args.shard:03d}.parquet", index=False)
    n_ok = sum(r["ok"] for r in rows)
    print(f"shard {args.shard} done: {n_ok} ok, {len(rows) - n_ok} failed", file=sys.stderr)


if __name__ == "__main__":
    main()
