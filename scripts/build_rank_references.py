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

"""Builds each ensemble member's per-label sorted-OOF-score reference array, used by
infer_core.ensemble_blend's empirical_percentile() to make "rank-mean" blending well-defined at
single-study inference time (see infer_core.py docstring — offline rank-mean is a population
statistic, meaningless for N=1 without a stored reference to rank against). One small .npz per
member, bundled into its Kaggle weights dataset alongside the fold checkpoints.

Usage:
    python scripts/build_rank_references.py --experiments exp006_dinov2b_slot_v3labels exp003_dinov2b_slot_v2labels exp005_dinov2s_slot_v3labels
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def load_full_oof(exp_dir: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(exp_dir.glob("fold*/oof.parquet"))]
    assert len(parts) == 5
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", required=True)
    args = parser.parse_args()

    rsna_root = Path(os.environ.get("RSNA_ROOT", "."))
    root = rsna_root / "artifacts"

    for name in args.experiments:
        oof = load_full_oof(root / name)
        arrays = {t: np.sort(oof[t].to_numpy().astype(np.float32)) for t in TARGETS}
        out_path = root / name / "rank_reference.npz"
        np.savez(out_path, **arrays)
        print(f"{name}: wrote {out_path} ({len(oof)} studies per label)")


if __name__ == "__main__":
    main()
