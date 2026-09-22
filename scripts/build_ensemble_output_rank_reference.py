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

"""Probe #4: builds a rank_reference.npz for the DEPLOYED ENSEMBLE'S OWN OUTPUT (not a real training
run — a synthetic "member" whose per-label distribution is the ensemble_recipe_v1.yaml blend applied
to every non-gold OOF study), so the outer exp018-vs-ensemble blend can use the same empirical-CDF
mechanism (infer_core.empirical_percentile) the 3-member ensemble already uses internally.

Per-study composite = the recipe's per-label method (prob-mean or rank-mean) applied to the 3
members' OOF: for a rank-mean label, each member's value is its OWN population rank within this OOF
set (the population IS the reference at build time, so this is exactly the offline analogue of
looking a live prediction up against each member's stored rank_reference.npz).

    python scripts/build_ensemble_output_rank_reference.py --out-name ensemble_v1_blend_output
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import TARGETS, load_full_oof  # noqa: E402

MEMBERS = ["exp006_dinov2b_slot_v3labels", "exp003_dinov2b_slot_v2labels", "exp005_dinov2s_slot_v3labels"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, default=None)
    ap.add_argument("--out-name", default="ensemble_v1_blend_output")
    a = ap.parse_args()
    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    art = root / "artifacts"
    recipe_path = a.recipe or (root / "code/src/rsna_knee/ensemble_recipe_v1.yaml")
    recipe = yaml.safe_load(recipe_path.read_text())["recipe"]

    oof = {m: load_full_oof(art / m) for m in MEMBERS}
    uids = sorted(set.intersection(*[set(oof[m].index) for m in MEMBERS]))
    P = {m: oof[m].loc[uids, TARGETS].to_numpy() for m in MEMBERS}          # [N, 12]
    R = {m: np.stack([rankdata(P[m][:, i]) / len(uids) for i in range(12)], axis=1) for m in MEMBERS}

    out = np.zeros((len(uids), len(TARGETS)), dtype=np.float32)
    for i, label in enumerate(TARGETS):
        spec = recipe[label]
        method = spec["method"]
        members = spec["members"]
        src = R if method == "rank-mean" else P
        out[:, i] = np.mean([src[m][:, i] for m in members], axis=0)

    out_dir = art / a.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {t: np.sort(out[:, i].astype(np.float32)) for i, t in enumerate(TARGETS)}
    np.savez(out_dir / "rank_reference.npz", **arrays)
    print(f"wrote {out_dir / 'rank_reference.npz'} ({len(uids)} studies per label, recipe {recipe_path.name})")


if __name__ == "__main__":
    main()
