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

"""Generate notebooks/submit-blend018.ipynb (probe #4: exp018 x0.75 + deployed ensemble x0.25, rank
space). Re-run after any change to infer_blend018.py."""
import json
from pathlib import Path

MD = """# RSNA Knee — probe #4 (exp018 x0.75 blended with the deployed 3-member ensemble, rank space)

**exp018 scored 0.929 public LB solo (probe #3); the deployed 3-member ensemble scores 0.893.** This
notebook blends the two in rank space at a FIXED weight — **0.75 exp018 / 0.25 ensemble** — chosen
from the LB gap and a per-label OOF diversity check (mean correlation 0.87; see
`artifacts/exp018_probe4_analysis.md`), NOT by maximising an OOF number (exp018's own OOF is known to
undersell it by several points relative to its LB score, the same report-label-proxy gap documented
for every experiment in this project — an OOF-optimal blend weight would be actively misleading here).

Each study is decoded TWICE — the two pipelines use different physical crops/spacing/slot layouts and
share no intermediate representation. Both outputs are converted to a percentile against their own
stored OOF reference distribution (the same empirical-CDF mechanism `rsna-knee-code`'s ensemble
notebook already uses for its rank-mean labels) before blending, so "rank" is well-defined for a
single live study.

**Measured H100 timing, T4-projected (`scripts/profile_decode_vs_model_blend018.py`, 150 real
studies):** ~61 min on T4 x2, ~122 min on T4 x1 — both comfortably inside the 8h Kaggle cap.

Attach: the competition data, `rsna-knee-blend018-code` (full `rsna_knee` package incl.
`infer_blend018.py`), `rsna-knee-blend018-refs` (the two rank references + the recipe yaml),
`rsna-knee-exp018-weights`, and the three deployed members' weight datasets
(`rsna-knee-exp006-weights`, `rsna-knee-exp003-weights`, `rsna-knee-exp005-weights` — each already
carries its own `rank_reference.npz`, used exactly as in the deployed ensemble notebook).
"""

C1 = """import importlib.util
import os, sys, time
from pathlib import Path

import pandas as pd
import torch

def _find_dir(root: str, name_contains: str, max_depth: int = 3) -> Path | None:
    root_path = Path(root)
    if not root_path.exists():
        return None
    candidates = [
        root_path / name_contains,
        root_path / "competitions" / name_contains,
        root_path / "datasets" / name_contains,
        root_path / "datasets" / "hrdevelops" / name_contains,
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    queue = [(root_path, 0)]
    while queue:
        curr, depth = queue.pop(0)
        if depth > 0 and name_contains in curr.name:
            return curr
        if depth < max_depth:
            try:
                for entry in curr.iterdir():
                    if entry.is_dir():
                        if name_contains in entry.name:
                            return entry
                        if entry.name not in ("test_series", "train_series", "series", "images"):
                            queue.append((entry, depth + 1))
            except (PermissionError, OSError):
                pass
    return None

CODE_DIR = _find_dir("/kaggle/input", "rsna-knee-blend018-code")
TEST_DIR = _find_dir("/kaggle/input", "rsna-knee-abnormality-detection")
REFS_DIR = _find_dir("/kaggle/input", "rsna-knee-blend018-refs")
EXP018_WEIGHTS_DIR = _find_dir("/kaggle/input", "rsna-knee-exp018-weights")
OLD_WEIGHTS_ROOT = _find_dir("/kaggle/input", "rsna-knee-exp006-weights")
OLD_WEIGHTS_ROOT = OLD_WEIGHTS_ROOT.parent if OLD_WEIGHTS_ROOT is not None else None  # the dir that HOLDS all 3 member dirs
OUT_PATH = Path("/kaggle/working/submission.csv")
for label, v in (("CODE_DIR", CODE_DIR), ("TEST_DIR", TEST_DIR), ("REFS_DIR", REFS_DIR),
                 ("EXP018_WEIGHTS_DIR", EXP018_WEIGHTS_DIR), ("OLD_WEIGHTS_ROOT", OLD_WEIGHTS_ROOT)):
    print(f"{label}: {v}")
assert CODE_DIR is not None, "Attach rsna-knee-blend018-code"
assert TEST_DIR is not None, "Attach the competition data"
assert REFS_DIR is not None, "Attach rsna-knee-blend018-refs"
assert EXP018_WEIGHTS_DIR is not None, "Attach rsna-knee-exp018-weights"
assert OLD_WEIGHTS_ROOT is not None, "Attach rsna-knee-exp006/003/005-weights"

init_files = list(CODE_DIR.rglob("__init__.py"))
assert init_files, f"No __init__.py found under {CODE_DIR}"
pkg_dir = init_files[0].parent
spec = importlib.util.spec_from_file_location("rsna_knee", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
rsna_knee = importlib.util.module_from_spec(spec)
sys.modules["rsna_knee"] = rsna_knee
spec.loader.exec_module(rsna_knee)

print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
      "devices:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
"""

C2 = """from rsna_knee.infer_blend018 import run_blend018

WEIGHT_EXP018 = 0.75   # 2026-09-22 instruction: fixed by the LB gap (0.929 vs 0.893) + diversity
                       # check, not by OOF argmax -- see artifacts/exp018_probe4_analysis.md

# The 3 deployed members are found individually so OLD_WEIGHTS_ROOT resolves to whichever ancestor
# directory Kaggle actually mounts them under (their own dataset dirs live one level below it).
for name in ("rsna-knee-exp006-weights", "rsna-knee-exp003-weights", "rsna-knee-exp005-weights"):
    assert (OLD_WEIGHTS_ROOT / name).is_dir(), f"missing {name} under {OLD_WEIGHTS_ROOT}"

EXP018_WEIGHTS = sorted(EXP018_WEIGHTS_DIR.glob("fold*_best.pt"))
assert len(EXP018_WEIGHTS) == 5, EXP018_WEIGHTS
print("exp018 weights:", [w.name for w in EXP018_WEIGHTS])

DEVICES = ["cuda:%d" % i for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
DECODE_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))
BUDGET_SECONDS = 8.0 * 3600 * 0.85
print("devices:", DEVICES, "decode workers:", DECODE_WORKERS, "weight(exp018):", WEIGHT_EXP018)
"""

C3 = """t0 = time.time()
sub_df = run_blend018(
    TEST_DIR / "test.csv", TEST_DIR / "test_series.csv", TEST_DIR / "test_series", TEST_DIR / "test_series",
    REFS_DIR / "ensemble_recipe_v1.yaml",
    OLD_WEIGHTS_ROOT, EXP018_WEIGHTS,
    REFS_DIR / "exp018_rank_reference.npz", REFS_DIR / "ensemble_output_rank_reference.npz",
    OUT_PATH, weight_exp018=WEIGHT_EXP018, devices=DEVICES, num_decode_workers=DECODE_WORKERS,
    budget_seconds=BUDGET_SECONDS,
)
print("total runtime %.1f min" % ((time.time() - t0) / 60))
"""

C4 = """from rsna_knee.dataset import TARGET_COLUMNS

test_df = pd.read_csv(TEST_DIR / "test.csv")
sub_df = pd.read_csv(OUT_PATH)[["StudyInstanceUID"] + TARGET_COLUMNS]
assert len(sub_df) == len(test_df), f"row count mismatch: {len(sub_df)} vs {len(test_df)}"
assert set(sub_df["StudyInstanceUID"]) == set(test_df["StudyInstanceUID"]), "study UID mismatch"
assert not sub_df[TARGET_COLUMNS].isna().any().any(), "NaNs in submission"
assert (sub_df[TARGET_COLUMNS] >= 0).all().all() and (sub_df[TARGET_COLUMNS] <= 1).all().all(), "probabilities out of [0,1]"
assert list(sub_df.columns) == ["StudyInstanceUID"] + TARGET_COLUMNS, f"header mismatch: {list(sub_df.columns)}"
sub_df.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH}: {len(sub_df)} rows, all checks passed.")
sub_df.head()
"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


nb = {"cells": [cell("markdown", MD), cell("code", C1), cell("code", C2), cell("code", C3), cell("code", C4)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent.parent / "notebooks/submit-blend018.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
