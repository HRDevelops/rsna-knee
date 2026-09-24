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

"""Generate notebooks/submit-exp021.ipynb: the 10-checkpoint seed ensemble (exp018 seed 42 + exp021 seed 43).

The only difference from notebooks/submit-exp018.ipynb (the scored 0.929 notebook) is how the weight list is
built: it now comes from TWO weight datasets and is asserted to be exactly 5 + 5. The inference code is the
SCORED `rsna-knee-exp018-code` dataset, unchanged -- its `run_exp018` takes a list of any length and
prob-means every model in it, so no library code is edited for this submission. No hard-exit tail
(docs/SUBMISSION_FINAL.md §4: os._exit makes Kaggle's papermill runner mark the run ERROR)."""
import json
from pathlib import Path

MD = """# RSNA Knee — exp021 submission: 10-checkpoint seed ensemble (exp018 × 2 seeds)

**exp018's model family, trained twice.** 5 folds with seed 42 (the scored exp018 checkpoints, public LB 0.929)
plus 5 folds with seed 43 (exp021: the identical recipe, only the seed changed). The 10 checkpoints'
probabilities are averaged. Same preprocessing, same windows, same driver as the scored exp018 notebook.

Attach: the competition data, **`rsna-knee-exp018-code`** (the scored inference code, unchanged),
**`rsna-knee-exp018-weights`** (the scored seed-42 checkpoints, unchanged) and **`rsna-knee-exp021-weights`**
(the five seed-43 checkpoints). Main-track candidate only; the efficiency-track pick stays exp018.
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

CODE_DIR = _find_dir("/kaggle/input", "rsna-knee-exp018-code")
TEST_DIR = _find_dir("/kaggle/input", "rsna-knee-abnormality-detection")
WEIGHTS_S42 = _find_dir("/kaggle/input", "rsna-knee-exp018-weights")
WEIGHTS_S43 = _find_dir("/kaggle/input", "rsna-knee-exp021-weights")
OUT_PATH = Path("/kaggle/working/submission.csv")
for label, v in (("CODE_DIR", CODE_DIR), ("TEST_DIR", TEST_DIR), ("WEIGHTS_S42", WEIGHTS_S42), ("WEIGHTS_S43", WEIGHTS_S43)):
    print(f"{label}: {v}")
assert CODE_DIR is not None, "Attach rsna-knee-exp018-code"
assert TEST_DIR is not None, "Attach the competition data"
assert WEIGHTS_S42 is not None, "Attach rsna-knee-exp018-weights"
assert WEIGHTS_S43 is not None, "Attach rsna-knee-exp021-weights"

init_files = list(CODE_DIR.rglob("__init__.py"))
assert init_files, f"No __init__.py found under {CODE_DIR}"
pkg_dir = init_files[0].parent
spec = importlib.util.spec_from_file_location("rsna_knee", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
rsna_knee = importlib.util.module_from_spec(spec)
sys.modules["rsna_knee"] = rsna_knee
spec.loader.exec_module(rsna_knee)

import timm
print("torch", torch.__version__, "timm", timm.__version__, "cuda:", torch.cuda.is_available(),
      "devices:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("infer_wmil.py size:", (pkg_dir / "infer_wmil.py").stat().st_size, "B (scored version: 8245 B)")
"""

C2 = """from rsna_knee.infer_wmil import run_exp018
from rsna_knee.wmil import TARGET_COLUMNS, WindowMIL

_probe = WindowMIL("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False)
print("arch ok, params %.1fM" % (sum(p.numel() for p in _probe.parameters()) / 1e6))
del _probe

# Exactly 5 seed-42 + 5 seed-43 checkpoints, or stop: a missing attachment must never silently turn this
# into a 5-model run that looks like the ensemble.
w42 = sorted(WEIGHTS_S42.glob("fold[0-4]_best.pt"))
w43 = sorted(WEIGHTS_S43.glob("fold[0-4]_seed43_best.pt"))
print("seed 42:", [w.name for w in w42])
print("seed 43:", [w.name for w in w43])
assert len(w42) == 5 and len(w43) == 5, f"expected 5 + 5 checkpoints, got {len(w42)} + {len(w43)}"
weights = w42 + w43

DEVICES = ["cuda:%d" % i for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
DECODE_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))
BUDGET_SECONDS = 8.0 * 3600 * 0.85     # leave margin under the 9h Kaggle cap; the guard drops models, never studies
print("devices:", DEVICES, "decode workers:", DECODE_WORKERS, "checkpoints:", len(weights))
"""

C3 = """t0 = time.time()
sub_df = run_exp018(
    TEST_DIR / "test.csv", TEST_DIR / "test_series.csv", TEST_DIR / "test_series", weights, OUT_PATH,
    devices=DEVICES, num_decode_workers=DECODE_WORKERS, budget_seconds=BUDGET_SECONDS,
)
print("total runtime %.1f min" % ((time.time() - t0) / 60))
"""

C4 = """test_df = pd.read_csv(TEST_DIR / "test.csv")
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
out = Path(__file__).resolve().parent.parent / "notebooks/submit-exp021.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
