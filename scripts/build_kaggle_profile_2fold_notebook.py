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

"""Generate notebooks/profile-kaggle-2fold.ipynb: end-to-end runtime of the SCORED exp018 driver with the
pre-declared 2-fold subset (folds 0, 1), at 3 vs 4 decode threads, on Kaggle's own 2 x T4 machine.

At 2 folds the run is decode-bound, so the thread count matters, and the first profiling run could not settle
it: its 4-thread decode ran second on files already in the page cache. Here each configuration runs on its
OWN disjoint set of 150 training studies, so both start cold (each Kaggle run is a fresh container). Profiling
only, not a submission."""
import json
from pathlib import Path

MD = """# exp018, 2 folds (0, 1): measured end-to-end runtime at 3 vs 4 decode threads, cold cache each

Profiling only. The scored `run_exp018` and scored weights, unchanged, on two DISJOINT sets of 150 training studies
(so neither configuration benefits from the other's page cache).
"""

C1 = """import importlib.util, json, os, sys, time
from pathlib import Path
import pandas as pd, torch

def _find_dir(root, name, max_depth=3):
    r = Path(root)
    for c in (r / name, r / "competitions" / name, r / "datasets" / name, r / "datasets" / "hrdevelops" / name):
        if c.is_dir():
            return c
    queue = [(r, 0)]
    while queue:
        cur, dep = queue.pop(0)
        if dep < max_depth:
            try:
                for e in cur.iterdir():
                    if e.is_dir():
                        if name in e.name:
                            return e
                        if e.name not in ("test_series", "train_series", "series", "images"):
                            queue.append((e, dep + 1))
            except (PermissionError, OSError):
                pass
    return None

CODE = _find_dir("/kaggle/input", "rsna-knee-exp018-code")
W = _find_dir("/kaggle/input", "rsna-knee-exp018-weights")
COMP = _find_dir("/kaggle/input", "rsna-knee-abnormality-detection")
init = list(CODE.rglob("__init__.py"))[0].parent
spec = importlib.util.spec_from_file_location("rsna_knee", init / "__init__.py", submodule_search_locations=[str(init)])
m = importlib.util.module_from_spec(spec); sys.modules["rsna_knee"] = m; spec.loader.exec_module(m)
from rsna_knee.infer_wmil import run_exp018
weights = [W / "fold0_best.pt", W / "fold1_best.pt"]
assert all(w.exists() for w in weights), weights
print("infer_wmil.py", (init / "infer_wmil.py").stat().st_size, "B | cpus", os.cpu_count(), "| gpus", torch.cuda.device_count())
"""

C2 = """sdf = pd.read_csv(COMP / "train_series.csv")
uids = sorted(sdf["StudyInstanceUID"].unique())
sets = {3: uids[200:350], 4: uids[350:500]}   # disjoint, and disjoint from the first profiling run's [:150]
R = {}
for workers, us in sets.items():
    wd = Path(f"/kaggle/working/_p{workers}"); wd.mkdir(exist_ok=True)
    pd.DataFrame({"StudyInstanceUID": us}).to_csv(wd / "test.csv", index=False)
    sdf[sdf.StudyInstanceUID.isin(us)].to_csv(wd / "test_series.csv", index=False)
    t0 = time.time()
    sub = run_exp018(wd / "test.csv", wd / "test_series.csv", COMP / "train_series", weights, wd / "sub.csv",
                     devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())], num_decode_workers=workers)
    dt = time.time() - t0
    R[f"{workers}_threads_s_total"] = dt
    R[f"{workers}_threads_s_per_study"] = dt / len(us)
    R[f"{workers}_threads_min_per_1300"] = dt / len(us) * 1300 / 60
    print(f"2 folds, {workers} decode threads: {dt:.1f} s for {len(us)} studies -> {dt/len(us):.3f} s/study "
          f"-> {dt/len(us)*1300/60:.1f} min per 1300 (incl. model load)", flush=True)
print(json.dumps(R, indent=1))
(Path("/kaggle/working") / "profile_2fold.json").write_text(json.dumps(R, indent=1))
"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


nb = {"cells": [cell("markdown", MD), cell("code", C1), cell("code", C2)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent.parent / "notebooks/profile-kaggle-2fold.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
