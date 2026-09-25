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

"""Generate notebooks/profile-kaggle-t4.ipynb: MEASURE exp018's per-study decode and model cost on Kaggle's
own hardware (the notebook CPUs + 2 x T4), which the project has so far only projected from H100 profiling
(decode rescaled by worker count, model x3.5). It is a profiling run, not a submission: it reads TRAIN studies
from the competition mount (the commit environment's test set is a 3-row placeholder), uses the SCORED
exp018 code and weights unchanged, and prints a JSON summary. Private, internet off."""
import json
from pathlib import Path

MD = """# exp018 on Kaggle hardware: measured decode vs model cost per study

Profiling only (no submission). Uses the scored `rsna-knee-exp018-code` and `rsna-knee-exp018-weights` unchanged,
on training studies from the competition mount. Measures, separately: decode throughput at 3 and 4 threads;
5-fold model time per study on ONE T4 (inputs pre-decoded); and the scored `run_exp018` end to end on 2 x T4.
"""

C1 = """import importlib.util, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch

def _find_dir(root, name, max_depth=3):
    # the scored notebooks' bounded search: known mount shapes first, then a depth-3 BFS that never
    # descends into DICOM series trees (an unbounded rglob over /kaggle/input once took 84 min)
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
print("CODE", CODE, "| W", W, "| COMP", COMP)
print("competition mount:", sorted(p.name for p in COMP.iterdir()))
init = list(CODE.rglob("__init__.py"))[0].parent
spec = importlib.util.spec_from_file_location("rsna_knee", init / "__init__.py", submodule_search_locations=[str(init)])
m = importlib.util.module_from_spec(spec); sys.modules["rsna_knee"] = m; spec.loader.exec_module(m)
from rsna_knee.infer_wmil import _decode_task, load_fold_models, run_exp018
from rsna_knee.wmil import predict_study
print("infer_wmil.py", (init / "infer_wmil.py").stat().st_size, "B | cpu_count", os.cpu_count(),
      "| sched_getaffinity", len(os.sched_getaffinity(0)), "| gpus", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
"""

C2 = """N = 150
TRAIN_DIR = COMP / "train_series"
assert TRAIN_DIR.is_dir(), "train_series not mounted -- cannot profile on real DICOMs"
sdf = pd.read_csv(COMP / "train_series.csv")
uids = sorted(sdf["StudyInstanceUID"].unique())[:N]
rows = (sdf.rename(columns={"StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid",
                            "Anatomical_Plane": "plane", "Fluid_Sensitive": "fluid"}))
tasks = [(u, rows[rows.study_uid == u][["series_uid", "plane", "fluid"]].to_dict("records"), str(TRAIN_DIR / u)) for u in uids]
R = {"n_studies": N, "cpu_count": os.cpu_count()}

# 1) decode-only throughput (the scored _decode_task), wall-clock per study
for w in (3, 4):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=w) as pool:
        out = list(pool.map(_decode_task, tasks))
    R[f"decode_s_per_study_{w}w"] = (time.time() - t0) / N
    print(f"decode, {w} threads: {R[f'decode_s_per_study_{w}w']:.3f} s/study "
          f"({sum(o[3] is None for o in out)}/{N} ok)", flush=True)
decoded = {o[0]: (o[1], o[2]) for o in out if o[3] is None}
"""

C3 = """# 2) model-only: 5 folds on ONE T4, inputs already decoded (fp16 exactly as the scored driver)
weights = sorted(W.glob("fold[0-4]_best.pt"))
b = load_fold_models(weights, ["cuda:0"])
models, res = b["models"]["cuda:0"], b["res"]
ids = list(decoded)[:60]
for u in ids[:2]:  # warm-up
    st = torch.from_numpy(decoded[u][0]).to("cuda:0")
    for mm in models:
        predict_study(mm, st, decoded[u][1], res, torch.float16)
torch.cuda.synchronize(); t0 = time.time()
for u in ids[2:]:
    st = torch.from_numpy(decoded[u][0]).to("cuda:0")
    for mm in models:
        predict_study(mm, st, decoded[u][1], res, torch.float16)
torch.cuda.synchronize()
R["model_s_per_study_1gpu_5fold"] = (time.time() - t0) / len(ids[2:])
print(f"model, 5 folds on one T4: {R['model_s_per_study_1gpu_5fold']:.3f} s/study", flush=True)
del models, b; torch.cuda.empty_cache()
"""

C4 = """# 3) the scored driver end to end on 2 x T4, DECODE_WORKERS as the scored notebook sets it
wd = Path("/kaggle/working/_prof"); wd.mkdir(exist_ok=True)
pd.DataFrame({"StudyInstanceUID": uids}).to_csv(wd / "test.csv", index=False)
sdf[sdf.StudyInstanceUID.isin(uids)].to_csv(wd / "test_series.csv", index=False)
workers = max(1, min((os.cpu_count() or 4) - 1, 8))
t0 = time.time()
sub = run_exp018(wd / "test.csv", wd / "test_series.csv", TRAIN_DIR, weights, wd / "sub.csv",
                 devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())], num_decode_workers=workers)
R["e2e_s_total"] = time.time() - t0
R["e2e_workers"] = workers
print(f"end to end: {R['e2e_s_total']:.1f} s for {N} studies (includes ~model load)", flush=True)

d, mdl = R["decode_s_per_study_3w"], R["model_s_per_study_1gpu_5fold"]
R["bound_dual_t4"] = "decode" if d > mdl / 2 else "model"
R["projected_1300_min_measured"] = max(d, mdl / 2) * 1300 / 60
print(json.dumps(R, indent=1))
(Path("/kaggle/working") / "profile_t4.json").write_text(json.dumps(R, indent=1))
"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


nb = {"cells": [cell("markdown", MD)] + [cell("code", s) for s in (C1, C2, C3, C4)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent.parent / "notebooks/profile-kaggle-t4.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
