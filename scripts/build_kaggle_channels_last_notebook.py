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

"""Generate notebooks/profile-kaggle-channels-last.ipynb: does channels_last speed up exp018's model on Kaggle's
T4 while keeping predictions equal to the scored run (float tolerance)? Measurement only, no submission.

200 TRAIN studies are decoded once with the scored `_decode_task`, then all 5 exp018 folds run on ONE T4 in:
  A  the scored path: wmil.predict_study unchanged (fp16 autocast, NCHW)              <- the reference
  B  channels_last: model weights and each window batch in channels_last, else identical to A
  C  B + torch.backends.cudnn.benchmark = True                                          (information only)
Reports s/study (5 folds) per variant and, vs A, the max |dprob| and per-label Spearman over the 200 studies."""
import json
from pathlib import Path

MD = """# exp018 on T4: scored path vs channels_last (speed and prediction match)

Measurement only. Scored `rsna-knee-exp018-code` and weights, unchanged; 200 training studies from the competition
mount, decoded once; all 5 folds on ONE T4, fp16 autocast exactly as the scored driver.
"""

C1 = """import importlib.util, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch

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
from rsna_knee.infer_wmil import _decode_task, load_fold_models
from rsna_knee import wmil
print("infer_wmil.py", (init / "infer_wmil.py").stat().st_size, "B | gpu", torch.cuda.get_device_name(0))

sdf = pd.read_csv(COMP / "train_series.csv")
uids = sorted(sdf["StudyInstanceUID"].unique())[600:800]      # disjoint from the earlier profiling runs
rows = sdf.rename(columns={"StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid",
                           "Anatomical_Plane": "plane", "Fluid_Sensitive": "fluid"})
tasks = [(u, rows[rows.study_uid == u][["series_uid", "plane", "fluid"]].to_dict("records"), str(COMP / "train_series" / u)) for u in uids]
with ThreadPoolExecutor(max_workers=3) as pool:
    dec = [o for o in pool.map(_decode_task, tasks) if o[3] is None]
print(f"decoded {len(dec)}/{len(uids)} studies", flush=True)
"""

C2 = """def predict_cl(model, stack_u8, present, res, dtype, chunk=64):
    # wmil.predict_study with ONE change: each window batch is converted to channels_last before the backbone
    dev = stack_u8.device
    cs = wmil.candidate_centers(present)
    centers = torch.from_numpy(cs).to(dev)[None]
    feats = []
    for s in range(0, centers.shape[1], chunk):
        w = wmil.gather_windows(stack_u8[None], centers[:, s:s + chunk])
        x = wmil.prepare_input(w, res).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=dev.type, dtype=dtype):
            feats.append(model.encode(x.view(1, -1, 3, res, res)).float())
    feats = torch.cat(feats, dim=1)
    with torch.autocast(device_type=dev.type, enabled=False):
        return model.head(feats)[0]

def run(models, fn, n_warm=3):
    out = {}
    for u, st_np, pres, _ in dec[:n_warm]:                      # warm-up, not timed
        st = torch.from_numpy(st_np).to("cuda:0")
        for mm in models:
            fn(mm, st, pres, res, torch.float16)
    torch.cuda.synchronize(); t0 = time.time()
    for u, st_np, pres, _ in dec:
        st = torch.from_numpy(st_np).to("cuda:0")
        ps = []
        for mm in models:
            lg = fn(mm, st, pres, res, torch.float16)
            if not torch.isfinite(lg).all():
                lg = fn(mm, st, pres, res, torch.float32)
            ps.append(torch.sigmoid(lg.float()).cpu().numpy())
        out[u] = np.mean(ps, 0)
    torch.cuda.synchronize()
    return out, (time.time() - t0) / len(dec)

weights = sorted(W.glob("fold[0-4]_best.pt"))
b = load_fold_models(weights, ["cuda:0"]); models, res = b["models"]["cuda:0"], b["res"]
R = {"n": len(dec)}
with torch.no_grad():
    pa, R["A_scored_s_per_study_5fold"] = run(models, wmil.predict_study)
    print(f"A scored       : {R['A_scored_s_per_study_5fold']:.3f} s/study (5 folds, one T4)", flush=True)
    models_cl = [mm.to(memory_format=torch.channels_last) for mm in models]
    pb, R["B_channels_last_s_per_study_5fold"] = run(models_cl, predict_cl)
    print(f"B channels_last: {R['B_channels_last_s_per_study_5fold']:.3f} s/study", flush=True)
    torch.backends.cudnn.benchmark = True
    pc, R["C_cl_benchmark_s_per_study_5fold"] = run(models_cl, predict_cl)
    print(f"C + benchmark  : {R['C_cl_benchmark_s_per_study_5fold']:.3f} s/study", flush=True)

from scipy.stats import spearmanr
cols = wmil.TARGET_COLUMNS
A = np.stack([pa[u] for u in pa])
for name, pp in (("B", pb), ("C", pc)):
    X = np.stack([pp[u] for u in pa])
    R[f"{name}_max_abs_dprob"] = float(np.abs(X - A).max())
    R[f"{name}_mean_abs_dprob"] = float(np.abs(X - A).mean())
    R[f"{name}_min_label_spearman"] = float(min(spearmanr(X[:, j], A[:, j]).correlation for j in range(len(cols))))
print(json.dumps(R, indent=1))
(Path("/kaggle/working") / "profile_channels_last.json").write_text(json.dumps(R, indent=1))
"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


nb = {"cells": [cell("markdown", MD), cell("code", C1), cell("code", C2)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent.parent / "notebooks/profile-kaggle-channels-last.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
