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

"""Generate notebooks/submit-blend-v5.ipynb and notebooks/profile-blend-v5.ipynb (offline blend gate PASSED at
alpha = 0.2, docs/EXPERIMENTS.md 2026-09-26).

The parent notebook is CPU-only and never imports torch. Each arm runs as its own `python` child process, built
from the SCORED notebook's code cells with the minimum substitutions (asserted to apply exactly once):
  - CoAt child  <- notebooks/submit-exp021.ipynb (== Kaggle `submit-exp021` v1, scored 0.929): OUT_PATH only.
  - slot child  <- notebooks/ensamble-notebook-scored.ipynb (Kaggle `ensamble-notebook`, scored 0.893): OUT_PATH,
                   and the code dataset name -> `rsna-knee-code-0916` (the 2026-09-16 bytes, commit d767fdc).
The profiling notebook additionally points each child's TEST_DIR at a fake test set of training studies.

    python scripts/build_blend_v5_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALPHA = 0.2                      # slot weight, from the pre-registered gate (deploy alpha)
PRIOR_TOL_FRAC = 0.005           # prior-fallback rows tolerated per child: 0 on the 3-row placeholder, <= 6 of 1,300
OUT_LINE = 'OUT_PATH = Path("/kaggle/working/submission.csv")'
TEST_LINE = 'TEST_DIR = _find_dir("/kaggle/input", "rsna-knee-abnormality-detection")'


def code_cells(nb: Path) -> str:
    d = json.loads(nb.read_text(encoding="utf-8"))
    return "\n\n".join("".join(c["source"]) for c in d["cells"] if c["cell_type"] == "code") + "\n"


def sub1(src: str, old: str, new: str) -> str:
    assert src.count(old) == 1, f"expected exactly one {old!r}, found {src.count(old)}"
    return src.replace(old, new)


def child_sources(profile: bool) -> tuple[str, str]:
    coat = code_cells(ROOT / "notebooks/submit-exp021.ipynb")
    slot = code_cells(ROOT / "notebooks/ensamble-notebook-scored.ipynb")
    coat = sub1(coat, OUT_LINE, 'OUT_PATH = Path(os.environ["BLEND_CHILD_OUT"])')
    slot = sub1(slot, OUT_LINE, 'OUT_PATH = Path(os.environ["BLEND_CHILD_OUT"])')
    slot = sub1(slot, 'CODE_DIR = _find_dir("/kaggle/input", "rsna-knee-code")',
                'CODE_DIR = _find_dir("/kaggle/input", "rsna-knee-code-0916")')
    if profile:
        coat = sub1(coat, TEST_LINE, TEST_LINE + '\nTEST_DIR = Path(os.environ["BLEND_PROFILE_TEST_DIR"])')
        slot = sub1(slot, TEST_LINE, TEST_LINE + '\nTEST_DIR = Path(os.environ["BLEND_PROFILE_TEST_DIR"])')
    return coat, slot


RUNNER = '''import os, signal, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
WORK = Path("/kaggle/working")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]
# Label priors written by BOTH scored drivers on a failed study (infer_wmil.LABEL_PRIORS == infer_core.LABEL_PRIORS)
PRIOR = np.array([0.1881, 0.0915, 0.3872, 0.2336, 0.2430, 0.1457, 0.3909, 0.3826, 0.3894, 0.1912, 0.2086, 0.1321])
ALPHA = __ALPHA__
PRIOR_TOL_FRAC = __TOL__
GRACE_S = 180            # a child still alive this long after its CSV is complete and unchanged is hung: kill it
POLL_S = 15


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


def validate(path: Path, uids: list) -> dict:
    """basic = complete and well-formed; ok = basic and prior-fallback rows within tolerance."""
    r = {"basic": False, "ok": False, "n_prior": None, "why": ""}
    if not path.exists():
        r["why"] = "no CSV"
        return r
    try:
        df = pd.read_csv(path)
    except Exception as e:
        r["why"] = f"unreadable: {type(e).__name__}"
        return r
    if list(df.columns) != ["StudyInstanceUID"] + TARGETS:
        r["why"] = "header mismatch"
        return r
    if len(df) != len(uids) or df["StudyInstanceUID"].duplicated().any() or set(df["StudyInstanceUID"]) != set(uids):
        r["why"] = f"rows/UIDs mismatch ({len(df)} rows vs {len(uids)})"
        return r
    v = df[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(v).all() or (v < 0).any() or (v > 1).any():
        r["why"] = "NaN or out-of-range values"
        return r
    r["basic"] = True
    r["n_prior"] = int(np.all(np.abs(v - PRIOR) < 1e-6, axis=1).sum())
    tol = int(PRIOR_TOL_FRAC * len(df))
    r["ok"] = r["n_prior"] <= tol
    r["why"] = f"{len(df)} rows, {r['n_prior']} on prior (tolerance {tol})"
    return r


def _kill(p):
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            pass


def run_child(name: str, src: str, out_csv: Path, timeout_s: float, uids: list, extra_env=None) -> dict:
    script, log = WORK / f"child_{name}.py", WORK / f"child_{name}.log"
    script.write_text(src)
    if out_csv.exists():
        out_csv.unlink()
    env = dict(os.environ, BLEND_CHILD_OUT=str(out_csv), PYTHONUNBUFFERED="1", **(extra_env or {}))
    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.Popen([sys.executable, "-u", str(script)], stdout=lf, stderr=subprocess.STDOUT, env=env,
                             cwd=str(WORK), start_new_session=True)
    pos, seen_mtime, stable_since, end = 0, None, None, None
    print(f"[{name}] started pid {p.pid}, timeout {timeout_s / 3600:.2f} h", flush=True)
    while True:
        rc = p.poll()
        with open(log, errors="replace") as lf:              # stream the child's log into this notebook's log
            lf.seek(pos)
            chunk = lf.read()
            pos = lf.tell()
        for line in chunk.splitlines():
            print(f"  [{name}] {line}", flush=True)
        if rc is not None:
            end = f"exit code {rc}"
            break
        if time.time() - t0 > timeout_s:
            _kill(p)
            end = "TIMEOUT, killed"
            break
        if out_csv.exists() and validate(out_csv, uids)["basic"]:
            m = out_csv.stat().st_mtime
            if m != seen_mtime:
                seen_mtime, stable_since = m, time.time()
            elif time.time() - stable_since > GRACE_S:
                _kill(p)
                end = f"alive {GRACE_S}s after writing a complete CSV, killed"
                break
        time.sleep(POLL_S)
    v = validate(out_csv, uids)
    v.update(name=name, end=end, minutes=(time.time() - t0) / 60)
    print(f"[{name}] {end} after {v['minutes']:.1f} min | CSV: {v['why']} | basic={v['basic']} ok={v['ok']}", flush=True)
    return v
'''


def parent_notebook(coat_src: str, slot_src: str) -> dict:
    md = f"""# Blend v5: exp021 (0.929) + scored slot ensemble (0.893), subprocess-isolated

Offline gate passed 2026-09-26 (pre-registered, nested CV): slot weight **alpha = {ALPHA}**, same for all 12 labels,
per-label ranks within the test batch. This notebook is CPU-only and never imports torch; each arm runs its
scored code as a separate `python` process:
- **CoAt child**: `submit-exp021` v1 code cells (scored 0.929), unchanged except the output path.
- **slot child**: `ensamble-notebook` code cells (scored 0.893), unchanged except the output path and the code
  dataset, `rsna-knee-code-0916` (the 2026-09-16 bytes that produced the scored run).

A child still alive {180}s after its CSV is complete is killed and its CSV used. Each CSV is validated (rows, UIDs,
no NaN, range, prior-fallback rows). Fallbacks: slot fails -> exp021 output unchanged; CoAt fails -> slot output.
"""
    c1 = RUNNER.replace("__ALPHA__", repr(ALPHA)).replace("__TOL__", repr(PRIOR_TOL_FRAC)) + f'''
COAT_SRC = {coat_src!r}
SLOT_SRC = {slot_src!r}
TEST_DIR = _find_dir("/kaggle/input", "rsna-knee-abnormality-detection")
assert TEST_DIR is not None, "Attach the competition data"
test_df = pd.read_csv(TEST_DIR / "test.csv")
UIDS = list(test_df["StudyInstanceUID"])
DEADLINE = T0 + 8.25 * 3600      # Kaggle's cap is 9 h; both children must be done by here
print(f"{{len(UIDS)}} test studies | alpha {{ALPHA}} | torch imported in parent: {{'torch' in sys.modules}}")
'''
    c2 = '''coat = run_child("coat", COAT_SRC, WORK / "preds_coat.csv", min(5 * 3600, DEADLINE - time.time() - 900), UIDS)
slot = run_child("slot", SLOT_SRC, WORK / "preds_slot.csv", max(600, DEADLINE - time.time() - 900), UIDS)
'''
    c3 = '''def _load(path):
    return pd.read_csv(path).set_index("StudyInstanceUID").loc[UIDS, TARGETS]

mode = None
try:
    if coat["ok"] and slot["ok"]:
        rc = _load(WORK / "preds_coat.csv").rank(pct=True, method="average")
        rs = _load(WORK / "preds_slot.csv").rank(pct=True, method="average")
        out, mode = (1 - ALPHA) * rc + ALPHA * rs, f"blend (alpha {ALPHA}, test-batch ranks)"
except Exception as e:
    print(f"[blend] failed ({type(e).__name__}: {e}); falling back", flush=True)
if mode is None:
    for cond, arm, label in ((coat["ok"], "coat", "exp021 only (slot failed)"),
                             (slot["ok"], "slot", "slot only (CoAt failed)"),
                             (coat["basic"], "coat", "exp021 only (both over prior tolerance)"),
                             (slot["basic"], "slot", "slot only (both over prior tolerance)")):
        if cond:
            out, mode = _load(WORK / f"preds_{arm}.csv"), label
            break
if mode is None:
    out, mode = pd.DataFrame([PRIOR] * len(UIDS), index=UIDS, columns=TARGETS), "ALL PRIOR (both children failed)"

sub = out.reset_index().rename(columns={"index": "StudyInstanceUID"})[["StudyInstanceUID"] + TARGETS]
assert len(sub) == len(test_df) and set(sub["StudyInstanceUID"]) == set(UIDS)
assert not sub[TARGETS].isna().any().any() and ((sub[TARGETS] >= 0) & (sub[TARGETS] <= 1)).all().all()
sub.to_csv(WORK / "submission.csv", index=False)
assert "torch" not in sys.modules, "parent must never import torch"
print(f"[blend-v5] wrote submission.csv: {len(sub)} rows | mode: {mode} | coat: {coat['why']} ({coat['end']}, "
      f"{coat['minutes']:.1f} min) | slot: {slot['why']} ({slot['end']}, {slot['minutes']:.1f} min) | "
      f"total {(time.time() - T0) / 60:.1f} min", flush=True)
sub.head()
'''
    return nb([cell("markdown", md), cell("code", c1), cell("code", c2), cell("code", c3)])


def profile_notebook(coat_src: str, slot_src: str) -> dict:
    md = """# Blend v5 runtime profile (not a submission)

Runs the two blend-v5 children exactly as the submission does (same scored code, same subprocess runner), each on
its own DISJOINT set of 120 training studies posing as the test set (cold page cache for each), and projects the
full ~1,300-study runtime. Model loading is included in each child's time.
"""
    c1 = RUNNER.replace("__ALPHA__", repr(ALPHA)).replace("__TOL__", repr(PRIOR_TOL_FRAC)) + f'''
import json
COAT_SRC = {coat_src!r}
SLOT_SRC = {slot_src!r}
COMP = _find_dir("/kaggle/input", "rsna-knee-abnormality-detection")
sdf = pd.read_csv(COMP / "train_series.csv")
uids = sorted(sdf["StudyInstanceUID"].unique())
SETS = {{"coat": uids[500:620], "slot": uids[620:740]}}   # disjoint from each other and from earlier profiling runs
'''
    c2 = '''R = {}
for name, src in (("coat", COAT_SRC), ("slot", SLOT_SRC)):
    us = SETS[name]
    d = Path("/tmp") / f"_fake_{name}"; d.mkdir(exist_ok=True)   # not under /kaggle/working: outputs must not follow the symlink
    pd.DataFrame({"StudyInstanceUID": us}).to_csv(d / "test.csv", index=False)
    sdf[sdf.StudyInstanceUID.isin(us)].to_csv(d / "test_series.csv", index=False)
    if not (d / "test_series").exists():
        (d / "test_series").symlink_to(COMP / "train_series")
    v = run_child(name, src, WORK / f"preds_{name}.csv", 3 * 3600, us, {"BLEND_PROFILE_TEST_DIR": str(d)})
    R[name] = {"n": len(us), "minutes": v["minutes"], "s_per_study": v["minutes"] * 60 / len(us),
               "min_per_1300": v["minutes"] / len(us) * 1300, "end": v["end"], "csv": v["why"], "ok": v["ok"]}
R["total_min_per_1300"] = R["coat"]["min_per_1300"] + R["slot"]["min_per_1300"]
R["total_h_per_1300"] = R["total_min_per_1300"] / 60
print(json.dumps(R, indent=1))
(WORK / "profile_blend_v5.json").write_text(json.dumps(R, indent=1))
assert "torch" not in sys.modules
'''
    return nb([cell("markdown", md), cell("code", c1), cell("code", c2)])


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    for profile, name, build in ((False, "submit-blend-v5", parent_notebook), (True, "profile-blend-v5", profile_notebook)):
        coat, slot = child_sources(profile)
        for s in (coat, slot):
            compile(s, "<child>", "exec")
        out = ROOT / f"notebooks/{name}.ipynb"
        out.write_text(json.dumps(build(coat, slot), indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
