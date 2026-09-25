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

"""Generate notebooks/submit-exp018-2fold.ipynb: the efficiency-track candidate chosen on 2026-09-26.

Built FROM the scored notebook (notebooks/submit-exp018.ipynb, public 0.929) by replacing exactly one code cell
-- the one that builds the weight list and the decode worker count. Every other cell, and the code and weight
datasets, are the scored ones, unchanged. The scored driver `run_exp018` prob-means whatever list it is given.

  weights       : folds {0, 1} only -- PRE-DECLARED before any 2-fold result was seen, not the best pair on
                  gold-58 (folds {0,1}: 0.9078; mean over all 10 pairs 0.9067; 5-fold 0.9091)
  decode threads: unchanged (the scored cpu_count()-1 = 3); a 4th thread was measured to give no gain on
                  Kaggle (hrdevelops/rsna-knee-profile-2fold, cold cache, disjoint study sets).
"""
import copy
import json
import sys
from pathlib import Path

# Measured 2026-09-26 (hrdevelops/rsna-knee-profile-2fold, cold cache, disjoint study sets): at 2 folds,
# 3 threads 0.934 s/study vs 4 threads 0.952 s/study after removing model-load time -- no gain from a 4th
# thread on Kaggle's 4 vCPUs (two GPU-feeder threads share them). So the scored worker formula is KEPT.
DECODE_WORKERS = None  # None => keep the scored cell's `max(1, min(cpu_count - 1, 8))` line unchanged

root = Path(__file__).resolve().parent.parent
src = json.loads((root / "notebooks/submit-exp018.ipynb").read_text(encoding="utf-8"))
nb = copy.deepcopy(src)
j = lambda c: "".join(c["source"]) if isinstance(c["source"], list) else c["source"]  # noqa: E731

code_idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
target = code_idx[1]  # the weights / devices / workers cell
old = j(nb["cells"][target])
assert 'weights = sorted(WEIGHTS_DIR.glob("fold*_best.pt"))' in old and "DECODE_WORKERS = " in old, old[:200]

new = old.replace(
    'weights = sorted(WEIGHTS_DIR.glob("fold*_best.pt")) or sorted(WEIGHTS_DIR.rglob("best.pt"))',
    '# 2-fold efficiency variant: folds {0, 1}, pre-declared (not chosen on any score). Exactly 2, or stop.\n'
    'weights = [WEIGHTS_DIR / "fold0_best.pt", WEIGHTS_DIR / "fold1_best.pt"]\n'
    'assert all(w.exists() for w in weights), [str(w) for w in weights]')
if DECODE_WORKERS is not None:
    new = new.replace("DECODE_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))", f"DECODE_WORKERS = {DECODE_WORKERS}")
assert new != old and "fold1_best.pt" in new
nb["cells"][target]["source"] = new

md_idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown"][0]
nb["cells"][md_idx]["source"] = (
    "# RSNA Knee — exp018, 2 folds (efficiency track)\n\n"
    "The scored exp018 notebook (public 0.929) with ONE change: it runs folds **{0, 1}** instead of all five, with "
    "the scored decode-thread setting. Measured on Kaggle's 2 x T4, the 5-fold run is model-bound (~53 min per "
    "1,300 studies); each fold costs ~9.7 min, so 2 folds (~20 min) is the efficiency-optimal count. Same scored "
    "code (`rsna-knee-exp018-code`) and weights (`rsna-knee-exp018-weights`).\n\n" + j(nb["cells"][md_idx]))

out = root / "notebooks/submit-exp018-2fold.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

changed = [i for i, (a, b) in enumerate(zip(src["cells"], nb["cells"])) if j(a) != j(b)]
print(f"wrote {out}; cells changed vs scored notebook: {changed} (markdown {md_idx}, weights cell {target})")
