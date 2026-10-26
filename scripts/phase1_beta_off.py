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

"""Post-hoc mechanism diagnostic of the stopped Phase 1 G_raw runs (no search, no rescaling): the trained G at its
learned beta vs the same weights with every beta_h set to 0, dense, fold-0 val. Reports mean |p_G - p_beta0|, the
macro-AUC difference, and the distribution of the actual cross-plane bias -beta_h d^2 / 2 over a sample of studies.

    RSNA_ROOT=... python scripts/phase1_beta_off.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from train import expand_env, load_soft_truth, macro_auc  # noqa: E402
from train_tokens import predict, to_dev  # noqa: E402
from rsna_knee.tokens import TokenMIL  # noqa: E402
from rsna_knee.wmil_v2 import TokenV2Dataset, collate_tokens, load_manifest_v2  # noqa: E402

A = Path(os.environ["RSNA_ROOT"]) / "artifacts"
EXP = sys.argv[1] if len(sys.argv) > 1 else "p1_graw"   # p2_glocal: local follow-up
out = []
for s in (42, 43):
    cfg = expand_env(yaml.safe_load((A / f"{EXP}_s{s}/fold0/config.yaml").read_text()))
    d, m = cfg["data"], cfg["model"]
    man = load_manifest_v2(Path(d["manifest_dir"]))
    side = pd.read_csv(d["final_side"]).set_index("study_uid")["final_side"].to_dict()
    folds = pd.read_parquet(d["folds"])
    val = sorted(set(folds[folds.fold == 0].study_uid) & set(man.study_uid))
    model = TokenMIL(m["arch"], grid=6, geometry="cross_plane", pretrained=False, grad_checkpointing=False,
                     features=m.get("features", "final")).cuda()
    model.load_state_dict(torch.load(A / f"{EXP}_s{s}/fold0/best.pt", map_location="cuda")["model"])
    beta = model.beta.detach().clone()
    ds = TokenV2Dataset(val, d["cache_dir"], man, None, side, set(), False, 12, grid=6, mode="dense", geometry=True,
                        **d["v2"])
    loader = lambda: DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate_tokens)  # noqa: E731
    pg = predict(model, loader(), m["res"], "cuda", torch.bfloat16).set_index("study_uid").sort_index()
    with torch.no_grad():
        model.beta.zero_()
    p0 = predict(model, loader(), m["res"], "cuda", torch.bfloat16).set_index("study_uid").sort_index()
    truth = load_soft_truth(Path(d["labels"]))
    ag, _ = macro_auc(pg.reset_index(), truth)
    a0, _ = macro_auc(p0.reset_index(), truth)
    diff = (pg - p0).abs().values
    # actual cross-plane bias values -beta_h d^2 / 2 (layer 0 and 1, all heads) on 20 studies
    vals = []
    for i in range(0, len(val), max(1, len(val) // 20)):
        it = ds[i]
        d2, cross = model.footprint_d2(it["mu"][None].cuda(), it["sigma"][None].cuda(), it["plane"][None].cuda())
        x = d2[cross].flatten()
        x = x[torch.randperm(len(x), device=x.device)[:20000]]
        for li in range(beta.shape[0]):
            vals.append((-0.5 * beta[li][:, None] * x[None]).flatten().cpu().numpy())
    v = np.concatenate(vals)
    q = np.percentile(v, [1, 5, 25, 50, 75, 95])
    print(f"seed {s}: macro-AUC G {ag:.4f} vs beta=0 {a0:.4f} (diff {ag - a0:+.4f}); mean |p_G - p_b0| {diff.mean():.2e}, "
          f"max {diff.max():.4f}; cross-plane bias -beta d2/2 percentiles 1/5/25/50/75/95: {np.round(q, 3).tolist()}",
          flush=True)
    out.append(dict(seed=s, auc_g=ag, auc_beta0=a0, mean_abs_dp=float(diff.mean()), max_abs_dp=float(diff.max()),
                    bias_pct=json.dumps(np.round(q, 4).tolist())))
pd.DataFrame(out).to_csv(A / f"{EXP}_beta_off_diagnostic.csv", index=False)
