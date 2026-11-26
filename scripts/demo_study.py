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

"""Course demo: run the efficiency-track model (exp018 folds 0 + 1, as submitted) on three gold-58 studies, which no
model ever trained on or was selected on, and save everything the demo video shows.

Study selection is a fixed rule on the expert labels only, written before any prediction was looked at:
order the gold studies by md5(StudyInstanceUID), then take (1) the first ACL-positive study, (2) the first
medial-OA-positive, ACL-negative study, (3) the first study with at most one positive finding.

Same decode / side / window path as the scored notebook (rsna_knee.wmil); attention recomputed exactly as in
scripts/viz_attention.py (asserted equal to the model head).

    RSNA_ROOT=... python scripts/demo_study.py      # -> $RSNA_ROOT/artifacts/demo/demo_studies.npz
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee import wmil  # noqa: E402
from viz_attention import attention_and_logits  # noqa: E402

T = wmil.TARGET_COLUMNS


def pick(gold: pd.DataFrame) -> list[tuple[str, str]]:
    g = gold.assign(h=[hashlib.md5(u.encode()).hexdigest() for u in gold.index]).sort_values("h")
    out = []
    acl = g[g["ACL"] == 1].index[0]
    out.append((acl, "first ACL-positive study"))
    oa = g[(g["Medial OA"] == 1) & (g["ACL"] == 0) & ~g.index.isin([acl])].index[0]
    out.append((oa, "first medial-OA-positive, ACL-negative study"))
    few = g[(g[T].sum(1) <= 1) & ~g.index.isin([acl, oa])].index[0]
    out.append((few, "first study with at most one positive finding"))
    return out


def main() -> None:
    root = Path(os.environ["RSNA_ROOT"])
    art, raw = root / "artifacts", root / "data/raw"
    out = art / "demo"
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(raw / "train.csv").set_index("StudyInstanceUID")
    gold = tr.dropna(subset=T)[T].astype(int)
    folds = pd.read_parquet(art / "folds.parquet")
    assert set(gold.index) == set(folds[folds.fold == -1].study_uid), "gold set must equal the held-out fold -1"
    chosen = pick(gold)

    sdf = pd.read_csv(raw / "train_series.csv").rename(columns={
        "StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid",
        "Anatomical_Plane": "plane", "Fluid_Sensitive": "fluid"})
    dev = "cuda"
    t = time.time()
    models = []
    for k in (0, 1):  # the efficiency-track submission: folds {0, 1}
        ck = torch.load(art / f"exp018_coatnet2_wmil_v3labels/fold{k}/best.pt", map_location="cpu", weights_only=False)
        m = wmil.WindowMIL(ck["arch"], pretrained=False, grad_checkpointing=False)
        m.load_state_dict(ck["model"], strict=True)
        models.append(m.to(dev).eval())
    res = int(ck["res"])
    t_load = time.time() - t
    torch.zeros(1, device=dev).sum().item()

    store, meta = {}, []
    for i, (uid, why) in enumerate(chosen):
        rows = sdf[sdf.study_uid == uid][["series_uid", "plane", "fluid"]].to_dict("records")
        sdir = raw / "train_series" / uid
        n_files = sum(1 for _ in sdir.rglob("*.dcm"))
        t = time.time()
        stack, present, metas = wmil.decode_study(sdir, rows)
        sides = wmil.resolve_slot_sides(metas, None)
        stack = wmil.normalize_orientation(stack, metas, sides)
        t_dec = time.time() - t
        t = time.time()
        st = torch.from_numpy(stack).to(dev)
        cs = wmil.candidate_centers(present)
        centers = torch.from_numpy(cs).to(dev)[None]
        att, lg = zip(*[attention_and_logits(m, st, centers, res) for m in models])
        torch.cuda.synchronize()
        t_inf = time.time() - t
        att, lg = np.stack(att), np.stack(lg)
        prob = (1 / (1 + np.exp(-lg))).mean(0)  # the notebook averages fold probabilities
        store[f"s{i}_stack"] = stack
        store[f"s{i}_present"] = present
        store[f"s{i}_centers"] = cs
        store[f"s{i}_slot_of"] = np.searchsorted(np.array(wmil.SLOT_OFF), cs, side="right") - 1
        store[f"s{i}_attn"] = att.mean(0).astype(np.float32)
        store[f"s{i}_prob"] = prob.astype(np.float32)
        store[f"s{i}_gold"] = gold.loc[uid, T].values.astype(np.int8)
        meta.append(dict(rule=why, n_series=len(rows), planes=sorted({r["plane"] for r in rows}), n_dicom=n_files,
                         n_windows=int(len(cs)), decode_s=round(t_dec, 2), model_s=round(t_inf, 2)))
        print(f"study {i + 1} ({why}): {len(rows)} series, {n_files} DICOM files, {len(cs)} windows, "
              f"decode {t_dec:.1f} s, 2 models {t_inf:.2f} s", flush=True)
    np.savez_compressed(out / "demo_studies.npz", **store)
    (out / "demo_meta.json").write_text(json.dumps(dict(targets=T, slots=[s[0] for s in wmil.SLOT_SPECS],
                                                        load_s=round(t_load, 1), gpu=torch.cuda.get_device_name(0),
                                                        studies=meta), indent=2))
    print(f"wrote {out / 'demo_studies.npz'} (model load {t_load:.1f} s)")


if __name__ == "__main__":
    main()
