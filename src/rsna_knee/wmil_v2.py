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

"""Window-MIL data on cache v2 for the Phase 0B preprocessing ladder (docs/research/PLAN_v1_3.md s4.2). One switch
per rung, everything else as exp018:

  rung  series        slices    plane_mix  crop
  B1    exp018_slots  exp018    uniform    c140    (exp018's 5 slots / 64 slices / windows; new pixels only)
  B2    exp018_slots  native    exp018     c140    (+ native-adjacent windows over all in-span slices)
  B3    all           native    exp018     c140    (+ every series, within-plane 1/N)
  B4    all           native    balanced   c140    (+ planes 1/3 each)
  B5    all           native    balanced   f160    (+ 160 mm field, no 336->384 resize)

`uniform` = uniform over all candidate windows (exp018's sampler). `exp018` = planes drawn with exp018's window shares
(Sag 28/54, Cor 16/54, Ax 10/54, renormalized over present planes); within a plane, exp018_slots weights series by
their exp018 slot window counts, `all` weights series equally. Laterality exactly as exp018 (wmil.normalize_orientation):
coronal/axial windows are mirrored when (side == R) != (row_x < 0); sagittal triplets are put in ascending patient-x
order and reversed for right knees. Evaluation is dense over all candidate windows (evenly thinned above a cap).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna_knee import wmil
from rsna_knee.cache_v2 import CENTRAL, CENTRAL_OFF, SPAN

PLANE_SHARE_EXP018 = {"Sagittal": 28 / 54, "Coronal": 16 / 54, "Axial": 10 / 54}
SLOT_WINDOWS = {spec[0]: spec[3] - 2 for spec in wmil.SLOT_SPECS}


def load_manifest_v2(manifest_dir: Path) -> pd.DataFrame:
    files = sorted(Path(manifest_dir).glob("shard_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cache-v2 manifests in {manifest_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["ok"]].copy()
    geo = []
    for m in df["meta"]:
        m = json.loads(m)
        iop = np.array(m["iop"])
        r, c = iop[:3], iop[3:]
        n = np.cross(r, c)
        ipp = np.array(m["ipp"])
        centre = ipp + r * m["ps_col"] * m["cols"] / 2 + c * m["ps_row"] * m["rows"] / 2
        sp = np.diff(ipp @ n)
        tag = str(m.get("Laterality") or m.get("ImageLaterality") or "")
        geo.append(dict(row_x=float(r[0]), normal_x=float(n[0]), cx=float(np.median(centre[:, 0])),
                        tag_laterality=tag, spacing=float(np.median(sp)) if len(sp) else 3.0,
                        slice_ok=m["slice_ok"]))
    g = pd.DataFrame(geo, index=df.index)
    return pd.concat([df.drop(columns=["meta"]), g], axis=1)


def _series_side(rows: list[dict], study_side, bilateral: bool) -> list:
    if bilateral:
        return [wmil.normalize_tag_side(r["tag_laterality"]) for r in rows]
    return wmil.resolve_slot_sides(rows, study_side if isinstance(study_side, str) else None)


class WMILv2Dataset(Dataset):
    def __init__(self, uids, cache_dir, man: pd.DataFrame, labels, final_side: dict, bilateral: set, train: bool,
                 k: int, series: str = "exp018_slots", slices: str = "exp018", plane_mix: str = "uniform",
                 crop: str = "c140", max_eval_windows: int = 192):
        self.uids, self.cache_dir, self.labels, self.train, self.k = uids, Path(cache_dir), labels, train, k
        self.final_side, self.bilateral = final_side, bilateral
        self.series_mode, self.slices, self.plane_mix, self.crop = series, slices, plane_mix, crop
        self.max_eval = max_eval_windows
        self.by_study = {u: g.to_dict("records") for u, g in man.groupby("study_uid")}

    def __len__(self):
        return len(self.uids)

    # ------------------------------------------------------------------ candidate windows per series
    def _windows(self, r: dict, slot: str | None) -> list[tuple]:
        n = int(r["n_slices"])
        order = np.arange(n) if (r["plane"] != "Sagittal" or r["normal_x"] >= 0) else np.arange(n)[::-1]
        lo, hi = int(round(SPAN[0] * (n - 1))), int(round(SPAN[1] * (n - 1)))
        if self.slices == "exp018":
            take = np.clip(np.round(np.linspace(lo, hi, wmil.SLOT_SPECS[[s[0] for s in wmil.SLOT_SPECS].index(slot)][3]))
                           .astype(int), 0, n - 1)
            return [tuple(order[take[j - 1:j + 2]]) for j in range(1, len(take) - 1)]
        step = max(1, int(round(3.0 / r["spacing"]))) if r["spacing"] < 2.0 else 1
        cs = range(max(lo, step), min(hi, n - 1 - step) + 1)
        return [tuple(order[[p - step, p, p + step]]) for p in cs] or [tuple(order[[max(0, n // 2 - 1), n // 2,
                                                                                    min(n - 1, n // 2 + 1)]])]

    def _select(self, uid):
        rows = self.by_study[uid]
        if self.series_mode == "exp018_slots":
            chosen = wmil.pick_slots([dict(series_uid=r["series_uid"], plane=r["plane"], fluid=r["fluid"],
                                           n_slices=r["n_slices"]) for r in rows])
            byuid = {r["series_uid"]: r for r in rows}
            return [(byuid[c["series_uid"]], wmil.SLOT_SPECS[i][0]) for i, c in sorted(chosen.items())]
        return [(r, None) for r in rows]

    def plan(self, i):
        """The windows of item i as (series_uid, channel-ordered source-slice triplet in stored order, mirror flag).
        __getitem__ renders exactly this plan, so the B1 check can compare it index-by-index with exp018."""
        uid = self.uids[i]
        sel = self._select(uid)
        sides = _series_side([r for r, _ in sel], self.final_side.get(uid), uid in self.bilateral)
        cands = []                      # (series index, triplet)
        for si, (r, slot) in enumerate(sel):
            cands += [(si, w) for w in self._windows(r, slot)]
        if self.train:
            rng = np.random.default_rng((torch.initial_seed() + 7919 * i) % (2 ** 32))
            picks = self._sample(sel, cands, rng)
        else:
            picks = cands if len(cands) <= self.max_eval else \
                [cands[j] for j in np.linspace(0, len(cands) - 1, self.max_eval).round().astype(int)]
        out = []
        for si, trip in picks:
            r, _ = sel[si]
            s, mirror = sides[si], False
            if s is not None:
                if r["plane"] == "Sagittal":
                    if s == "R":
                        trip = tuple(trip[::-1])
                else:
                    mirror = (s == "R") != (r["row_x"] < 0)
            out.append((r["series_uid"], tuple(int(t) for t in trip), mirror))
        return out

    def __getitem__(self, i):
        uid = self.uids[i]
        vols = {}
        out = []
        for sid, trip, mirror in self.plan(i):
            if sid not in vols:
                vols[sid] = np.load(self.cache_dir / uid / f"{sid}.npy", mmap_mode="r")
            w = np.stack([vols[sid][t] for t in trip])
            if self.crop == "c140":
                w = w[:, CENTRAL_OFF:CENTRAL_OFF + CENTRAL, CENTRAL_OFF:CENTRAL_OFF + CENTRAL]
            if mirror:
                w = w[:, :, ::-1]
            out.append(np.ascontiguousarray(w))
        K = self.k if self.train else self.max_eval
        H = out[0].shape[-1]
        win = np.zeros((K, 3, H, H), np.uint8)
        mask = np.zeros(K, bool)
        win[:len(out)] = np.stack(out)[:K]
        mask[:min(len(out), K)] = True
        res = {"uid": uid, "windows": torch.from_numpy(win), "wmask": torch.from_numpy(mask)}
        if self.labels is not None:
            res["labels"] = torch.tensor([float(self.labels.loc[uid, f"{c}__probability"]) for c in wmil.TARGET_COLUMNS],
                                         dtype=torch.float32)
        return res

    def _sample(self, sel, cands, rng):
        if self.plane_mix == "uniform":
            idx = rng.choice(len(cands), size=self.k, replace=len(cands) < self.k)
            return [cands[j] for j in idx]
        by_series = {}
        for si, w in cands:
            by_series.setdefault(si, []).append(w)
        planes = sorted({sel[si][0]["plane"] for si in by_series})
        if self.plane_mix == "exp018":
            pw = np.array([PLANE_SHARE_EXP018[p] for p in planes])
        else:                                                   # balanced
            pw = np.ones(len(planes))
        pw = pw / pw.sum()
        picks = []
        for _ in range(self.k):
            p = planes[rng.choice(len(planes), p=pw)]
            ss = [si for si in by_series if sel[si][0]["plane"] == p]
            if self.series_mode == "exp018_slots":
                sw = np.array([SLOT_WINDOWS[sel[si][1]] for si in ss], float)
            else:
                sw = np.ones(len(ss))
            si = ss[rng.choice(len(ss), p=sw / sw.sum())]
            ws = by_series[si]
            picks.append((si, ws[rng.integers(len(ws))]))
        return picks


def collate_v2(batch):
    out = {"uid": [b["uid"] for b in batch]}
    for key in ("windows", "wmask") + (("labels",) if "labels" in batch[0] else ()):
        out[key] = torch.stack([b[key] for b in batch])
    return out
