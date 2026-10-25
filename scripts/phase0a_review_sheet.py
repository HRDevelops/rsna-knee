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

"""Blinded 40-pair geometry review (docs/research/PHASE1_PREREG.md s6; descriptive only, cannot change Phase 0A's
inconclusive label). Picks 40 cross-plane pairs from the 145 non-pilot Phase 0A studies, 10 per stratum
(low / moderate / high correction with agreeing starts; disagreeing starts or gross tail), balanced over pair type and
field strength where possible. For each pair renders the fixed series' slice through the study centre with the moving
series resampled onto it, at the raw DICOM placement and at the MI correction, as two images A and B in random order
(checkerboard + colour overlay, same crop and window). Writes OUT/img/*.png, OUT/review.html (rating form, saves in the
browser and exports CSV) and OUT/key.csv (which of A/B is raw; keep apart from the rater). Image data: local only.

    RSNA_ROOT=... python scripts/phase0a_review_sheet.py OUT
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase0a_registration as p0a  # noqa: E402

out = Path(sys.argv[1])
(out / "img").mkdir(parents=True, exist_ok=True)
root = Path(p0a.os.environ["RSNA_ROOT"])
art = root / "artifacts/phase0a"
pil = set((art / "pilot_studies.txt").read_text().split())
d = pd.concat([pd.read_parquet(f) for f in sorted(art.glob("pairs_*.parquet"))])
ser = pd.concat([pd.read_parquet(f) for f in sorted(art.glob("series_*.parquet"))])
b = d[(d.inject == 0) & d.ok & (d.pair_type == "cross") & ~d.study.isin(pil)].copy()
b["kind"] = b.fixed_plane.str[:3] + "-" + b.moving_plane.str[:3]
agree = b.n_agree_1mm == b.n_starts_ok
gross = (b.disp_max_mm > 10) | (b.rot_deg > 5)
b["stratum"] = np.select([~agree | gross, b.disp_max_mm > 3, b.disp_max_mm > 1.5], ["4_unstable_or_gross", "3_high",
                                                                                     "2_moderate"], "1_low")
rng = random.Random(20261026)
picked = []
for st, g in b.groupby("stratum"):
    g = g.sample(frac=1, random_state=7)
    # balance: round-robin over pair types; within a type alternate 3 T / not-3 T
    by_kind = {}
    for r in g.itertuples():
        by_kind.setdefault(r.kind, {True: [], False: []})[bool(r.field >= 2.5)].append(r)
    take, turn = [], {k: True for k in by_kind}
    while len(take) < 10 and any(v[True] or v[False] for v in by_kind.values()):
        for kind in sorted(by_kind):
            v = by_kind[kind]
            if len(take) >= 10:
                break
            first = turn[kind]
            src = v[first] if v[first] else v[not first]
            if src:
                take.append(src.pop(0))
                turn[kind] = not first
    picked += take
rng.shuffle(picked)
key_rows, items = [], []
raw_dir = root / "data/raw/train_series"
for i, r in enumerate(picked):
    sa = ser[(ser.study == r.study) & (ser.family == r.fixed)].iloc[0]
    sb = ser[(ser.study == r.study) & (ser.family == r.moving)].iloc[0]
    fa, _ = p0a.load_series(raw_dir / r.study / sa.series_uid)
    mb, _ = p0a.load_series(raw_dir / r.study / sb.series_uid)
    m = np.array(r.centre)
    M = np.array(r.M).reshape(4, 4)
    k = int(round(np.array(fa.TransformPhysicalPointToContinuousIndex(tuple(m)))[2]))
    k = min(max(k, 0), fa.GetSize()[2] - 1)
    ci = np.array(fa.TransformPhysicalPointToContinuousIndex(tuple(m)))[:2]
    sp = fa.GetSpacing()[0]
    half = int(round(50.0 / sp))                    # 100 mm crop around the centre
    f = sitk.GetArrayFromImage(fa)[k]
    y0, x0 = int(ci[1]) - half, int(ci[0]) - half

    def crop(a):
        pad = np.pad(a, half, mode="constant")
        return pad[y0 + half:y0 + 3 * half, x0 + half:x0 + 3 * half]

    def render(T):
        mv = sitk.GetArrayFromImage(sitk.Resample(mb, fa, T, sitk.sitkLinear, 0.0))[k]
        F, Mv = crop(f), crop(mv)
        cb = np.where(((np.indices(F.shape) // max(1, int(10 / sp))).sum(0) % 2) == 0, F, Mv)
        ov = np.stack([Mv, F, Mv], -1)
        img = np.concatenate([np.stack([cb] * 3, -1), np.ones((F.shape[0], 4, 3)), ov], 1)
        return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(
            (2 * 300 + 4, 300), Image.BILINEAR)

    raw_first = rng.random() < 0.5
    views = {"raw": render(sitk.Euler3DTransform()), "mi": render(p0a.euler(M, m))}
    order = ["raw", "mi"] if raw_first else ["mi", "raw"]
    for lab, which in zip("AB", order):
        views[which].save(out / "img" / f"pair{i + 1:02d}_{lab}.png")
    key_rows.append(dict(pair=i + 1, A=order[0], B=order[1], study=r.study, fixed=r.fixed, moving=r.moving,
                         kind=r.kind, field=r.field, stratum=r.stratum, corr_mm=r.disp_max_mm, rot_deg=r.rot_deg))
    items.append(i + 1)
    print(f"pair {i + 1}/40 {r.kind} {r.stratum}", flush=True)
pd.DataFrame(key_rows).to_csv(out / "key.csv", index=False)

opts = {"better": ["A", "B", "indistinguishable", "unrateable"], "magnitude": ["none", "subtle", "clear", "gross"],
        "confidence": ["1", "2", "3", "4", "5"], "mismatch": ["translation", "rotation", "both", "cannot tell"]}
rows_html = []
for i in items:
    ctl = "".join(
        f"<div class='q'><b>{q}</b> " + " ".join(
            f"<label><input type='radio' name='p{i}_{q}' value='{v}'>{v}</label>" for v in vs) + "</div>"
        for q, vs in opts.items())
    rows_html.append(f"<section><h3>Pair {i}</h3><div class='imgs'><figure><img src='img/pair{i:02d}_A.png'>"
                     f"<figcaption>A</figcaption></figure><figure><img src='img/pair{i:02d}_B.png'>"
                     f"<figcaption>B</figcaption></figure></div>{ctl}</section>")
html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Geometry review</title>
<style>body{{font:14px system-ui;margin:16px;max-width:1300px}} .imgs{{display:flex;gap:12px;flex-wrap:wrap}}
img{{max-width:100%}} section{{border-top:1px solid #ccc;padding:8px 0}} .q{{margin:4px 0}} label{{margin-right:10px}}</style>
</head><body><h1>Blinded cross-plane geometry review (40 pairs)</h1>
<p>Each image: left = checkerboard of the two series on the same plane, right = overlay (green = fixed series, magenta =
other series resampled). A and B are two spatial placements of the same pair. Judge which placement makes the anatomy
agree better (bone contours, joint line, condyles, tibial plateau). Do not guess which is which. Answers are saved in
this browser; press Export when done.</p>{''.join(rows_html)}
<p><button onclick='exp()'>Export CSV</button> <span id='n'></span></p>
<script>
const items={json.dumps(items)}, qs={json.dumps(list(opts))};
function save(){{const s={{}};document.querySelectorAll('input:checked').forEach(e=>s[e.name]=e.value);
 try{{localStorage.setItem('geomreview',JSON.stringify(s))}}catch(e){{}}
 document.getElementById('n').textContent=Object.keys(s).length+' answers';}}
document.addEventListener('change',save);
try{{const s=JSON.parse(localStorage.getItem('geomreview')||'{{}}');for(const k in s){{const e=document.querySelector(
 `input[name="${{k}}"][value="${{s[k]}}"]`);if(e)e.checked=true}}}}catch(e){{}} save();
function exp(){{let c='pair,'+qs.join(',')+'\\n';for(const i of items){{c+=i+','+qs.map(q=>{{const e=document.querySelector(
 `input[name="p${{i}}_${{q}}"]:checked`);return e?e.value:''}}).join(',')+'\\n'}}
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([c],{{type:'text/csv'}}));
 a.download='geometry_review_ratings.csv';a.click()}}
</script></body></html>"""
(out / "review.html").write_text(html, encoding="utf-8")
print("wrote", out / "review.html")
