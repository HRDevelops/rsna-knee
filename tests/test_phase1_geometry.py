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

"""Pre-launch tests required by docs/research/PHASE1_PREREG.md s2 (no validation metric anywhere). Needs cache v2.

    RSNA_ROOT=... python tests/test_phase1_geometry.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee import cache_v2 as cv  # noqa: E402
from rsna_knee.tokens import TokenMIL  # noqa: E402
from rsna_knee.wmil_v2 import TokenV2Dataset, load_manifest_v2  # noqa: E402

R = Path(os.environ["RSNA_ROOT"])
MAN = load_manifest_v2(R / "artifacts/cache_manifests_v2")
SIDE = pd.read_csv(R / "artifacts/final_side.csv").set_index("study_uid")["final_side"].to_dict()
OPT = dict(series="all", slices="native", plane_mix="balanced", crop="f160")
UIDS = sorted(MAN.study_uid.unique())[:30]
FEAT = os.environ.get("FEATURES", "final")   # local: PHASE2_LOCAL_PREREG


def ds(geometry=True, mode="train", donor=None, uids=UIDS):
    return TokenV2Dataset(uids, R / "data/cache_v2", MAN, None, SIDE, set(), mode == "train", 12, grid=6, mode=mode,
                          geometry=geometry, donor=donor, **OPT)


def test_identity_at_beta0():
    torch.manual_seed(0)
    b = TokenMIL(pretrained=False, grid=6, geometry="none", grad_checkpointing=False, features=FEAT).eval()
    torch.manual_seed(0)
    g = TokenMIL(pretrained=False, grid=6, geometry="cross_plane", grad_checkpointing=False, features=FEAT).eval()
    sb, sg = b.state_dict(), g.state_dict()
    assert set(sg) - set(sb) == {"beta"} and all(torch.equal(sb[k], sg[k]) for k in sb), "shared init differs"
    item = ds(mode="dense")[0]
    K = item["windows"].shape[0]
    feats = torch.randn(1, K * 36, b.proj[1].in_features)
    args = [item[k][None] for k in ("slot", "plane", "pos", "meta", "valid")]
    with torch.no_grad():
        zb = b(feats, *args)
        zg = g(feats, *args, mu=item["mu"][None], sigma=item["sigma"][None])
    diff = (zb - zg).abs().max().item()
    assert diff < 1e-6, diff
    with torch.no_grad():                                         # geometry must be live once beta > 0
        g.beta.fill_(0.5)
        z2 = g(feats, *args, mu=item["mu"][None], sigma=item["sigma"][None])
    assert (z2 - zb).abs().max().item() > 1e-4
    print(f"  identity at beta=0: max |z_G - z_Bcap| = {diff:.2e}; beta=0.5 changes logits (ok)")


def test_padding_masked():
    torch.manual_seed(0)
    g = TokenMIL(pretrained=False, grid=6, geometry="cross_plane", grad_checkpointing=False, features=FEAT).eval()
    with torch.no_grad():
        g.beta.fill_(0.3)
    for i in range(len(UIDS)):
        item = ds(mode="dense")[i]
        if (item["valid"] < 0.5).any():
            break
    else:
        print("  padding: no padded token in the sample (skipped)")
        return
    K = item["windows"].shape[0]
    feats = torch.randn(1, K * 36, g.proj[1].in_features)
    args = [item[k][None] for k in ("slot", "plane", "pos", "meta", "valid")]
    bad = (item["valid"].flatten() < 0.5)
    f2 = feats.clone()
    f2[0, bad] = torch.randn(int(bad.sum()), feats.shape[-1]) * 100
    with torch.no_grad():
        z1 = g(feats, *args, mu=item["mu"][None], sigma=item["sigma"][None])
        z2 = g(f2, *args, mu=item["mu"][None], sigma=item["sigma"][None])
    assert (z1 - z2).abs().max().item() < 1e-4, "masked tokens influence the output"
    print(f"  padding: {int(bad.sum())} masked tokens have no influence (ok)")


def test_paired_sampling_and_donor_only_geometry():
    torch.manual_seed(1)
    a = ds(geometry=False)[3]
    torch.manual_seed(1)
    b = ds(geometry=True)[3]
    for k in ("windows", "slot", "plane", "pos", "meta", "valid"):
        assert torch.equal(a[k], b[k]), k
    donor = {UIDS[0]: UIDS[1]}
    x, y = ds(mode="dense", uids=UIDS[:1])[0], ds(mode="dense", uids=UIDS[:1], donor=donor)[0]
    for k in ("windows", "slot", "plane", "pos", "meta", "valid"):
        assert torch.equal(x[k], y[k]), k
    assert not torch.allclose(x["mu"], y["mu"]), "donor swap did not change geometry"
    print("  paired arms draw identical windows; donor swap changes only mu / sigma (ok)")


def test_footprint_reprojection_and_mirror():
    d = ds(mode="dense")
    n_checked = 0
    for i, u in enumerate(UIDS[:10]):
        for sid, trip, mirror, si, r, p in d._plan(i):
            mu, sig = d._footprint(u, r, trip, mirror)
            iop = np.asarray(r["iop"])
            rv, cvv = iop[:3], iop[3:]
            # independent recomputation of the (row 2, col 3) cell centre through cache_v2.network_to_patient
            ucol = (3 + 0.5) * 64 - 0.5
            ucol = 383 - ucol if mirror else ucol
            vrow = (2 + 0.5) * 64 - 0.5
            ref = np.mean([cv.network_to_patient(r["ipp"][t], rv, cvv, r["ps_row"], r["ps_col"], r["aff"], ucol, vrow)
                           for t in trip], 0)
            assert np.allclose(mu[2, 3], ref, atol=1e-3), (mu[2, 3], ref)
            if mirror:                                   # mirrored cell b equals unmirrored cell g-1-b
                mu0, _ = d._footprint(u, r, trip, False)
                assert np.allclose(mu[:, ::-1], mu0, atol=1e-3)
            ev = np.linalg.eigvalsh(sig)
            assert ev.min() > 0 and np.isclose(ev.max(), (160 / 6) ** 2 / 12, rtol=0.3)
            n_checked += 1
    print(f"  footprints: {n_checked} windows reproject exactly; mirror consistent; Sigma positive definite (ok)")


if __name__ == "__main__":
    test_footprint_reprojection_and_mirror()
    test_paired_sampling_and_donor_only_geometry()
    test_identity_at_beta0()
    test_padding_masked()
    print("ALL PHASE 1 PRE-LAUNCH TESTS PASSED")
