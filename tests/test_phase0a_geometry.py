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

"""Synthetic end-to-end check of scripts/phase0a_registration.py (needs SimpleITK; runs in ~1-2 min on 4 cores).

A smooth 3D phantom is defined in patient space and "acquired" twice with exact, known geometry: a sagittal-like stack
and an oblique coronal-like stack (thick slices = slab averages, 3.5 mm spacing, different contrast). Because both stacks
sample the same physical object, the correct registration is the identity, and an injected rigid error must be
recovered. This tests the sign and direction conventions (index order, direction matrix, fixed->moving mapping, P o T0)
that the real audit relies on.

    python -m pytest tests/test_phase0a_geometry.py -q        or        python tests/test_phase0a_geometry.py
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("p0a", ROOT / "scripts" / "phase0a_registration.py")
p0a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0a)
sitk = p0a.sitk

RNG = np.random.default_rng(0)
BLOBS = [(RNG.uniform(-45, 45, 3), RNG.uniform(6, 18), RNG.uniform(0.3, 1.0)) for _ in range(40)]


def phantom(p: np.ndarray, contrast: int) -> np.ndarray:
    """Intensity at physical points p (N, 3): Gaussian blobs + two oblique 'tissue' boundaries."""
    v = np.zeros(len(p))
    for c, s, a in BLOBS:
        v += a * np.exp(-np.sum((p - c) ** 2, 1) / (2 * s * s))
    v += 0.6 * (p @ np.array([0.3, 0.2, 0.93]) > 5.0) + 0.4 * (p @ np.array([0.8, -0.6, 0.0]) > -12.0)
    inside = np.linalg.norm(p / np.array([60.0, 55.0, 70.0]), axis=1) < 1.0
    v = v * inside
    return v if contrast == 0 else np.where(inside, np.sqrt(np.abs(v)) * (1.2 - 0.5 * (v > 0.8)), 0.0)


def acquire(r, c, n, origin, ps, size, n_slices, gap, thick, contrast) -> sitk.Image:
    r, c, n = (np.array(x, float) / np.linalg.norm(x) for x in (r, c, n))
    j, i = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    vol = np.zeros((n_slices, size, size), np.float32)
    for k in range(n_slices):
        o = origin + k * gap * n
        acc = 0
        for dz in np.linspace(-thick / 2, thick / 2, 5):                    # slab average over the slice profile
            pts = o + dz * n + (i.ravel()[:, None] * ps) * r + (j.ravel()[:, None] * ps) * c
            acc = acc + phantom(pts, contrast)
        vol[k] = (acc / 5).reshape(size, size)
    vol = (vol - vol.min()) / (vol.max() - vol.min())
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((ps, ps, gap))
    img.SetOrigin(tuple(float(v) for v in origin))
    img.SetDirection(tuple(float(v) for v in np.column_stack([r, c, n]).ravel()))
    return sitk.BinShrink(img, [2, 2, 1])


def make_pair():
    sag = acquire([0, 1, 0], [0, 0, -1], [1, 0, 0], np.array([-42.0, -60.0, 60.0]), 0.5, 240, 25, 3.5, 3.0, 0)
    th = math.radians(7)                                                      # oblique coronal, rotated about z
    r = np.array([math.cos(th), math.sin(th), 0.0])
    n = np.cross(r, [0, 0, -1])
    o = -60.0 * r + np.array([0, 0, 60.0]) - 40.0 * n
    cor = acquire(r, [0, 0, -1], n, o, 0.5, 240, 23, 3.5, 3.0, 1)
    return sag, cor


def test_moved_and_to_matrix():
    sag, _ = make_pair()
    m = np.array([1.0, 2.0, 3.0])
    P = p0a.rigid(np.array([0.2, 0.9, -0.3]), 4.0, np.array([3.0, -2.0, 6.0]), m)
    idx = (10, 20, 5)
    x = np.array(sag.TransformIndexToPhysicalPoint(idx))
    y = np.array(p0a.moved(sag, P).TransformIndexToPhysicalPoint(idx))
    assert np.allclose(P[:3, :3] @ x + P[:3, 3], y, atol=1e-6)
    t = sitk.Euler3DTransform()
    t.SetCenter(tuple(m))
    t.SetRotation(0.02, -0.03, 0.05)
    t.SetTranslation((1.5, -2.0, 0.7))
    M = p0a.to_matrix(t, m)
    q = np.array([10.0, -20.0, 30.0])
    assert np.allclose(M[:3, :3] @ q + M[:3, 3], t.TransformPoint(tuple(q)), atol=1e-6)


def test_identity_and_injection_recovery():
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(4)
    sag, cor = make_pair()
    m = np.array([0.0, 0.0, 0.0])
    probes = np.array([m] + [m + 20.0 * np.array(s) for s in
                             [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]])
    rng = np.random.default_rng(5)
    mask, nvox = p0a.fixed_mask(sag, m, cor)
    assert nvox > p0a.MIN_MASK_VOXELS
    M0, info = p0a.register(sag, cor, mask, m, probes, rng, seed=17)
    dev = p0a.deviation(M0, probes)
    print("identity:", {k: round(v, 3) for k, v in dev.items()}, {k: info[k] for k in ("best", "n_agree_1mm")})
    assert dev["disp_max_mm"] < 1.0 and dev["rot_deg"] < 0.5
    assert info["mi_best"] <= info["mi_dicom"]
    # euler() must reproduce the matrix it is built from
    assert np.allclose(p0a.to_matrix(p0a.euler(M0, m), m), M0, atol=1e-6)
    for mm, deg in p0a.INJECT:
        P = p0a.rigid(np.array([0.4, -0.7, 0.6]), deg, mm * np.array([0.6, 0.0, 0.8]), m)
        corp = p0a.moved(cor, P)
        mask_p, _ = p0a.fixed_mask(sag, m, cor, corp)
        Mp, infop = p0a.register(sag, corp, mask_p, m, probes, rng, seed=27)
        err = p0a.deviation(np.linalg.inv(P @ M0) @ Mp, probes)
        print(f"inject {mm} mm / {deg} deg -> recovery error", {k: round(v, 3) for k, v in err.items()},
              {k: infop[k] for k in ("best", "n_agree_1mm")})
        assert err["disp_max_mm"] < 1.0 and err["rot_deg"] < 0.5


if __name__ == "__main__":
    test_moved_and_to_matrix()
    test_identity_and_injection_recovery()
    print("OK")
