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

"""Unit tests of the cache-v2 geometry contract (plan s4.1): reprojection to native/patient coordinates <= 0.5 px,
centring identical to exp018's centre crop, analytic validity equal to rendered padding."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee import cache_v2 as cv  # noqa: E402


def _blob_image(rows, cols, rc):
    v, u = np.mgrid[0:rows, 0:cols]
    return np.exp(-((v - rc[0]) ** 2 + (u - rc[1]) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)


def test_reprojection_subpixel():
    rng = np.random.default_rng(0)
    for rows, cols, ps_r, ps_c in [(512, 512, 0.3125, 0.3125), (320, 256, 0.5, 0.625), (448, 384, 0.39, 0.39)]:
        aff = cv.affine_for(rows, cols, ps_r, ps_c)
        for _ in range(5):
            rc = (rng.uniform(rows * 0.3, rows * 0.7), rng.uniform(cols * 0.3, cols * 0.7))
            img = cv.resample_exact(_blob_image(rows, cols, rc), aff)
            vv, uu = np.mgrid[0:cv.SIZE, 0:cv.SIZE]
            w = img ** 4
            v_net, u_net = (w * vv).sum() / w.sum(), (w * uu).sum() / w.sum()
            a_c, b_c, a_r, b_r = aff
            err_native = np.hypot(a_r * v_net + b_r - rc[0], a_c * u_net + b_c - rc[1])
            err_net_px = err_native * min(ps_r, ps_c) / cv.SPACING
            assert err_net_px < 0.5, (rows, cols, err_net_px)


def test_patient_mapping_and_centre():
    r, c = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    ipp = np.array([10.0, -80.0, 80.0])
    aff = cv.affine_for(512, 512, 0.3125, 0.3125)
    centre_net = (cv.SIZE - 1) / 2
    p = cv.network_to_patient(ipp, r, c, 0.3125, 0.3125, aff, centre_net, centre_net)
    expected = ipp + 511 / 2 * 0.3125 * r + 511 / 2 * 0.3125 * c          # native image centre
    assert np.allclose(p, expected, atol=1e-6)
    p2 = cv.network_to_patient(ipp, r, c, 0.3125, 0.3125, aff, centre_net + 1, centre_net)
    assert np.isclose(np.linalg.norm(p2 - p), cv.SPACING, atol=1e-9)        # one network pixel = 0.4167 mm


def test_validity_matches_rendered_padding():
    aff = cv.affine_for(256, 320, 0.5, 0.5)                                  # 128 x 160 mm < 160 mm field
    img = cv.resample_exact(np.ones((256, 320), np.float32), aff)
    vm = cv.valid_mask(256, 320, aff)
    assert np.all(img[vm] > 0.999)
    assert np.all(img[~vm] < 1e-6) or np.mean(img[~vm] < 1e-6) > 0.99


if __name__ == "__main__":
    test_reprojection_subpixel()
    test_patient_mapping_and_centre()
    test_validity_matches_rendered_padding()
    print("OK")
