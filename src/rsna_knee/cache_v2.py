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

"""Cache v2 (docs/research/PLAN_v1_3.md s4.1): every series, every native slice, 160 mm field of view at 384 px
(0.4167 mm/px = exp018's pixel size), EXACT affine + linear interpolation, no added prefilter.

Geometry contract. For a network pixel (u', v') (column, row; pixel centres at integer indices) of slice k:
    native column  u = a_c * u' + b_c        native row  v = a_r * v' + b_r
with a = t / PixelSpacing and b chosen so the network grid is centred on the native image centre, i.e. the same
"centre crop / pad" as exp018 but without zoom's corner-aligned rounding. Patient position (LPS, mm):
    x = IPP_k + u * ps_col * r + v * ps_row * c
Pixels whose native coordinates fall outside [0, cols-1] x [0, rows-1] are padding (value 0); their share per token is
the validity v_j of plan s2.4, computable from (a, b, rows, cols) alone. Slices are stored in ascending order of their
IPP projection on the slice normal n = r x c, UNMIRRORED: the laterality mirror is applied at load time (and must be
folded into the affine there).

The central 336 px of the 384 grid is exp018's 140 mm crop on exp018's grid (ladder rung B1).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from scipy.ndimage import map_coordinates

from rsna_knee.preprocess import _order_from_headers

FOV_MM = 160.0
SIZE = 384
SPACING = FOV_MM / SIZE          # 0.41667 mm
CENTRAL = 336                    # exp018's 140 mm crop inside the 384 grid
CENTRAL_OFF = (SIZE - CENTRAL) // 2
SPAN = (0.06, 0.94)
META_TAGS = ["SliceThickness", "SpacingBetweenSlices", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle",
             "MagneticFieldStrength", "Manufacturer", "ManufacturerModelName", "SeriesDescription", "ScanningSequence",
             "SequenceVariant", "ScanOptions", "MRAcquisitionType", "AcquisitionMatrix",
             "InPlanePhaseEncodingDirection", "PercentSampling", "PercentPhaseFieldOfView", "PixelBandwidth",
             "EchoTrainLength", "Laterality", "ImageLaterality"]


def affine_for(rows: int, cols: int, ps_row: float, ps_col: float, t: float = SPACING, size: int = SIZE):
    """(a_c, b_c, a_r, b_r): native = a * network + b, network grid centred on the native image centre."""
    a_c, a_r = t / ps_col, t / ps_row
    b_c = (cols - 1) / 2.0 - a_c * (size - 1) / 2.0
    b_r = (rows - 1) / 2.0 - a_r * (size - 1) / 2.0
    return a_c, b_c, a_r, b_r


def resample_exact(arr: np.ndarray, aff, size: int = SIZE) -> np.ndarray:
    a_c, b_c, a_r, b_r = aff
    g = np.arange(size, dtype=np.float64)
    vv, uu = np.meshgrid(a_r * g + b_r, a_c * g + b_c, indexing="ij")     # (row, col) native coordinates
    return map_coordinates(arr.astype(np.float32), [vv, uu], order=1, mode="constant", cval=0.0, prefilter=False)


def valid_mask(rows: int, cols: int, aff, size: int = SIZE) -> np.ndarray:
    a_c, b_c, a_r, b_r = aff
    g = np.arange(size, dtype=np.float64)
    vr = (a_r * g + b_r >= 0) & (a_r * g + b_r <= rows - 1)
    vc = (a_c * g + b_c >= 0) & (a_c * g + b_c <= cols - 1)
    return np.outer(vr, vc)


def network_to_patient(ipp, r, c, ps_row, ps_col, aff, u_net, v_net):
    a_c, b_c, a_r, b_r = aff
    u = a_c * np.asarray(u_net, float) + b_c
    v = a_r * np.asarray(v_net, float) + b_r
    return np.asarray(ipp, float) + np.multiply.outer(u * ps_col, np.asarray(r, float)) \
        + np.multiply.outer(v * ps_row, np.asarray(c, float))


def _val(h, k):
    v = h.get(k, None)
    if v is None:
        return None
    if isinstance(v, (pydicom.multival.MultiValue, list, tuple)):
        return [str(x) for x in v]
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def decode_series_v2(series_dir: Path) -> tuple[np.ndarray, dict]:
    """One series -> uint8 [n, 384, 384] (all native slices, ascending along the normal) + geometry/metadata."""
    files = sorted(glob.glob(str(Path(series_dir) / "*.dcm")))
    headers = [pydicom.dcmread(f, stop_before_pixels=True, force=True) for f in files]
    order, ometa = _order_from_headers(headers)
    files = [files[i] for i in order]
    headers = [headers[i] for i in order]
    h0 = headers[0]
    iop = np.array([float(x) for x in h0.ImageOrientationPatient])
    r, c = iop[:3] / np.linalg.norm(iop[:3]), iop[3:] / np.linalg.norm(iop[3:])
    ipp = np.array([[float(x) for x in h.ImagePositionPatient] for h in headers])
    ps_row, ps_col = (float(x) for x in h0.PixelSpacing)
    rows, cols = int(h0.Rows), int(h0.Columns)
    aff = affine_for(rows, cols, ps_row, ps_col)
    vol = np.zeros((len(files), SIZE, SIZE), np.float32)
    ok = np.zeros(len(files), bool)
    for k, f in enumerate(files):
        try:
            ds = pydicom.dcmread(f, force=True)
            arr = ds.pixel_array.astype(np.float32)
            try:
                arr = apply_modality_lut(arr, ds).astype(np.float32)
            except Exception:  # noqa: BLE001
                pass
            if arr.shape != (rows, cols):
                raise ValueError(f"slice shape {arr.shape} != {(rows, cols)}")
            vol[k] = resample_exact(arr, aff)
            ok[k] = True
        except Exception:  # noqa: BLE001 - one bad slice must not kill the series
            ok[k] = False
    if not ok.any():
        raise ValueError("no slice decoded")
    n = len(files)
    lo, hi = int(round(SPAN[0] * (n - 1))), int(round(SPAN[1] * (n - 1)))
    span = vol[lo:hi + 1][ok[lo:hi + 1]] if ok[lo:hi + 1].any() else vol[ok]
    cen = span[:, CENTRAL_OFF:CENTRAL_OFF + CENTRAL, CENTRAL_OFF:CENTRAL_OFF + CENTRAL]
    vm = valid_mask(rows, cols, aff)[CENTRAL_OFF:CENTRAL_OFF + CENTRAL, CENTRAL_OFF:CENTRAL_OFF + CENTRAL]
    vals = cen[:, vm] if vm.any() else cen.reshape(len(cen), -1)
    p1, p995 = np.percentile(vals, [1, 99.5])
    if p995 <= p1:
        p995 = p1 + 1.0
    u8 = (np.clip((vol - p1) / (p995 - p1), 0, 1) * 255.0).round().astype(np.uint8)
    u8[~ok] = 0
    meta = dict(n_slices=n, rows=rows, cols=cols, ps_row=ps_row, ps_col=ps_col, aff=list(aff),
                iop=iop.tolist(), ipp=ipp.tolist(), slice_ok=ok.tolist(), p1=float(p1), p995=float(p995),
                order_mismatch=bool(ometa.get("order_mismatch", False)),
                irregular_spacing=bool(ometa.get("irregular_spacing", False)),
                **{k: _val(h0, k) for k in META_TAGS})
    return u8, meta
