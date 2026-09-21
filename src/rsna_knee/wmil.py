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

"""exp018 — adjacent-slice Window-MIL (2026-09-21): one shared implementation for the cache builder, the
training dataset, and the Kaggle inference notebook, so preprocessing cannot drift between them.

Design (in-house; the public dreaddevelopment single-model kernels were read only as an architectural
reference — no third-party trained weights, labels or code are used):

  study  -> 5 fixed slots, 64 slices total: Sagittal-FS 18, Sagittal-nonFS 14, Coronal-FS 12,
            Coronal-nonFS 8, Axial 12 (slot layout below; presence-masked, absent slot = zeros)
  slice  -> 140 mm field of view resampled to 336 px (0.4167 mm/px); slices taken linearly from the
            6%-94% span of the geometrically ordered series (ImagePositionPatient projected on the slice
            normal — same ordering code as the deployed pipeline, rsna_knee.preprocess)
  window -> 3 adjacent slices [t-1, t, t+1] of the SAME slot as an RGB image (never spans two slots);
            a full study has <= 54 candidate windows (interior slices of each present slot)
  model  -> CoAtNet-2 (fully fine-tuned) per window -> per-finding attention-MIL over ALL of the study's
            windows (no per-series CLS, no slot bottleneck) -> 12 logits

Laterality (see normalize_orientation): medial/lateral targets need "medial" on a fixed side. Coronal and
axial: mirror the image horizontally for right knees. Sagittal: the medial-lateral axis is the SLICE axis, so
right knees get their slice order reversed (sagittal slices are stored in ascending patient-x order).
"""
from __future__ import annotations

import copy
import glob
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.preprocess import _order_from_headers, normalize_uint8, resample_and_crop

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# (name, plane, fluid_sensitive flag or None for "any", n_slices)
SLOT_SPECS = [
    ("Sagittal_FS", "Sagittal", 1, 18),
    ("Sagittal_nonFS", "Sagittal", 0, 14),
    ("Coronal_FS", "Coronal", 1, 12),
    ("Coronal_nonFS", "Coronal", 0, 8),
    ("Axial_any", "Axial", None, 12),
]
NUM_SLOTS = len(SLOT_SPECS)
SLOT_LEN = [s[3] for s in SLOT_SPECS]
SLOT_OFF = [int(x) for x in np.concatenate([[0], np.cumsum(SLOT_LEN)[:-1]])]
SLOT_PLANE = [s[1] for s in SLOT_SPECS]
DEPTH = int(sum(SLOT_LEN))  # 64
IMG = 336
FOV_MM = 140.0
SPAN = (0.06, 0.94)
TARGET_SPACING = FOV_MM / IMG
MAX_WINDOWS = sum(max(l - 2, 0) for l in SLOT_LEN)  # 54
GEOM_SIDE_MIN_OFFSET_MM = 20.0  # |median image-centre x| below this -> side unresolved (near isocentre)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------------------------- slot selection
def pick_slots(rows: list[dict]) -> dict[int, dict]:
    """rows: dicts with series_uid, plane (Anatomical_Plane), fluid (Fluid_Sensitive 0/1), n_slices.
    Longest matching series per slot; a series is used by at most one slot. The non-fluid slots fall back
    to an unused fluid-sensitive series of the same plane (a study with two fluid-sensitive sagittals and
    no other still fills both sagittal slots). Deterministic (ties broken by series_uid)."""
    used: set[str] = set()
    chosen: dict[int, dict] = {}
    for i, (_, plane, fluid, _) in enumerate(SLOT_SPECS):
        cands = [r for r in rows if r["plane"] == plane and r["series_uid"] not in used
                 and (fluid is None or int(r["fluid"]) == fluid)]
        if not cands and fluid == 0:
            cands = [r for r in rows if r["plane"] == plane and r["series_uid"] not in used]
        if cands:
            best = max(cands, key=lambda r: (int(r["n_slices"]), str(r["series_uid"])))
            chosen[i] = best
            used.add(best["series_uid"])
    return chosen


# ------------------------------------------------------------------------------------------- decoding
def _geometry(headers: list, order: np.ndarray) -> dict:
    """Orientation/side descriptors from headers only. row_x = x component of the in-plane row direction
    (sign says whether image-right points to the patient's left), normal_x likewise for the slice normal,
    cx = median patient-x of the image centre over slices (negative = right of the isocentre)."""
    iop = [float(x) for x in headers[0].get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
    row, col = np.array(iop[0:3]), np.array(iop[3:6])
    normal = np.cross(row, col)
    xs = []
    for ds in headers:
        try:
            ipp = np.array([float(x) for x in ds.ImagePositionPatient])
            ps = [float(x) for x in ds.PixelSpacing]
            centre = ipp + row * ps[1] * float(ds.Columns) / 2 + col * ps[0] * float(ds.Rows) / 2
            xs.append(float(centre[0]))
        except Exception:  # noqa: BLE001 - missing geometry tags on some slices
            continue
    tag = str(headers[0].get("Laterality", "") or headers[0].get("ImageLaterality", "") or "")
    return {"row_x": float(row[0]), "normal_x": float(normal[0]),
            "cx": float(np.median(xs)) if xs else float("nan"), "tag_laterality": tag}


def decode_series_slot(series_dir: Path, plane: str, n_take: int) -> tuple[np.ndarray, dict]:
    """One series -> [n_take, IMG, IMG] uint8 plus meta. Header-only ordering, then only the n_take chosen
    files are pixel-decoded. Sagittal series are put in ASCENDING patient-x order regardless of the
    scanner's normal sign (so a later 'reverse for right knees' is a well-defined laterality normalisation)."""
    files = sorted(Path(series_dir).glob("*.dcm"))
    headers = [pydicom.dcmread(str(f), force=True, stop_before_pixels=True) for f in files]
    if not headers:
        raise ValueError("no .dcm files in series dir")
    order, meta = _order_from_headers(headers)
    geom = _geometry(headers, order)
    if plane == "Sagittal" and geom["normal_x"] < 0:
        order = order[::-1]
    n = len(order)
    lo, hi = int(round(SPAN[0] * (n - 1))), int(round(SPAN[1] * (n - 1)))
    take = np.clip(np.round(np.linspace(lo, hi, n_take)).astype(int), 0, n - 1)

    decoded: dict[int, np.ndarray] = {}
    for t in sorted(set(int(v) for v in take)):
        try:
            ds = pydicom.dcmread(str(files[order[t]]), force=True)
            decoded[t] = resample_and_crop(ds, TARGET_SPACING, IMG)
        except Exception:  # noqa: BLE001 - one bad slice must not kill the series
            decoded[t] = None
    good = [t for t in decoded if decoded[t] is not None]
    if not good:
        raise ValueError("every selected slice failed to decode")
    slices = []
    for t in take:
        t = int(t)
        if decoded[t] is None:  # nearest successfully decoded slice
            t = min(good, key=lambda g: abs(g - t))
        slices.append(decoded[t])
    vol = normalize_uint8(np.stack(slices, axis=0))
    meta = {**meta, **geom, "n_native": n}
    return vol, meta


def decode_study(study_dir: Path, series_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list]:
    """series_rows: dicts {series_uid, plane, fluid}; n_slices is counted here from the directory. Returns
    (stack [DEPTH, IMG, IMG] uint8 in slot layout, present [NUM_SLOTS] bool, per-slot meta or None)."""
    rows = []
    for r in series_rows:
        d = Path(study_dir) / str(r["series_uid"])
        n = len(glob.glob(str(d / "*.dcm")))
        if n:
            rows.append({**r, "n_slices": n})
    chosen = pick_slots(rows)
    stack = np.zeros((DEPTH, IMG, IMG), dtype=np.uint8)
    present = np.zeros(NUM_SLOTS, dtype=bool)
    metas: list = [None] * NUM_SLOTS
    for i in range(NUM_SLOTS):
        r = chosen.get(i)
        if r is None:
            continue
        try:
            vol, meta = decode_series_slot(Path(study_dir) / str(r["series_uid"]), SLOT_PLANE[i], SLOT_LEN[i])
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] {study_dir.name}/{r['series_uid']} slot {i} decode failed: {type(e).__name__}: {e}")
            continue
        stack[SLOT_OFF[i]:SLOT_OFF[i] + SLOT_LEN[i]] = vol
        present[i] = True
        metas[i] = {**meta, "series_uid": str(r["series_uid"])}
    return stack, present, metas


# -------------------------------------------------------------------------------------- laterality
def normalize_tag_side(v) -> str | None:
    v = str(v or "").strip().upper()
    if v in ("R", "RIGHT"):
        return "R"
    if v in ("L", "LEFT"):
        return "L"
    return None


def side_from_geometry(cx: float) -> str | None:
    if cx is None or not np.isfinite(cx) or abs(cx) < GEOM_SIDE_MIN_OFFSET_MM:
        return None
    return "R" if cx < 0 else "L"


def resolve_slot_sides(metas: list, study_side: str | None = None) -> list[str | None]:
    """Per-slot side. `study_side` (training: the validated report/tag side from final_side.csv) wins when
    given. Otherwise the study's DICOM Laterality tags, if unanimous; otherwise the geometric image-centre
    rule, applied per study (median over its slots). Never guessed beyond that: None -> no flip."""
    if study_side in ("L", "R"):
        return [study_side if m is not None else None for m in metas]
    tags = {normalize_tag_side(m["tag_laterality"]) for m in metas if m is not None} - {None}
    if len(tags) == 1:
        s = next(iter(tags))
    else:
        cxs = [m["cx"] for m in metas if m is not None and np.isfinite(m["cx"])]
        s = side_from_geometry(float(np.median(cxs))) if cxs else None
    return [s if m is not None else None for m in metas]


def normalize_orientation(stack: np.ndarray, metas: list, sides: list) -> np.ndarray:
    """Return a copy in the canonical (left-knee) frame. Coronal/axial: horizontal mirror for right knees
    (XOR the image's own left-right sense, row_x < 0). Sagittal: reverse the slice order for right knees."""
    out = stack.copy()
    for i in range(NUM_SLOTS):
        m, s = metas[i], sides[i]
        if m is None or s is None:
            continue
        o, l = SLOT_OFF[i], SLOT_LEN[i]
        if SLOT_PLANE[i] == "Sagittal":
            if s == "R":
                out[o:o + l] = out[o:o + l][::-1]
        else:
            if (s == "R") != (m["row_x"] < 0):
                out[o:o + l] = out[o:o + l][:, :, ::-1]
    return out


# ---------------------------------------------------------------------------------------------- windows
def candidate_centers(present: np.ndarray) -> np.ndarray:
    """Global stack indices usable as window centres: the interior slices of each present slot, so
    [c-1, c, c+1] never leaves its own slot."""
    cs = []
    for i in range(NUM_SLOTS):
        if present[i]:
            o, l = SLOT_OFF[i], SLOT_LEN[i]
            cs.extend(range(o + 1, o + l - 1))
    return np.array(cs, dtype=np.int64)


def gather_windows(stack: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """stack [B, DEPTH, H, W] (any dtype), centers [B, K] -> [B, K, 3, H, W]."""
    b = torch.arange(stack.shape[0], device=stack.device)[:, None, None]
    idx = centers[:, :, None] + torch.tensor([-1, 0, 1], device=stack.device)[None, None, :]
    return stack[b, idx]


def prepare_input(windows_u8: torch.Tensor, res: int, gain: torch.Tensor | None = None) -> torch.Tensor:
    """uint8 [B, K, 3, H, W] -> normalised float [B*K, 3, res, res] (ImageNet statistics; bilinear resize
    to the backbone's native resolution). gain [B] (training intensity jitter) scales each study's windows
    before clipping to [0, 1]."""
    B, K = windows_u8.shape[:2]
    x = windows_u8.flatten(0, 1).float().div_(255.0)
    if gain is not None:
        x = (x * gain.repeat_interleave(K).view(-1, 1, 1, 1)).clamp_(0.0, 1.0)
    if x.shape[-1] != res or x.shape[-2] != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# -------------------------------------------------------------------------------------------- model
class WindowMIL(nn.Module):
    """CoAtNet-2 window encoder + per-finding attention-MIL over all of a study's windows.

    forward(x, wmask): x [B, K, 3, R, R] normalised float, wmask [B, K] bool (False = padding window).
    Each finding has its own attention over windows: a_{b,k,o} = softmax_k(w_o . tanh(W h_{b,k})), pooled
    c_{b,o} = sum_k a h, logit_o = <c_o, clsW_o> + b_o. No per-series token, no slot bottleneck."""

    def __init__(self, arch: str = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained: bool = True,
                 n_targets: int = 12, drop: float = 0.2, grad_checkpointing: bool = False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=3,
                                          global_pool="avg")
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        d = self.backbone.num_features
        self.norm = nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n_targets))
        self.cls_w = nn.Parameter(torch.zeros(n_targets, d))
        self.cls_b = nn.Parameter(torch.zeros(n_targets))
        nn.init.trunc_normal_(self.cls_w, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, K = x.shape[:2]
        return self.backbone(x.flatten(0, 1)).view(B, K, -1)

    def head(self, feats: torch.Tensor, wmask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(feats)
        a = self.att(h)  # [B, K, n]
        if wmask is not None:
            a = a.masked_fill(~wmask.unsqueeze(-1), -1e4)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def forward(self, x: torch.Tensor, wmask: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encode(x), wmask)


@torch.no_grad()
def predict_study(model: WindowMIL, stack_u8: torch.Tensor, present: np.ndarray, res: int,
                  dtype: torch.dtype, chunk: int = 64) -> torch.Tensor:
    """One study, ALL of its candidate windows. stack_u8 [DEPTH, H, W] on the model's device (orientation
    already normalised). Returns 12 logits. Falls back to the centre slice window if a study has no slots."""
    dev = stack_u8.device
    cs = candidate_centers(present)
    if len(cs) == 0:
        return torch.zeros(len(TARGET_COLUMNS), device=dev)
    centers = torch.from_numpy(cs).to(dev)[None]
    feats = []
    for s in range(0, centers.shape[1], chunk):
        w = gather_windows(stack_u8[None], centers[:, s:s + chunk])
        with torch.autocast(device_type=dev.type, dtype=dtype):
            feats.append(model.encode(prepare_input(w, res).view(1, -1, 3, res, res)).float())
    feats = torch.cat(feats, dim=1)
    with torch.autocast(device_type=dev.type, enabled=False):
        return model.head(feats)[0]


def average_state_dicts(states: list[dict]) -> dict:
    """Uniform average of float tensors (parameters and BN running statistics); integer buffers taken from
    the last state. Equivalent to the reference recipe's top-k epoch weight averaging."""
    out = {}
    for k, v in states[-1].items():
        if v.is_floating_point():
            out[k] = sum(s[k].float() for s in states).div_(len(states)).to(v.dtype)
        else:
            out[k] = copy.deepcopy(v)
    return out
