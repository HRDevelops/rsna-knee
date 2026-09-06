"""Dataset for the 2.5D slot model (CLAUDE.md §5, configs/exp001_dinov2s_slot.yaml).

Per study: assigns each of its series to one of 6 canonical slots
{Sagittal, Coronal, Axial} x {Fluid_Sensitive 0/1}, picking the longest series when a slot has
duplicates. Loads the cached uint8 [n_slices,384,384] stack, uniformly samples T slices, resizes
to a DINOv2-patch-14-compatible size, applies laterality normalization (flip right knees), and
(train split only) geometric/intensity augmentation.

Laterality: per CLAUDE.md §5 + the 2026-09-07 strategy review, bilateral-conflict studies (flagged
in artifacts/final_side.csv as source in {conflict_resolved_by_report, conflict_unresolved} — see
docs/LATERALITY_RESOLUTION.md) are handled PER-SERIES using that series' own DICOM Laterality tag,
not the single per-study final_side value every other study uses — a study-level side decision is
wrong for a genuinely bilateral exam.
"""
from __future__ import annotations

import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as TF

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
# Indices (into TARGET_COLUMNS) that swap meaning under a left-right mirror. MCL has no lateral
# counterpart target in this taxonomy, so it is deliberately NOT swapped (config aug comment
# flags this as something to "handle explicitly" — this is that handling: nothing to do).
LATERALITY_SWAP_PAIRS = [(2, 3), (4, 5)]  # (Medial Meniscus, Lateral Meniscus), (Medial OA, Lateral OA)

PLANES = ["Sagittal", "Coronal", "Axial"]
SLOT_NAMES = ["Sagittal_FS", "Sagittal_nonFS", "Coronal_FS", "Coronal_nonFS", "Axial_FS", "Axial_nonFS"]
NUM_SLOTS = 6

_DINOV2_PATCH = 14


def slot_index(plane: str, fluid_sensitive: int) -> int | None:
    if plane not in PLANES:
        return None
    p = PLANES.index(plane)
    fs = 0 if int(fluid_sensitive) == 1 else 1
    return p * 2 + fs


def _nearest_patch_multiple(size: int, patch: int = _DINOV2_PATCH) -> int:
    return max(patch, round(size / patch) * patch)


def load_cache_manifest(manifest_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(manifest_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No cache manifests found under {manifest_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df["ok"]]


def normalize_tag_side(v) -> str | None:
    v = str(v or "").strip().upper()
    if v in ("R", "RIGHT"):
        return "R"
    if v in ("L", "LEFT"):
        return "L"
    if v == "B":
        return "BILATERAL"
    return None


def resolve_side(
    study_uid: str,
    series_uid: str,
    bilateral_conflict_studies: set[str],
    series_tag_side: dict[tuple[str, str], str | None],
    final_side_map: dict[str, str | None],
) -> str | None:
    """Per-study side, EXCEPT for studies flagged as having a genuine cross-series L/R tag
    conflict (docs/LATERALITY_RESOLUTION.md) — those are almost certainly bilateral exams, so a
    single per-study side decision is wrong; each series there uses its OWN DICOM tag instead.
    Pulled out as a standalone function so it's unit-testable without constructing a full dataset
    (see tests/test_laterality.py)."""
    if study_uid in bilateral_conflict_studies:
        return series_tag_side.get((study_uid, series_uid))
    side = final_side_map.get(study_uid)
    return side if isinstance(side, str) else None


class RSNAKneeDataset(Dataset):
    def __init__(
        self,
        cfg: dict,
        cache_dir: Path,
        train_series_csv: Path,
        cache_manifest_dir: Path,
        labels_parquet: Path,
        folds_parquet: Path,
        final_side_csv: Path,
        decode_audit_parquet: Path,
        fold: int,
        split: str,  # "train" | "val" | "gold"
    ):
        assert split in ("train", "val", "gold")
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.is_train = split == "train"
        self.T = cfg["data"]["slices_per_series"]
        self.out_size = _nearest_patch_multiple(cfg["data"]["image_size"])
        aug = cfg.get("aug", {})
        self.aug_cfg = aug

        series_df = pd.read_csv(train_series_csv).rename(
            columns={"StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid"}
        )
        manifest_df = load_cache_manifest(cache_manifest_dir)[["study_uid", "series_uid", "n_slices"]]
        series_df = series_df.merge(manifest_df, on=["study_uid", "series_uid"], how="inner")

        folds_df = pd.read_parquet(folds_parquet)[["study_uid", "fold"]]
        if split == "gold":
            study_uids = set(folds_df.loc[folds_df["fold"] == -1, "study_uid"])
        elif split == "val":
            study_uids = set(folds_df.loc[folds_df["fold"] == fold, "study_uid"])
        else:
            study_uids = set(folds_df.loc[(folds_df["fold"] != fold) & (folds_df["fold"] != -1), "study_uid"])

        labels_df = pd.read_parquet(labels_parquet).set_index("study_uid")
        self.has_confidence_weight = f"{TARGET_COLUMNS[0]}__confidence_weight" in labels_df.columns
        self.labels_df = labels_df

        final_side_df = pd.read_csv(final_side_csv)
        self.final_side_map = final_side_df.set_index("study_uid")["final_side"].to_dict()
        self.bilateral_conflict_studies = set(
            final_side_df.loc[
                final_side_df["source"].isin(["conflict_resolved_by_report", "conflict_unresolved"]),
                "study_uid",
            ]
        )

        decode_audit_df = pd.read_parquet(decode_audit_parquet)[["study_uid", "series_uid", "tag_Laterality"]]
        self.series_tag_side = {
            (row.study_uid, row.series_uid): normalize_tag_side(row.tag_Laterality)
            for row in decode_audit_df.itertuples()
        }

        study_uids &= set(labels_df.index)
        study_uids &= set(series_df["study_uid"])
        self.study_uids = sorted(study_uids)

        self.slot_map: dict[str, dict[int, tuple[str, int]]] = {}  # study -> slot_idx -> (series_uid, n_slices)
        for study_uid, group in series_df[series_df["study_uid"].isin(study_uids)].groupby("study_uid"):
            slots: dict[int, tuple[str, int]] = {}
            for row in group.itertuples():
                idx = slot_index(row.Anatomical_Plane, row.Fluid_Sensitive)
                if idx is None:
                    continue
                if idx not in slots or row.n_slices > slots[idx][1]:
                    slots[idx] = (row.series_uid, row.n_slices)
            self.slot_map[study_uid] = slots

        self.study_uids = [u for u in self.study_uids if self.slot_map.get(u)]

    def __len__(self) -> int:
        return len(self.study_uids)

    def _get_side(self, study_uid: str, series_uid: str) -> str | None:
        return resolve_side(study_uid, series_uid, self.bilateral_conflict_studies,
                             self.series_tag_side, self.final_side_map)

    def _load_slot_volume(self, study_uid: str, series_uid: str) -> np.ndarray:
        path = self.cache_dir / study_uid / f"{series_uid}.npy"
        vol = np.load(path)  # [n_slices, 384, 384] uint8
        n = vol.shape[0]
        idx = np.linspace(0, n - 1, self.T).round().astype(int)
        idx = np.clip(idx, 0, n - 1)
        return vol[idx]  # [T, 384, 384]

    def _labels_for(self, study_uid: str) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.labels_df.loc[study_uid]
        probs = torch.tensor([float(row[f"{c}__probability"]) for c in TARGET_COLUMNS], dtype=torch.float32)
        if self.has_confidence_weight:
            weights = torch.tensor(
                [float(row[f"{c}__confidence_weight"]) for c in TARGET_COLUMNS], dtype=torch.float32
            )
        else:
            weights = torch.ones(len(TARGET_COLUMNS), dtype=torch.float32)
        return probs, weights

    def __getitem__(self, idx: int) -> dict:
        study_uid = self.study_uids[idx]
        slots = dict(self.slot_map[study_uid])  # copy, may be mutated by fluid-flag-corrupt aug

        if self.is_train and self.aug_cfg.get("fluid_flag_corrupt_prob", 0) > 0:
            if random.random() < self.aug_cfg["fluid_flag_corrupt_prob"]:
                plane = random.choice(range(3))
                fs_idx, nonfs_idx = plane * 2, plane * 2 + 1
                a, b = slots.get(fs_idx), slots.get(nonfs_idx)
                if a is not None or b is not None:
                    slots[fs_idx], slots[nonfs_idx] = b, a

        volumes = torch.zeros(NUM_SLOTS, self.T, self.out_size, self.out_size, dtype=torch.float32)
        mask = torch.zeros(NUM_SLOTS, dtype=torch.bool)

        do_lat_swap = self.is_train and random.random() < self.aug_cfg.get("laterality_swap_prob", 0)
        geom_params = self._sample_geometric_params() if self.is_train else None

        for slot_idx, entry in slots.items():
            if entry is None:
                continue
            series_uid, _ = entry
            vol = self._load_slot_volume(study_uid, series_uid)  # [T,384,384] uint8
            t = torch.from_numpy(vol.astype(np.float32) / 255.0).unsqueeze(1)  # [T,1,384,384]

            side = self._get_side(study_uid, series_uid)
            flip = side == "R"
            if do_lat_swap:
                flip = not flip
            if flip:
                t = torch.flip(t, dims=[-1])

            t = F.interpolate(t, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)

            if self.is_train:
                t = self._augment(t, geom_params)

            volumes[slot_idx] = t.squeeze(1)
            mask[slot_idx] = True

        probs, weights = self._labels_for(study_uid)
        if do_lat_swap:
            probs = probs.clone()
            weights = weights.clone()
            for i, j in LATERALITY_SWAP_PAIRS:
                probs[i], probs[j] = probs[j].item(), probs[i].item()
                weights[i], weights[j] = weights[j].item(), weights[i].item()

        return {
            "study_uid": study_uid,
            "volumes": volumes,       # [6, T, H, W], single-channel intensity
            "mask": mask,             # [6] bool, True = slot present
            "labels": probs,          # [12] soft targets
            "weights": weights,       # [12] per-target confidence weight
        }

    def _sample_geometric_params(self) -> dict:
        a = self.aug_cfg
        return {
            "angle": random.uniform(-a.get("rotate_deg", 0), a.get("rotate_deg", 0)),
            "scale": 1.0 + random.uniform(-a.get("scale", 0), a.get("scale", 0)),
            "translate": (
                random.uniform(-a.get("translate", 0), a.get("translate", 0)) * self.out_size,
                random.uniform(-a.get("translate", 0), a.get("translate", 0)) * self.out_size,
            ),
        }

    def _augment(self, t: torch.Tensor, geom: dict) -> torch.Tensor:
        # t: [T,1,H,W]. Same affine params applied across the whole T-stack (temporally consistent).
        t = TF.affine(
            t, angle=geom["angle"], translate=list(geom["translate"]), scale=geom["scale"], shear=[0.0, 0.0]
        )
        gain = 1.0 + random.uniform(-1, 1) * self.aug_cfg.get("intensity_jitter", 0)
        t = (t * gain).clamp(0, 1)
        if self.aug_cfg.get("coarse_dropout", False) and random.random() < 0.5:
            h, w = t.shape[-2:]
            ph, pw = random.randint(h // 8, h // 4), random.randint(w // 8, w // 4)
            y0, x0 = random.randint(0, h - ph), random.randint(0, w - pw)
            t[:, :, y0:y0 + ph, x0:x0 + pw] = 0.0
        return t


def collate_fn(batch: list[dict]) -> dict:
    return {
        "study_uid": [b["study_uid"] for b in batch],
        "volumes": torch.stack([b["volumes"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "weights": torch.stack([b["weights"] for b in batch]),
    }
