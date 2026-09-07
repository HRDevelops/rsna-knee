"""Train the 2.5D slot model for one fold (CLAUDE.md §5, §11; EXECUTION_PLAN Phase 3.1).

Resumable checkpoints, deterministic seed, config snapshot saved next to weights, OOF predictions
saved as parquet, wall-time guard (checkpoints and exits cleanly if approaching the SLURM time
limit rather than getting killed mid-epoch), gold-58 evaluated every run as a tripwire (CLAUDE.md
§7) — never trained on, logged but not used for model selection.

Usage (see slurm/train.sbatch, one job per fold via --array):
    python scripts/train.py --config configs/exp001_dinov2s_slot.yaml --fold 0 \
        --out $RSNA_ROOT/artifacts --resume auto
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.dataset import TARGET_COLUMNS, RSNAKneeDataset, collate_fn  # noqa: E402
from rsna_knee.model import SlotModel, get_param_groups  # noqa: E402


def expand_env(obj):
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
                else:
                    self.shadow[k] = v.clone()

    def apply_to(self, model: nn.Module) -> dict:
        original = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)
        return original

    def restore(self, model: nn.Module, original: dict) -> None:
        model.load_state_dict(original)


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp(min=1e-6)


_GRAD_NORM_GROUPS = {
    "backbone": lambda name: name.startswith("backbone."),
    "slice_agg": lambda name: name.startswith(("slice_pos_embed.", "slice_cls", "slice_transformer.")),
    "study_head": lambda name: name.startswith(
        ("slot_embed.", "slot_transformer.", "target_queries", "target_attn.", "head.")
    ),
}


def component_grad_norms(model: nn.Module) -> dict[str, float]:
    """Grad norm per architectural component, computed BEFORE clipping — used to diagnose which
    part of the from-scratch model (if any) is exploding/vanishing early in training."""
    sums = {k: 0.0 for k in _GRAD_NORM_GROUPS}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        norm_sq = p.grad.detach().float().norm(2).item() ** 2
        for group, pred in _GRAD_NORM_GROUPS.items():
            if pred(name):
                sums[group] += norm_sq
                break
    return {k: v ** 0.5 for k, v in sums.items()}


@torch.no_grad()
def run_inference(model: SlotModel, loader: DataLoader, device: str, autocast_dtype) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        volumes = batch["volumes"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            logits = model(volumes, mask)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        for uid, p in zip(batch["study_uid"], probs):
            row = {"study_uid": uid}
            row.update({c: float(v) for c, v in zip(TARGET_COLUMNS, p)})
            rows.append(row)
    return pd.DataFrame(rows)


def macro_auc(pred_df: pd.DataFrame, truth: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    matched = pred_df[pred_df["study_uid"].isin(truth)].reset_index(drop=True)
    truth_matrix = np.stack([truth[uid] for uid in matched["study_uid"]])  # [N, 12], same row order as matched

    per_label = {}
    for i, col in enumerate(TARGET_COLUMNS):
        y_true = truth_matrix[:, i]
        y_score = matched[col].to_numpy()
        if len(set(y_true)) < 2:
            per_label[col] = float("nan")
            continue
        per_label[col] = roc_auc_score(y_true, y_score)
    valid = [v for v in per_label.values() if not np.isnan(v)]
    return (float(np.mean(valid)) if valid else float("nan")), per_label


def load_gold_truth(gold_csv: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(gold_csv).rename(columns={"StudyInstanceUID": "study_uid"})
    df = df[df[TARGET_COLUMNS[0]].notna()]
    return {
        r["study_uid"]: np.array([float(r[c]) for c in TARGET_COLUMNS])
        for r in df.to_dict("records")
    }


def load_soft_truth(labels_parquet: Path) -> dict[str, np.ndarray]:
    df = pd.read_parquet(labels_parquet)
    return {
        r["study_uid"]: np.array([1.0 if r[f"{c}__probability"] >= 0.5 else 0.0 for c in TARGET_COLUMNS])
        for r in df.to_dict("records")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=str, default="auto", choices=["auto", "none"])
    parser.add_argument("--raw", type=Path, default=None, help="data/raw (for gold-58 truth); default $RSNA_ROOT/data/raw")
    parser.add_argument("--cache-manifests", type=Path, default=None)
    parser.add_argument("--max-hours", type=float, default=7.0, help="wall-clock guard, leaves margin under sbatch --time")
    parser.add_argument("--debug-grad-norms", action="store_true",
                         help="log per-component grad norms every 10 batches (fold-collapse diagnosis)")
    parser.add_argument("--seed-override", type=int, default=None,
                         help="override config seed, for multi-seed stability sweeps")
    parser.add_argument("--eval-raw-too", action="store_true",
                         help="also run OOF/gold58 eval with raw (non-EMA) weights, for comparison")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = expand_env(yaml.safe_load(f))
    seed = args.seed_override if args.seed_override is not None else cfg["seed"]
    set_seed(seed)

    rsna_root = Path(os.environ.get("RSNA_ROOT", args.out.parent))
    raw_dir = args.raw or (rsna_root / "data" / "raw")
    cache_manifests = args.cache_manifests or (rsna_root / "artifacts" / "cache_manifests")

    fold_dir_name = f"fold{args.fold}" if args.seed_override is None else f"fold{args.fold}_seed{seed}"
    exp_dir = args.out / cfg["experiment"] / fold_dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, exp_dir / "config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if cfg["train"]["precision"] == "bf16" else torch.float16

    common_kwargs = dict(
        cfg=cfg,
        cache_dir=Path(cfg["data"]["cache_dir"]),
        train_series_csv=raw_dir / "train_series.csv",
        cache_manifest_dir=cache_manifests,
        labels_parquet=Path(cfg["data"]["labels"]),
        folds_parquet=Path(cfg["data"]["folds"]),
        final_side_csv=rsna_root / "artifacts" / "final_side.csv",
        decode_audit_parquet=rsna_root / "artifacts" / "decode_audit.parquet",
        fold=args.fold,
    )
    train_ds = RSNAKneeDataset(split="train", **common_kwargs)
    val_ds = RSNAKneeDataset(split="val", **common_kwargs)
    gold_ds = RSNAKneeDataset(split="gold", **common_kwargs)
    print(f"train={len(train_ds)} val={len(val_ds)} gold={len(gold_ds)}", file=sys.stderr)

    bs = cfg["train"]["batch_size_studies"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=8,
                               collate_fn=collate_fn, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate_fn)
    gold_loader = DataLoader(gold_ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = SlotModel(cfg["model"]["backbone"], num_targets=cfg["model"]["num_targets"]).to(device)
    param_groups = get_param_groups(model, cfg["train"]["lr"], cfg["train"]["backbone_lr_mult"],
                                     cfg["train"]["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = build_scheduler(optimizer, cfg["train"]["warmup_epochs"], cfg["train"]["epochs"],
                                 len(train_loader))
    ema = EMA(model, cfg["train"]["ema"])

    start_epoch = 0
    ckpt_path = exp_dir / "checkpoint.pt"
    if args.resume == "auto" and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        ema.shadow = ckpt["ema"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}", file=sys.stderr)

    gold_truth = load_gold_truth(raw_dir / "train.csv")
    val_truth = load_soft_truth(Path(cfg["data"]["labels"]))

    t_start = time.time()
    max_seconds = args.max_hours * 3600
    epochs = cfg["train"]["epochs"]

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            volumes = batch["volumes"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(volumes, mask)
                loss = weighted_bce_loss(logits, labels, weights) if cfg["train"]["label_confidence_weighting"] \
                    else nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            if args.debug_grad_norms and n_batches % 10 == 0:
                norms = component_grad_norms(model)
                print(f"  epoch {epoch} batch {n_batches}: loss={loss.item():.4f} "
                      f"grad_norms={ {k: round(v, 4) for k, v in norms.items()} }", file=sys.stderr)
            # Clip backbone and head/slice-agg params SEPARATELY, not with one combined norm.
            # Diagnostic finding (2026-09-07): backbone grad norm is wildly spiky (1-25+) while
            # the from-scratch head sits at 0.1-4 — a single global clip_grad_norm_ computes one
            # scale factor from the combined norm, so a backbone spike forces the SAME aggressive
            # down-scaling onto the tiny, freshly-initialized head gradients too, intermittently
            # starving their learning signal. This is very likely the actual collapse mechanism.
            for group in param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], 1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

            if time.time() - t_start > max_seconds:
                print(f"Wall-clock guard triggered mid-epoch {epoch} — checkpointing and exiting.",
                      file=sys.stderr)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "ema": ema.shadow, "epoch": epoch - 1},
                           ckpt_path)
                return

        print(f"epoch {epoch}: train_loss={epoch_loss / max(n_batches, 1):.4f}", file=sys.stderr)

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "ema": ema.shadow, "epoch": epoch}, ckpt_path)

        if time.time() - t_start > max_seconds:
            print(f"Wall-clock guard triggered after epoch {epoch} — stopping early.", file=sys.stderr)
            break

    if args.eval_raw_too:
        raw_oof_df = run_inference(model, val_loader, device, autocast_dtype)
        raw_macro, raw_per_label = macro_auc(raw_oof_df, val_truth)
        print(f"[raw, non-EMA] OOF macro-AUC fold {args.fold}: {raw_macro:.4f} "
              f"(EMA masking check: compare against the EMA number below)", file=sys.stderr)
        raw_oof_df.to_parquet(exp_dir / "oof_raw.parquet", index=False)

    # Final eval with EMA weights.
    original_state = ema.apply_to(model)

    oof_df = run_inference(model, val_loader, device, autocast_dtype)
    oof_path = exp_dir / "oof.parquet"
    oof_df.to_parquet(oof_path, index=False)
    val_macro, val_per_label = macro_auc(oof_df, val_truth)
    print(f"OOF macro-AUC (vs soft labels, val fold {args.fold}): {val_macro:.4f}", file=sys.stderr)
    for k, v in val_per_label.items():
        print(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: n/a", file=sys.stderr)

    gold_df = run_inference(model, gold_loader, device, autocast_dtype)
    gold_path = exp_dir / "gold58_predictions.parquet"
    gold_df.to_parquet(gold_path, index=False)
    gold_macro, gold_per_label = macro_auc(gold_df, gold_truth)
    print(f"Gold-58 tripwire macro-AUC: {gold_macro:.4f}", file=sys.stderr)
    for k, v in gold_per_label.items():
        print(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: n/a", file=sys.stderr)

    ema.restore(model, original_state)

    summary = {
        "experiment": cfg["experiment"], "fold": args.fold, "epochs_completed": epoch,
        "val_macro_auc": val_macro, "gold58_macro_auc": gold_macro,
        **{f"val_{k}": v for k, v in val_per_label.items()},
        **{f"gold_{k}": v for k, v in gold_per_label.items()},
    }
    pd.DataFrame([summary]).to_csv(exp_dir / "summary.csv", index=False)
    print(f"Done. OOF -> {oof_path}, gold58 -> {gold_path}, summary -> {exp_dir / 'summary.csv'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
