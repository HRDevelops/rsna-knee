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
import subprocess
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
# SlotModel/get_param_groups (2D) vs Slot3DModel/get_param_groups (exp007 3D) are imported inside
# main() based on cfg["model"]["family"] — see the model_family branch below.


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


def kl_bce_mix_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor,
                     distill_labels: torch.Tensor, distill_weight: float) -> torch.Tensor:
    """exp008 (efficiency-track distillation): mix of BCE-to-labels_v3 (the usual supervised term,
    weighted by per-study/per-label confidence same as always) and KL-divergence-to-the-ensemble-
    teacher's soft predictions (unweighted — the teacher's own per-label agreement-based confidence
    is already baked into ensemble_teacher_v1.parquet's blend, no need to double-apply weighting
    here). Both targets are Bernoulli (independent per-label sigmoid outputs, not a softmax
    distribution), so KL is computed per-label via the binary-KL identity, not nn.KLDivLoss (which
    assumes a categorical/softmax distribution over the last dim)."""
    bce = weighted_bce_loss(logits, labels, weights)
    probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    teacher = distill_labels.clamp(1e-6, 1 - 1e-6)
    kl = teacher * (teacher.log() - probs.log()) + (1 - teacher) * ((1 - teacher).log() - (1 - probs).log())
    kl = kl.mean()
    return (1 - distill_weight) * bce + distill_weight * kl


def quota_headroom_gb(min_gb: float = 15.0) -> tuple[float | None, bool]:
    """BeeGFS quota headroom in GB, or (None, True) if the check itself fails — a broken quota
    probe must never block training. See docs/EXPERIMENTS.md 2026-09-08: the shared 400GiB quota
    across ALL of the user's HPC projects (not just this one) caused a real mid-training push
    failure once; this guard is the checkpoint-write-side backstop for the same risk."""
    try:
        out = subprocess.run(
            ["beegfs-ctl", "--getquota", "--uid", str(os.getuid()), "--csv"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        # Output has a banner line + blank line + CSV header + one data row — the data row is
        # always last, so anchor on that rather than trying to filter out the banner text.
        data_line = [l for l in out.strip().splitlines() if l][-1]
        _, _, used, hard = data_line.split(",")[:4]
        headroom_gb = (float(hard) - float(used)) / 1e9
        return headroom_gb, headroom_gb >= min_gb
    except Exception as e:
        print(f"quota check failed ({e}); proceeding without the guard", file=sys.stderr)
        return None, True


def safe_torch_save(state: dict, path: Path, min_gb: float = 15.0) -> None:
    """Checkpoint write guarded by quota headroom — logs loudly and SKIPS the write instead of
    crashing the run when headroom is tight (CLAUDE.md §11 resumability: a skipped checkpoint just
    means resume falls back to the previous one, whereas a mid-write ENOSPC can corrupt it).

    2026-09-08 bug (exp003, all 5 folds): the wall-clock-triggered save is the LAST CHANCE to
    preserve a run's progress before the process exits — applying the same 15GB comfort floor to
    it as to a routine per-epoch save (which gets many more chances to succeed later) meant every
    exp003 fold lost 23/25 epochs of a healthy, nearly-finished run because headroom sat at
    ~11.5GB, above what an actual ENOSPC would need but below the floor. Callers on a last-chance
    path should pass a much lower min_gb (sized to the actual checkpoint, not a round comfort
    number) so the write is only skipped when it would truly risk corrupting the filesystem."""
    headroom_gb, ok = quota_headroom_gb(min_gb)
    if not ok:
        print(f"QUOTA GUARD: headroom {headroom_gb:.1f}GB < {min_gb}GB floor — "
              f"SKIPPING checkpoint write to {path}", file=sys.stderr)
        return
    torch.save(state, path)


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
def run_inference(model: nn.Module, loader: DataLoader, device: str, autocast_dtype) -> pd.DataFrame:
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

    # Pre-flight quota check (2026-09-09: exp008 ran all 5 folds for ~2h with headroom already
    # below the 15GB routine-save floor from the start — every single checkpoint write across the
    # whole run was silently skipped, so 4/5 folds "completed" with zero deployable weights.
    # safe_torch_save's per-write guard is correct but silent-by-design; this catches the
    # already-too-tight case loudly at job START instead of discovering it hours later).
    startup_headroom, startup_ok = quota_headroom_gb(min_gb=15.0)
    if not startup_ok:
        print(f"QUOTA GUARD: starting with headroom {startup_headroom}GB, already below the "
              f"15GB routine-save floor — every checkpoint write this run will be skipped until "
              f"headroom recovers. Proceeding anyway (eval-only artifacts still save), but no "
              f"weights will be deployable unless quota is freed before this run finishes.",
              file=sys.stderr)

    rsna_root = Path(os.environ.get("RSNA_ROOT", args.out.parent))
    raw_dir = args.raw or (rsna_root / "data" / "raw")
    cache_manifests = args.cache_manifests or (rsna_root / "artifacts" / "cache_manifests")

    fold_dir_name = f"fold{args.fold}" if args.seed_override is None else f"fold{args.fold}_seed{seed}"
    exp_dir = args.out / cfg["experiment"] / fold_dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, exp_dir / "config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if cfg["train"]["precision"] == "bf16" else torch.float16

    distill_labels = cfg["data"].get("distill_labels")
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
        distill_labels_parquet=Path(distill_labels) if distill_labels else None,
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

    model_family = cfg["model"].get("family", "vit2d")
    if model_family == "resnet3d":
        # exp007: 3D ResNet-50 ensemble-diversity family (src/rsna_knee/model_3d.py) — a separate
        # module, not a branch inside SlotModel, so the working 2D path (exp001-006) is untouched.
        from rsna_knee.model_3d import Slot3DModel, get_param_groups as get_param_groups_fn
        model = Slot3DModel(num_targets=cfg["model"]["num_targets"]).to(device)
    else:
        from rsna_knee.model import SlotModel, get_param_groups as get_param_groups_fn
        # target_slot_prior: optional {target_idx: [slot_idx, ...]} anatomy-aware init bias
        # (2026-09-17 ACL workstream) -- config keys/values are plain ints/lists from YAML, dict
        # keys arrive as ints already since YAML mapping keys aren't quoted in our configs.
        target_slot_prior = cfg["model"].get("target_slot_prior")
        model = SlotModel(cfg["model"]["backbone"], num_targets=cfg["model"]["num_targets"],
                           target_slot_prior=target_slot_prior).to(device)
    param_groups = get_param_groups_fn(model, cfg["train"]["lr"], cfg["train"]["backbone_lr_mult"],
                                        cfg["train"]["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups)
    # Gradient accumulation (2026-09-17, exp010 resolution/slice-count ablation): defaults to 1
    # (a no-op — every existing config without this field steps every batch exactly as before,
    # byte-for-byte the same training dynamics as exp002-009). Set > 1 to raise T/image_size
    # without cutting the effective batch size when a higher-res config no longer fits the same
    # micro-batch in VRAM: batch_size_studies becomes the per-step micro-batch, and the EFFECTIVE
    # batch is batch_size_studies * grad_accum_steps.
    grad_accum_steps = cfg["train"].get("grad_accum_steps", 1)
    # 2026-09-18 bugfix: scheduler.step() fires once per accumulation WINDOW, not once per batch, so
    # steps_per_epoch must be the number of optimizer steps per epoch. The first version of grad
    # accumulation passed len(train_loader) here, which with grad_accum_steps=2 made warmup last
    # twice as many epochs and stopped the cosine decay at its midpoint (found while planning
    # exp015; affected only exp014, already void). A no-op when grad_accum_steps == 1.
    scheduler = build_scheduler(optimizer, cfg["train"]["warmup_epochs"], cfg["train"]["epochs"],
                                 -(-len(train_loader) // grad_accum_steps))
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
    epoch = start_epoch - 1  # defined even if the loop body never runs (start_epoch >= epochs —
    # e.g. resuming with epochs set equal to the checkpoint's epoch, to force an eval-only pass
    # without further training); reflects "last completed epoch" correctly in both cases.

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        window_has_nan = False
        for batch in train_loader:
            volumes = batch["volumes"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(volumes, mask)
                if cfg["train"]["loss"] == "kl_bce_mix":
                    distill_labels = batch["distill_labels"].to(device, non_blocking=True)
                    loss = kl_bce_mix_loss(logits, labels, weights, distill_labels,
                                            cfg["train"]["distill_weight"])
                elif cfg["train"]["label_confidence_weighting"]:
                    loss = weighted_bce_loss(logits, labels, weights)
                else:
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            # NaN/Inf guard, accumulation-aware: a poisoned micro-batch's gradient, once summed
            # into the accumulator, corrupts every other micro-batch's gradient in the SAME window
            # (unlike the grad_accum_steps=1 case, it can't be skipped in isolation) — so flag the
            # whole window and skip the step for all of it, not just this micro-batch.
            if not torch.isfinite(loss):
                print(f"  epoch {epoch} batch {n_batches}: non-finite loss ({loss.item()}) — "
                      f"poisons this accumulation window, will skip its optimizer step.", file=sys.stderr)
                window_has_nan = True
                n_batches += 1
                is_window_end = (n_batches % grad_accum_steps == 0) or (n_batches == len(train_loader))
                if is_window_end:
                    optimizer.zero_grad(set_to_none=True)
                    window_has_nan = False
                continue
            (loss / grad_accum_steps).backward()
            if args.debug_grad_norms and n_batches % 10 == 0:
                norms = component_grad_norms(model)
                print(f"  epoch {epoch} batch {n_batches}: loss={loss.item():.4f} "
                      f"grad_norms={ {k: round(v, 4) for k, v in norms.items()} }", file=sys.stderr)

            epoch_loss += loss.item()
            n_batches += 1
            is_window_end = (n_batches % grad_accum_steps == 0) or (n_batches == len(train_loader))
            if not is_window_end:
                continue

            if window_has_nan:
                print(f"  epoch {epoch} batch {n_batches}: skipping optimizer step — a non-finite "
                      f"loss landed earlier in this accumulation window.", file=sys.stderr)
                optimizer.zero_grad(set_to_none=True)
                window_has_nan = False
                continue

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
            optimizer.zero_grad(set_to_none=True)

            if time.time() - t_start > max_seconds:
                print(f"Wall-clock guard triggered mid-epoch {epoch} — checkpointing and exiting.",
                      file=sys.stderr)
                # Last-chance save: use a much lower floor than the routine per-epoch save (below).
                # Skipping THIS one means total loss of the run's progress, not just a delayed
                # retry next epoch — see the safe_torch_save docstring for the 2026-09-08 incident
                # this fixes (exp003 lost 23/25 epochs on all 5 folds this way).
                safe_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                  "scheduler": scheduler.state_dict(), "ema": ema.shadow, "epoch": epoch - 1},
                                 ckpt_path, min_gb=3.0)
                return

        print(f"epoch {epoch}: train_loss={epoch_loss / max(n_batches, 1):.4f}", file=sys.stderr)

        # "last" checkpoint only — this is the sole checkpoint file per fold (overwritten every
        # epoch, not accumulated), so there is no per-epoch retention to prune. "best" (lean,
        # resume-state-free) is written once below, after the single end-of-training eval.
        safe_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
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

    # Lean "best" weights (model + EMA shadow only, no optimizer/scheduler state) — a fraction of
    # checkpoint.pt's size, and what notebooks/submit-*.ipynb should load (CLAUDE.md §9), not the
    # full resume checkpoint. Written once here since eval currently only runs at end of training;
    # if per-epoch validation is added later, gate this write on val_macro improving instead.
    safe_torch_save({"model": model.state_dict(), "ema": ema.shadow, "epoch": epoch,
                      "val_macro_auc": val_macro, "gold58_macro_auc": gold_macro},
                     exp_dir / "best.pt")

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
