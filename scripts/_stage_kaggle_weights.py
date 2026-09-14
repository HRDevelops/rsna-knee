"""One-off: strip best.pt to ema-only (drop the unused 'model' key) into a clean Kaggle staging
dir per experiment, excluding checkpoint.pt/oof.parquet/etc. Reports size before/after."""
import sys
from pathlib import Path
import torch

RSNA_ROOT = Path("/shared/home/husnain.rasool/rsna-knee")
STAGING = RSNA_ROOT / "kaggle_staging"

EXPERIMENTS = {
    "exp003_dinov2b_slot_v2labels": "rsna-knee-exp003-weights",
    "exp005_dinov2s_slot_v3labels": "rsna-knee-exp005-weights",
    "exp006_dinov2b_slot_v3labels": "rsna-knee-exp006-weights",
    "exp008_dinov2s_slot_distilled": "rsna-knee-exp008-weights",
}

for exp_name, slug in EXPERIMENTS.items():
    src_dir = RSNA_ROOT / "artifacts" / exp_name
    dst_dir = STAGING / slug
    dst_dir.mkdir(parents=True, exist_ok=True)

    total_before = total_after = 0
    for fold_dir in sorted(src_dir.glob("fold*")):
        src_ckpt = fold_dir / "best.pt"
        if not src_ckpt.exists():
            print(f"  [WARN] {src_ckpt} missing, skipping", file=sys.stderr)
            continue
        before = src_ckpt.stat().st_size
        total_before += before

        ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
        lean = {"ema": ckpt["ema"], "epoch": ckpt.get("epoch"),
                "val_macro_auc": ckpt.get("val_macro_auc"), "gold58_macro_auc": ckpt.get("gold58_macro_auc")}

        dst_fold = dst_dir / fold_dir.name
        dst_fold.mkdir(parents=True, exist_ok=True)
        dst_ckpt = dst_fold / "best.pt"
        torch.save(lean, dst_ckpt)
        after = dst_ckpt.stat().st_size
        total_after += after
        print(f"  {exp_name}/{fold_dir.name}: {before/1e6:.0f}MB -> {after/1e6:.0f}MB", file=sys.stderr)

    rank_ref = src_dir / "rank_reference.npz"
    if rank_ref.exists():
        import shutil
        shutil.copy2(rank_ref, dst_dir / "rank_reference.npz")
        print(f"  {exp_name}: copied rank_reference.npz ({rank_ref.stat().st_size/1e6:.1f}MB)", file=sys.stderr)

    print(f"{exp_name} -> {dst_dir}: total {total_before/1e9:.2f}GB -> {total_after/1e9:.2f}GB "
          f"({100*(1-total_after/max(total_before,1)):.0f}% reduction)\n", file=sys.stderr)
