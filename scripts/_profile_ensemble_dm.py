import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
import pandas as pd
import torch
from rsna_knee.dataset import TARGET_COLUMNS, _patch_size_for_backbone
from rsna_knee.infer_core import decode_study_slots, load_ensemble_recipe, nearest_patch_multiple, run_study_inference
from rsna_knee.model import SlotModel

RSNA_ROOT = Path("/shared/home/husnain.rasool/rsna-knee")
device = "cuda"
loaded = load_ensemble_recipe(RSNA_ROOT / "code" / "src" / "rsna_knee" / "ensemble_recipe_v1.yaml")
members, recipe = loaded["members"], loaded["recipe"]
out_size = nearest_patch_multiple(384, _patch_size_for_backbone(next(iter(members.values()))["backbone"]))

member_models = {}
for name, spec in members.items():
    wdir = RSNA_ROOT / "artifacts" / name
    ms = []
    for fold_dir in sorted(wdir.glob("fold*")):
        ckpt = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
        m = SlotModel(spec["backbone"], num_targets=len(TARGET_COLUMNS), pretrained=False).to(device)
        m.load_state_dict(ckpt["ema"])
        m.eval()
        ms.append(m)
    member_models[name] = ms
    print(f"{name}: {len(ms)}/5 folds", file=sys.stderr)

raw = RSNA_ROOT / "artifacts" / "synthetic_loadtest"
test_df = pd.read_csv(raw / "test.csv").iloc[:20].reset_index(drop=True)
series_df = pd.read_csv(raw / "test_series.csv").rename(columns={"StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid"})
series_root = raw / "test_series"
uids = list(test_df["StudyInstanceUID"])

t0 = time.time()
decoded = {}
for uid in uids:
    study_series = series_df[series_df["study_uid"] == uid].to_dict("records")
    slots, tag_sides = decode_study_slots(uid, study_series, series_root / uid, 24, out_size)
    decoded[uid] = (slots, tag_sides)
decode_elapsed = time.time() - t0
print(f"Decode (serial): {decode_elapsed:.1f}s total, {decode_elapsed/len(uids):.3f}s/study", file=sys.stderr)

t0 = time.time()
for uid in uids:
    slots, tag_sides = decoded[uid]
    for name, models in member_models.items():
        run_study_inference(slots, tag_sides, models, device, 24, out_size, torch.float16)
model_elapsed = time.time() - t0
print(f"Model (3 members x 5-fold): {model_elapsed:.1f}s total, {model_elapsed/len(uids):.3f}s/study", file=sys.stderr)

total = decode_elapsed + model_elapsed
print(f"\nSplit: decode {decode_elapsed/total:.1%}, model {model_elapsed/total:.1%} of {total:.1f}s combined", file=sys.stderr)
