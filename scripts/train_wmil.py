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

"""exp018 trainer: adjacent-slice Window-MIL with a fully fine-tuned CoAtNet-2 (rsna_knee.wmil).

One fold per invocation (same fold definition as every earlier experiment; fold -1 = gold-58, never trained
on and never used for selection). Per epoch: train on k_train random windows per study, evaluate on ALL
candidate windows of the held-out fold, keep the top-3 epochs by FOLD-VALIDATION macro-AUC; at the end average
those three epochs' weights and write the OOF / gold-58 predictions of the averaged model.

Outputs match scripts/train.py so compare_runs.py / build_ensemble.py work unchanged:
artifacts/<experiment>/fold<k>/{oof.parquet, gold58_predictions.parquet, summary.csv, best.pt, config.yaml}.

    python scripts/train_wmil.py --config configs/exp018_coatnet2_wmil_v3labels.yaml --fold 0 --out artifacts
    python scripts/train_wmil.py ... --smoke        # 2 epochs on ~40 studies, wiring check
"""
from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsna_knee.wmil import (  # noqa: E402
    MAX_WINDOWS, NUM_SLOTS, SLOT_OFF, TARGET_COLUMNS, WindowMIL, average_state_dicts, candidate_centers,
    gather_windows, normalize_orientation, normalize_tag_side, prepare_input, resolve_slot_sides,
)
from rsna_knee.wmil_v2 import WMILv2Dataset, collate_v2, load_manifest_v2  # noqa: E402  (Phase 0B ladder)
from train import (  # noqa: E402  (reuse the exact metric / IO conventions of the deployed pipeline)
    expand_env, load_gold_truth, load_soft_truth, macro_auc, quota_headroom_gb, safe_torch_save, set_seed,
)


# ------------------------------------------------------------------------------------------- dataset
def load_manifest(manifest_dir: Path) -> pd.DataFrame:
    files = sorted(Path(manifest_dir).glob("shard_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cache manifests in {manifest_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df["ok"]].set_index("study_uid")


class WMILDataset(Dataset):
    def __init__(self, uids: list[str], cache_dir: Path, manifest: pd.DataFrame, labels: pd.DataFrame | None,
                 final_side: dict, bilateral: set, train: bool, k: int):
        self.uids, self.cache_dir, self.man, self.labels = uids, Path(cache_dir), manifest, labels
        self.final_side, self.bilateral, self.train, self.k = final_side, bilateral, train, k

    def __len__(self) -> int:
        return len(self.uids)

    def _metas(self, uid: str):
        r = self.man.loc[uid]
        present = np.array([bool(r[f"present_{i}"]) for i in range(NUM_SLOTS)])
        metas = [{"row_x": float(r[f"row_x_{i}"]), "cx": float(r[f"cx_{i}"]), "tag_laterality": r[f"tag_{i}"]}
                 if present[i] else None for i in range(NUM_SLOTS)]
        return present, metas

    def __getitem__(self, i: int) -> dict:
        uid = self.uids[i]
        present, metas = self._metas(uid)
        stack = np.load(self.cache_dir / f"{uid}.npy")
        if uid in self.bilateral:  # genuine bilateral exam: each series uses its own tag
            sides = [normalize_tag_side(m["tag_laterality"]) if m else None for m in metas]
        else:
            fs = self.final_side.get(uid)
            sides = resolve_slot_sides(metas, fs if isinstance(fs, str) else None)
        stack = normalize_orientation(stack, metas, sides)
        cs = candidate_centers(present)
        if self.train:
            if len(cs) == 0:
                cs = np.array([SLOT_OFF[0] + 1])
            rng = np.random.default_rng((torch.initial_seed() + 7919 * i) % (2 ** 32))  # per-worker, per-epoch
            pick = rng.choice(len(cs), size=self.k, replace=len(cs) < self.k)
            centers, wmask = cs[pick], np.ones(self.k, dtype=bool)
        else:
            n = len(cs)
            centers = np.full(MAX_WINDOWS, cs[0] if n else 1, dtype=np.int64)
            wmask = np.zeros(MAX_WINDOWS, dtype=bool)
            centers[:n], wmask[:n] = cs, True
            if n == 0:
                wmask[0] = True
        out = {"uid": uid, "stack": torch.from_numpy(np.ascontiguousarray(stack)),
               "centers": torch.from_numpy(centers.astype(np.int64)), "wmask": torch.from_numpy(wmask)}
        if self.labels is not None:
            out["labels"] = torch.tensor(
                [float(self.labels.loc[uid, f"{c}__probability"]) for c in TARGET_COLUMNS], dtype=torch.float32)
        return out


def collate(batch: list[dict]) -> dict:
    out = {"uid": [b["uid"] for b in batch]}
    for key in ("stack", "centers", "wmask") + (("labels",) if "labels" in batch[0] else ()):
        out[key] = torch.stack([b[key] for b in batch])
    return out


# ------------------------------------------------------------------------------------------- eval
@torch.no_grad()
def predict(model: WindowMIL, loader: DataLoader, res: int, device: str, dtype) -> pd.DataFrame:
    model.eval()
    rows = []
    for b in loader:
        if "windows" in b:                                      # cache v2: windows already gathered
            w = b["windows"].to(device, non_blocking=True)
        else:
            w = gather_windows(b["stack"].to(device, non_blocking=True), b["centers"].to(device))
        wmask = b["wmask"].to(device)
        B, K = w.shape[:2]
        flat = w.flatten(0, 1)
        feats = []
        with torch.autocast(device_type="cuda", dtype=dtype):  # chunked: dense v2 eval has up to 2 x 192 windows
            for s in range(0, flat.shape[0], 96):
                feats.append(model.backbone(prepare_input(flat[s:s + 96].unsqueeze(0), res)).float())
        feats = torch.cat(feats).view(B, K, -1)
        logits = model.head(feats, wmask)
        for uid, p in zip(b["uid"], torch.sigmoid(logits).float().cpu().numpy()):
            rows.append({"study_uid": uid, **{c: float(v) for c, v in zip(TARGET_COLUMNS, p)}})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    ap.add_argument("--max-hours", type=float, default=5.5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = expand_env(yaml.safe_load(args.config.read_text()))
    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    device = "cuda"
    dtype = torch.bfloat16 if cfg["train"].get("precision", "bf16") == "bf16" else torch.float16
    root = Path(os.environ.get("RSNA_ROOT", args.out.parent))
    exp_dir = args.out / cfg["experiment"] / f"fold{args.fold}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        shutil.copy(args.config, exp_dir / "config.yaml")

    hr, ok = quota_headroom_gb(15.0)
    if not ok:
        print(f"QUOTA GUARD: starting with headroom {hr}GB < 15GB; checkpoint writes will be skipped", file=sys.stderr)

    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    v2 = int(d.get("cache_version", 1)) == 2
    manifest = load_manifest_v2(Path(d["manifest_dir"])) if v2 else load_manifest(Path(d["manifest_dir"]))
    labels = pd.read_parquet(d["labels"]).set_index("study_uid")
    folds = pd.read_parquet(d["folds"])[["study_uid", "fold"]]
    side_df = pd.read_csv(d["final_side"])
    final_side = side_df.set_index("study_uid")["final_side"].to_dict()
    bilateral = set(side_df.loc[side_df["source"].isin(["conflict_resolved_by_report", "conflict_unresolved"]), "study_uid"])

    usable = set(manifest["study_uid"] if v2 else manifest.index) & set(labels.index)
    train_u = sorted(usable & set(folds.loc[(folds.fold != args.fold) & (folds.fold != -1), "study_uid"]))
    val_u = sorted(usable & set(folds.loc[folds.fold == args.fold, "study_uid"]))
    gold_u = sorted(usable & set(folds.loc[folds.fold == -1, "study_uid"]))
    assert not (set(train_u) & (set(val_u) | set(gold_u))), "train/val/gold overlap"
    epochs, bs, k = t["epochs"], t["batch_studies"], t["k_train"]
    if args.smoke:
        train_u, val_u, gold_u, epochs, bs, k = train_u[:24], val_u[:8], gold_u[:6], 2, 4, 4
    print(f"fold {args.fold}: train {len(train_u)} | val {len(val_u)} | gold {len(gold_u)} | epochs {epochs} "
          f"| k_train {k} | arch {m['arch']} res {m['res']}", flush=True)

    if v2:
        rung = d.get("v2", {}) or {}
        print(f"cache v2 rung: {rung}", flush=True)
        mk = lambda u, tr, kk: WMILv2Dataset(u, d["cache_dir"], manifest, labels, final_side, bilateral, tr, kk,  # noqa: E731
                                             **rung)
        coll = collate_v2
    else:
        mk = lambda u, tr, kk: WMILDataset(u, d["cache_dir"], manifest, labels, final_side, bilateral, tr, kk)  # noqa: E731
        coll = collate
    train_loader = DataLoader(mk(train_u, True, k), batch_size=bs, shuffle=True, num_workers=args.workers,
                              collate_fn=coll, drop_last=True, persistent_workers=args.workers > 0,
                              pin_memory=True)
    eval_bs = 2
    val_loader = DataLoader(mk(val_u, False, k), batch_size=eval_bs, num_workers=4, collate_fn=coll)
    gold_loader = DataLoader(mk(gold_u, False, k), batch_size=eval_bs, num_workers=2, collate_fn=coll)

    prev = np.clip(np.stack([[labels.loc[u, f"{c}__probability"] for c in TARGET_COLUMNS] for u in train_u]).mean(0),
                   0.03, 0.7)
    pos_weight = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = WindowMIL(m["arch"], pretrained=True, drop=m.get("head_dropout", 0.2),
                      grad_checkpointing=m.get("grad_checkpointing", True),
                      img_size=m.get("img_size"), backbone_kwargs=m.get("backbone_kwargs")).to(device)

    # ---- optional model EMA (exp020). Absent from a config => not constructed, so exp018 is unchanged.
    # decay=0.9999 has a 1/(1-d)=10,000-step horizon, but a fold here is only ~440 steps x 16 epochs =
    # ~7,040 steps: without warm-up the EMA would still carry 0.9999^7040 = 49.5% of the ImageNet INIT
    # at the end of training. ModelEmaV3's warm-up ramps the effective decay from ~0 so the average is
    # never init-dominated; several decays can be tracked at once (one lerp per step each) and the best
    # is chosen by FOLD-VALIDATION AUC only -- gold-58 stays quarantined, as everywhere else here.
    ema_cfg = t.get("ema", {}) or {}
    emas: list = []
    if ema_cfg.get("enabled", False):
        from timm.utils import ModelEmaV3  # noqa: E402
        for dcy in ema_cfg.get("decays", [0.9999]):
            emas.append({"decay": float(dcy),
                         "ema": ModelEmaV3(model, decay=float(dcy),
                                           use_warmup=bool(ema_cfg.get("use_warmup", True))),
                         "best": {"val_auc": -1.0, "epoch": -1}})
        print(f"EMA enabled: decays {[e['decay'] for e in emas]} warmup={ema_cfg.get('use_warmup', True)} "
              f"eval_from_epoch={ema_cfg.get('eval_from_epoch', 0)}", flush=True)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": t["bb_lr"]}, {"params": head, "lr": t["head_lr"]}],
                            weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[t["bb_lr"], t["head_lr"]],
                                                total_steps=max(len(train_loader) * epochs, 1),
                                                pct_start=t["pct_start"])
    val_truth, gold_truth = load_soft_truth(Path(d["labels"])), load_gold_truth(root / "data/raw/train.csv")

    ckpt_path, topk_path = exp_dir / "checkpoint.pt", exp_dir / "topk.json"
    hist: list[dict] = []
    topk: list[dict] = []  # {"epoch", "val_auc", "file"}
    start_epoch = 0
    if args.resume == "auto" and ckpt_path.exists() and not args.smoke:
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        for i, e in enumerate(emas):  # EMA weights must survive the wall-clock-guard resume path
            if "ema" in ck and i < len(ck["ema"]):
                e["ema"].module.load_state_dict(ck["ema"][i]["state"])
                e["best"] = ck["ema"][i]["best"]
        start_epoch, hist = ck["epoch"] + 1, ck["hist"]
        topk = json.loads(topk_path.read_text()) if topk_path.exists() else []
        print(f"resumed from epoch {start_epoch}", flush=True)

    t_start = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        tot, n_b, t0 = 0.0, 0, time.time()
        for b in train_loader:
            if "windows" in b:                                  # cache v2: windows already gathered
                w = b["windows"].to(device, non_blocking=True)
            else:
                w = gather_windows(b["stack"].to(device, non_blocking=True), b["centers"].to(device))
            gain = None
            if cfg["aug"].get("intensity_jitter", 0) > 0:
                gain = 1.0 + (torch.rand(w.shape[0], device=device) * 2 - 1) * cfg["aug"]["intensity_jitter"]
            y = b["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype):
                x = prepare_input(w, m["res"], gain).view(w.shape[0], w.shape[1], 3, m["res"], m["res"])
                logits = model(x)
            loss = lossf(logits.float(), y)
            if not torch.isfinite(loss):
                print("non-finite loss; skipping batch", flush=True)
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step(); sched.step()
            for e in emas:
                e["ema"].update(model)
            tot += loss.item(); n_b += 1
        val_df = predict(model, val_loader, m["res"], device, dtype)
        val_auc, _ = macro_auc(val_df, val_truth)
        rec = {"epoch": epoch, "train_loss": tot / max(n_b, 1), "val_auc": val_auc, "sec": time.time() - t0}

        # EMA branches are evaluated on the SAME fold-validation set and selected on it alone. Skipping
        # the first few epochs costs nothing informative (an early EMA trails a fast-moving model) and
        # keeps the epoch-time overhead down.
        for e in emas:
            if epoch < int(ema_cfg.get("eval_from_epoch", 0)):
                continue
            ema_df = predict(e["ema"].module, val_loader, m["res"], device, dtype)
            ema_auc, _ = macro_auc(ema_df, val_truth)
            rec[f"val_auc_ema{e['decay']}"] = ema_auc
            print(f"  EMA(decay={e['decay']}): fold-val macro-AUC {ema_auc:.4f}", flush=True)
            if ema_auc > e["best"]["val_auc"]:
                e["best"] = {"val_auc": ema_auc, "epoch": epoch}
                if True:  # saved even under --smoke, so the smoke run exercises the finalisation block
                    safe_torch_save(
                        {"model": {kk: v.detach().cpu() for kk, v in e["ema"].module.state_dict().items()},
                         "arch": m["arch"], "res": m["res"], "img_size": m.get("img_size"),
                         "backbone_kwargs": m.get("backbone_kwargs"), "targets": TARGET_COLUMNS, "ema_decay": e["decay"], "epoch": epoch, "val_macro_auc": ema_auc},
                        exp_dir / f"best_ema{e['decay']}.pt", min_gb=5.0)
        hist.append(rec)
        print(f"epoch {epoch}: loss {rec['train_loss']:.4f} | fold-val macro-AUC {val_auc:.4f} | {rec['sec']:.0f}s", flush=True)

        # top-k by FOLD-VALIDATION AUC (gold-58 never consulted)
        if len(topk) < t["topk"] or val_auc > min(e["val_auc"] for e in topk):
            f = exp_dir / f"top_ep{epoch:02d}.pt"
            safe_torch_save({"model": {kk: v.detach().cpu() for kk, v in model.state_dict().items()}}, f, min_gb=5.0)
            topk.append({"epoch": epoch, "val_auc": val_auc, "file": f.name})
            topk.sort(key=lambda e: -e["val_auc"])
            for e in topk[t["topk"]:]:
                (exp_dir / e["file"]).unlink(missing_ok=True)
            del topk[t["topk"]:]
            topk_path.write_text(json.dumps(topk))
        if not args.smoke:
            ck_out = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                      "scheduler": sched.state_dict(), "epoch": epoch, "hist": hist}
            if emas:
                ck_out["ema"] = [{"decay": e["decay"], "state": e["ema"].module.state_dict(),
                                  "best": e["best"]} for e in emas]
            safe_torch_save(ck_out, ckpt_path)
        if time.time() - t_start > args.max_hours * 3600 and epoch < epochs - 1:
            print(f"wall-clock guard after epoch {epoch}; stopping before finalisation (re-run resumes)", flush=True)
            return

    # ---- average the top-k epochs' weights and evaluate the averaged model
    states = [torch.load(exp_dir / e["file"], map_location="cpu")["model"] for e in topk]
    model.load_state_dict(average_state_dicts(states))
    val_df = predict(model, val_loader, m["res"], device, dtype)
    val_df.to_parquet(exp_dir / "oof.parquet", index=False)
    val_macro, val_pl = macro_auc(val_df, val_truth)
    gold_df = predict(model, gold_loader, m["res"], device, dtype)
    gold_df.to_parquet(exp_dir / "gold58_predictions.parquet", index=False)
    gold_macro, gold_pl = macro_auc(gold_df, gold_truth)
    print(f"averaged top-{len(topk)} epochs {[e['epoch'] for e in topk]}: fold-val macro-AUC {val_macro:.4f} | "
          f"gold-58 macro-AUC {gold_macro:.4f}", flush=True)
    for kk, v in gold_pl.items():
        print(f"  gold {kk}: {v:.4f}", flush=True)

    if not args.smoke:
        safe_torch_save({"model": model.state_dict(), "arch": m["arch"], "res": m["res"],
                         "img_size": m.get("img_size"), "backbone_kwargs": m.get("backbone_kwargs"),
                         "targets": TARGET_COLUMNS, "topk_epochs": [e["epoch"] for e in topk],
                         "val_macro_auc": val_macro, "gold58_macro_auc": gold_macro}, exp_dir / "best.pt", min_gb=3.0)
        for e in topk:
            (exp_dir / e["file"]).unlink(missing_ok=True)

    # ---- evaluate each EMA branch head-to-head with the top-k average, on BOTH fold-val and gold-58.
    # Selection already happened (fold-val only); gold-58 here is a read-out, exactly as for the
    # top-k model above, so the comparison the decision gate needs is in one place.
    # Deliberately AFTER best.pt is written and wrapped in try/except: this is the experimental branch,
    # and nothing in it may cost a completed run its primary, exp018-equivalent output.
    ema_summary: dict[str, float] = {}
    for e in emas:
        ckf = exp_dir / f"best_ema{e['decay']}.pt"
        if not ckf.exists():
            print(f"EMA(decay={e['decay']}): no checkpoint at {ckf.name} (nothing evaluated)", flush=True)
            continue
        try:
            ema_model = WindowMIL(m["arch"], pretrained=False, drop=m.get("head_dropout", 0.2),
                                  grad_checkpointing=False, img_size=m.get("img_size"),
                                  backbone_kwargs=m.get("backbone_kwargs")).to(device)
            ema_model.load_state_dict(torch.load(ckf, map_location=device)["model"])
            e_val, _ = macro_auc(predict(ema_model, val_loader, m["res"], device, dtype), val_truth)
            e_gold_df = predict(ema_model, gold_loader, m["res"], device, dtype)
            e_gold, _ = macro_auc(e_gold_df, gold_truth)
            e_gold_df.to_parquet(exp_dir / f"gold58_predictions_ema{e['decay']}.parquet", index=False)
            ema_summary[f"ema{e['decay']}_val_macro_auc"] = e_val
            ema_summary[f"ema{e['decay']}_gold58_macro_auc"] = e_gold
            ema_summary[f"ema{e['decay']}_epoch"] = e["best"]["epoch"]
            print(f"EMA(decay={e['decay']}, best epoch {e['best']['epoch']}): fold-val {e_val:.4f} | "
                  f"gold-58 {e_gold:.4f}   [top-{len(topk)} avg: fold-val {val_macro:.4f} | "
                  f"gold-58 {gold_macro:.4f}]", flush=True)
            del ema_model
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"EMA(decay={e['decay']}) evaluation FAILED ({type(exc).__name__}: {exc}); the top-k "
                  f"result above is unaffected", file=sys.stderr, flush=True)
    (exp_dir / "history.json").write_text(json.dumps({"hist": hist, "topk": topk}, indent=1))
    pd.DataFrame([{
        "experiment": cfg["experiment"], "fold": args.fold, "epochs_completed": len(hist),
        "val_macro_auc": val_macro, "gold58_macro_auc": gold_macro, **ema_summary,
        **{f"val_{kk}": v for kk, v in val_pl.items()}, **{f"gold_{kk}": v for kk, v in gold_pl.items()},
    }]).to_csv(exp_dir / "summary.csv", index=False)
    print(f"done -> {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
