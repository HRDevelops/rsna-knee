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

"""Phase 0C / Phase 1 trainer for the token transformer (rsna_knee.tokens.TokenMIL) on cache v2.

Same budget, schedule, loss and selection as exp018 (16 epochs, 8 studies x 12 windows per step, OneCycle, pos_weight
BCE, top-3 epochs by FOLD-VALIDATION macro-AUC averaged; gold-58 never used for selection). Per epoch, fold-val is
scored in BAGGED mode (the pre-registered default). At the end the top-3-averaged model is scored in both modes:
bagged (R = 8 passes of 12 windows) and dense (all windows, one pass), each with its own OOF parquet, so the two modes
can be compared with a paired cluster bootstrap on identical weights. Dense is skipped and logged as INFEASIBLE if it
runs out of memory or exceeds `dense_max_sec_per_study` (pre-registered feasibility exclusion).

    python scripts/train_tokens.py --config configs/p0c_g6_s42.yaml --fold 0 --out $RSNA_ROOT/artifacts [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsna_knee.tokens import TokenMIL  # noqa: E402
from rsna_knee.wmil import TARGET_COLUMNS, average_state_dicts  # noqa: E402
from rsna_knee.wmil_v2 import TokenV2Dataset, collate_tokens, load_manifest_v2  # noqa: E402
from train import expand_env, load_gold_truth, load_soft_truth, macro_auc, set_seed  # noqa: E402


def to_dev(b, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def run_model(model, b, res, dtype, gain=None):
    """Logits per pass: returns [B, P, 12] where P = number of passes in the item (1 for train/dense, R for bagged)."""
    B = b["windows"].shape[0]
    P = int(b["pass_id"].max().item()) + 1
    K = b["windows"].shape[1] // P
    with torch.autocast(device_type="cuda", dtype=dtype):
        feats = model.encode(b["windows"], res, gain)                     # [B, P*K*g*g, C]
        gg = model.grid * model.grid
        sh = lambda t: t.reshape(B * P, K, *t.shape[2:])                  # noqa: E731
        logits = model(feats.reshape(B * P, K * gg, -1), sh(b["slot"]), sh(b["plane"]), sh(b["pos"]),
                       sh(b["meta"]), sh(b["valid"]), mu=sh(b["mu"]), sigma=sh(b["sigma"]))
    return logits.float().view(B, P, -1)


@torch.no_grad()
def predict(model, loader, res, device, dtype, max_sec=None):
    model.eval()
    rows, t0 = [], time.time()
    for n, b in enumerate(loader):
        b = to_dev(b, device)
        p = torch.sigmoid(run_model(model, b, res, dtype)).mean(1).cpu().numpy()
        for uid, v in zip(b["uid"], p):
            rows.append({"study_uid": uid, **{c: float(x) for c, x in zip(TARGET_COLUMNS, v)}})
        if max_sec is not None and n >= 4 and (time.time() - t0) / len(rows) > max_sec:
            raise TimeoutError(f"{(time.time() - t0) / len(rows):.2f} s/study > {max_sec}")
    return pd.DataFrame(rows)


def donor_map(target_u, pool_u, man, final_side):
    """PHASE1_PREREG s3: donor = a training-fold study with the same set of planes and the same side, nearest in
    total slice count (ties: lowest UID)."""
    g = man.groupby("study_uid")
    info = {u: (frozenset(x.plane), str(final_side.get(u)), int(x.n_slices.sum())) for u, x in g}
    pool = sorted(u for u in pool_u if u in info)
    out = {}
    for u in target_u:
        if u not in info:
            continue
        pl, sd, n = info[u]
        cands = [v for v in pool if info[v][0] == pl and info[v][1] == sd and v != u]
        if cands:
            out[u] = min(cands, key=lambda v: (abs(info[v][2] - n), v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-hours", type=float, default=5.5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--beta-grad-probe", type=int, default=0,
                    help="PHASE1_PREREG s2: run N training steps (no evaluation), log |dL/dbeta|, print eta_beta, exit")
    a = ap.parse_args()
    cfg = expand_env(yaml.safe_load(a.config.read_text()))
    set_seed(cfg["seed"])
    device, dtype = "cuda", torch.bfloat16
    root = Path(os.environ["RSNA_ROOT"])
    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    exp_dir = a.out / cfg["experiment"] / f"fold{a.fold}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    if not a.smoke:
        shutil.copy(a.config, exp_dir / "config.yaml")

    man = load_manifest_v2(Path(d["manifest_dir"]))
    labels = pd.read_parquet(d["labels"]).set_index("study_uid")
    folds = pd.read_parquet(d["folds"])[["study_uid", "fold"]]
    side_df = pd.read_csv(d["final_side"])
    final_side = side_df.set_index("study_uid")["final_side"].to_dict()
    bilateral = set(side_df.loc[side_df["source"].isin(["conflict_resolved_by_report", "conflict_unresolved"]),
                                "study_uid"])
    usable = set(man["study_uid"]) & set(labels.index)
    train_u = sorted(usable & set(folds.loc[(folds.fold != a.fold) & (folds.fold != -1), "study_uid"]))
    val_u = sorted(usable & set(folds.loc[folds.fold == a.fold, "study_uid"]))
    gold_u = sorted(usable & set(folds.loc[folds.fold == -1, "study_uid"]))
    assert not (set(train_u) & (set(val_u) | set(gold_u)))
    epochs, bs, k = t["epochs"], t["batch_studies"], t["k_train"]
    if a.smoke:
        train_u, val_u, gold_u, epochs, bs = train_u[:24], val_u[:8], gold_u[:6], 2, 4
    g = m["grid"]
    print(f"fold {a.fold}: train {len(train_u)} | val {len(val_u)} | gold {len(gold_u)} | grid {g} | {d['v2']}",
          flush=True)

    geom = m.get("geometry", "none")

    def mk(u, mode, lab=True, donor=None):
        return TokenV2Dataset(u, d["cache_dir"], man, labels if lab else None, final_side, bilateral,
                              mode == "train", k, grid=g, mode=mode, n_bags=m.get("n_bags", 8),
                              geometry=geom != "none", donor=donor, **d["v2"])
    train_loader = DataLoader(mk(train_u, "train"), batch_size=bs, shuffle=True, num_workers=a.workers,
                              collate_fn=collate_tokens, drop_last=True, persistent_workers=a.workers > 0,
                              pin_memory=True)
    bag = lambda u: DataLoader(mk(u, "bagged"), batch_size=2, num_workers=4, collate_fn=collate_tokens)  # noqa: E731
    dense = lambda u: DataLoader(mk(u, "dense"), batch_size=1, num_workers=4, collate_fn=collate_tokens)  # noqa: E731

    prev = np.clip(np.stack([[labels.loc[u, f"{c}__probability"] for c in TARGET_COLUMNS] for u in train_u]).mean(0),
                   0.03, 0.7)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32,
                                                         device=device))
    model = TokenMIL(m["arch"], grid=g, d=m.get("d", 384), heads=m.get("heads", 8), layers=m.get("layers", 2),
                     drop=m.get("head_dropout", 0.2), grad_checkpointing=m.get("grad_checkpointing", True),
                     geometry=geom, features=m.get("features", "final")).to(device)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    head = [p for n, p in model.named_parameters()
            if not n.startswith("backbone.") and n != "beta" and p.requires_grad]
    # beta: projected SGD without momentum or weight decay at a constant, pre-registered eta (PHASE1_PREREG s2)
    beta_opt = torch.optim.SGD([model.beta], lr=float(m["eta_beta"]), momentum=0.0)         if model.beta is not None and not a.beta_grad_probe else None
    opt = torch.optim.AdamW([{"params": bb, "lr": t["bb_lr"]}, {"params": head, "lr": t["head_lr"]}],
                            weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[t["bb_lr"], t["head_lr"]],
                                                total_steps=max(len(train_loader) * epochs, 1), pct_start=t["pct_start"])
    val_truth, gold_truth = load_soft_truth(Path(d["labels"])), load_gold_truth(root / "data/raw/train.csv")
    ck_path, topk_path = exp_dir / "checkpoint.pt", exp_dir / "topk.json"
    hist, topk, start = [], [], 0
    if ck_path.exists() and not a.smoke:
        ck = torch.load(ck_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start, hist = ck["epoch"] + 1, ck["hist"]
        topk = json.loads(topk_path.read_text()) if topk_path.exists() else []
        print(f"resumed at epoch {start}", flush=True)
    t_start = time.time()
    probe = []
    for ep in range(start, epochs):
        model.train()
        tot, nb, t0 = 0.0, 0, time.time()
        for b in train_loader:
            b = to_dev(b, device)
            gain = 1.0 + (torch.rand(b["windows"].shape[0], device=device) * 2 - 1) * cfg["aug"]["intensity_jitter"]
            logits = run_model(model, b, m["res"], dtype, gain)[:, 0]
            loss = lossf(logits, b["labels"])
            opt.zero_grad(set_to_none=True)
            if model.beta is not None:
                model.beta.grad = None
            if not torch.isfinite(loss):
                print("non-finite loss; skipped", flush=True)
                sched.step()
                continue
            loss.backward()
            if a.beta_grad_probe:
                probe.append(model.beta.grad.detach().abs().flatten().cpu().numpy())
                model.beta.grad = None                           # probe: beta stays exactly 0
                if len(probe) >= a.beta_grad_probe:
                    med = float(np.median(np.concatenate(probe)))
                    eta = 0.5 / (len(train_loader) * med) if med > 0 else float("nan")
                    print(f"BETA PROBE: {len(probe)} steps, steps/epoch {len(train_loader)}, median |dL/dbeta| "
                          f"{med:.3e}, per layer/head medians {np.median(np.stack(probe), 0).round(6).tolist()} "
                          f"-> eta_beta = 0.5 / (steps_per_epoch x median) = {eta:.4g}", flush=True)
                    return
            torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if n != "beta"], t["grad_clip"])
            opt.step()
            sched.step()
            if beta_opt is not None:
                beta_opt.step()
                with torch.no_grad():
                    model.beta.clamp_(min=0.0)                    # projection onto beta >= 0
            tot, nb = tot + loss.item(), nb + 1
        val_auc, _ = macro_auc(predict(model, bag(val_u), m["res"], device, dtype), val_truth)
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "val_auc_bagged": val_auc, "sec": time.time() - t0,
                     "beta": None if model.beta is None else model.beta.detach().cpu().tolist()})
        print(f"epoch {ep}: loss {tot / max(nb, 1):.4f} | fold-val macro-AUC (bagged) {val_auc:.4f} | "
              f"{time.time() - t0:.0f}s", flush=True)
        if not a.smoke:
            f = f"top_ep{ep:02d}.pt"
            topk.append({"epoch": ep, "val_auc": val_auc, "file": f})
            topk = sorted(topk, key=lambda e: -e["val_auc"])
            torch.save({"model": model.state_dict()}, exp_dir / f)
            for e in topk[t["topk"]:]:
                (exp_dir / e["file"]).unlink(missing_ok=True)
            topk = topk[:t["topk"]]
            topk_path.write_text(json.dumps(topk))
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "epoch": ep, "hist": hist}, ck_path)
        if (time.time() - t_start) / 3600 > a.max_hours and ep < epochs - 1:
            print("wall-clock guard: stopping; resubmit to resume", flush=True)
            return
    if not a.smoke:
        model.load_state_dict(average_state_dicts([torch.load(exp_dir / e["file"], map_location="cpu")["model"]
                                                   for e in topk]))
        torch.save({"model": model.state_dict(), "topk": topk}, exp_dir / "best.pt")
    summary = {"experiment": cfg["experiment"], "fold": a.fold, "grid": g, "epochs_completed": len(hist)}
    evals = [("bagged", bag), ("dense", dense)]
    if geom != "none":                                   # donor-swap control (PHASE1_PREREG s3), dense, inference only
        donor = donor_map(val_u + gold_u, train_u, man, final_side)
        evals.append(("dense_donor", lambda u: DataLoader(mk(u, "dense", donor=donor), batch_size=1, num_workers=4,
                                                          collate_fn=collate_tokens)))
        summary["beta"] = json.dumps(model.beta.detach().cpu().tolist())
    for mode, mkl in evals:
        try:
            torch.cuda.reset_peak_memory_stats()
            vdf = predict(model, mkl(val_u), m["res"], device, dtype,
                          max_sec=m.get("dense_max_sec_per_study", 3.0) if mode.startswith("dense") else None)
            gdf = predict(model, mkl(gold_u), m["res"], device, dtype)
            vauc, vper = macro_auc(vdf, val_truth)
            gauc, _ = macro_auc(gdf, gold_truth)
            vdf.to_parquet(exp_dir / f"oof_{mode}.parquet", index=False)
            gdf.to_parquet(exp_dir / f"gold58_{mode}.parquet", index=False)
            summary.update({f"val_macro_auc_{mode}": vauc, f"gold58_macro_auc_{mode}": gauc,
                            f"peak_gb_{mode}": torch.cuda.max_memory_allocated() / 1e9,
                            **{f"val_{mode}_{c}": v for c, v in vper.items()}})
            print(f"FINAL {mode}: fold-val {vauc:.4f} | gold-58 {gauc:.4f} (not used for selection)", flush=True)
        except (torch.cuda.OutOfMemoryError, TimeoutError) as e:
            torch.cuda.empty_cache()
            summary[f"val_macro_auc_{mode}"] = float("nan")
            summary[f"{mode}_infeasible"] = f"{type(e).__name__}: {e}"[:200]
            print(f"FINAL {mode}: INFEASIBLE ({type(e).__name__}: {e})", flush=True)
    pd.DataFrame([summary]).to_csv(exp_dir / "summary.csv", index=False)
    (exp_dir / "history.json").write_text(json.dumps(hist))
    print(f"done -> {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
