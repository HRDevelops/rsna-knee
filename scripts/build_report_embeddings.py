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

"""exp016 auxiliary targets: frozen-multilingual-encoder embeddings of every study's report.

Text encoder: BAAI/bge-m3 (MIT license — verified on the model card 2026-09-20; XLM-R-large based,
100+ languages, 8192-token context, dense embedding = L2-normalized [CLS]). Chosen over "translate to
English with the LLM stack first" because it needs no 4,407-report translation pass (hours of vLLM,
plus translation noise/hallucination on medical text) and, being cross-lingually aligned, puts a Dutch
and a Spanish report about the same finding near each other — which is what the image side can learn.
The encoder is used ONLY at training time to make targets; reports do not exist at test time and the
deployed image model contains no text component.

Two steps:
  1. embed:   every report -> artifacts/report_emb/raw_bgem3.npy (+ uids.txt, meta.json). Frozen
              external encoder, so this step cannot leak anything about validation labels.
  2. targets: per fold k, artifacts/report_emb/fold{k}_target.npz with `uids` and `target` for the
              fold's TRAIN studies only (fold not in {k, -1}) — validation-fold and gold-58 reports
              are never written to any file training can read. Target = "all-but-the-top" (Mu &
              Viswanath 2018) post-processing fit on THAT fold's train embeddings only: subtract the
              mean, remove the top-D principal components, L2-normalize. Rationale: the dominant
              directions of a multilingual sentence-embedding matrix encode language and site/template
              style; images carry scanner/site cues, so a regression to the raw embedding could be
              minimized by recognizing the site rather than the pathology.

--diagnostics prints (a) how much of the embedding variance crude language groups explain before vs.
after the removal, and (b) a report-embedding -> label linear probe (train fold -> val fold AUC vs
labels_v2_final), raw vs. residualized — the val-fold reports are used ONLY inside this printout.

    python scripts/build_report_embeddings.py --device cuda --diagnostics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_NAME = "BAAI/bge-m3"
# The official repo ships only a pickled pytorch_model.bin, which transformers refuses to torch.load on
# torch < 2.6 (CVE-2025-32434; this venv has 2.4). The HF safetensors-conversion bot's PR commit below
# holds the same weights as model.safetensors — a format that cannot execute code — and is pinned by
# commit hash so the run is reproducible. Provenance is third-party-converted, so embed() runs a
# functional self-test (a translation pair must sit closer than an unrelated sentence) and the sha is
# recorded in meta.json.
SAFETENSORS_REVISION = "9a0624b896d81da7492a910ffa53731274b6cf3d"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def selftest(tok, model, device: str) -> None:
    """A randomly-initialized or mis-loaded encoder would fail this: an English sentence and its Spanish
    translation must be clearly closer than an unrelated finding."""
    import torch
    import torch.nn.functional as F
    sents = ["Complete tear of the anterior cruciate ligament.",
             "Rotura completa del ligamento cruzado anterior.",
             "Small joint effusion without fracture or bone marrow edema."]
    with torch.inference_mode():
        enc = tok(sents, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e = F.normalize(model(**enc).last_hidden_state[:, 0].float(), dim=-1)
    same, diff = float(e[0] @ e[1]), float(e[0] @ e[2])
    print(f"selftest: cos(EN, ES translation)={same:.3f}  cos(EN, unrelated)={diff:.3f}", file=sys.stderr)
    assert same > diff + 0.1, "encoder self-test failed — weights did not load correctly"


def embed_reports(texts: list, device: str, max_len: int, batch_size: int) -> tuple[np.ndarray, int]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from huggingface_hub import constants
    from safetensors.torch import load_file
    from transformers import AutoConfig

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # from_pretrained(revision=<PR commit sha>) is not resolvable (a conversion-PR commit is not reachable
    # by hash like a branch), but transformers' own auto-conversion already put the file in the HF cache.
    # Build the model from the OFFICIAL config and load those tensors directly, requiring zero missing keys.
    st_path = (Path(constants.HF_HUB_CACHE) / "models--BAAI--bge-m3" / "snapshots" / SAFETENSORS_REVISION
               / "model.safetensors")
    assert st_path.exists(), f"{st_path} not found in the HF cache"
    model = AutoModel.from_config(AutoConfig.from_pretrained(MODEL_NAME))
    missing, unexpected = model.load_state_dict(load_file(str(st_path)), strict=False)
    print(f"safetensors load: {len(missing)} missing, {len(unexpected)} unexpected keys {unexpected[:3]}", file=sys.stderr)
    assert not missing, f"missing keys: {missing[:5]}"
    model = model.to(torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    selftest(tok, model, device)
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    n_truncated = 0
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            batch = [texts[i] for i in idx]
            n_truncated += sum(len(tok(t, add_special_tokens=True)["input_ids"]) > max_len for t in batch)
            enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            cls = model(**enc).last_hidden_state[:, 0]
            out[idx] = F.normalize(cls.float(), dim=-1).cpu().numpy()
            if (start // batch_size) % 50 == 0:
                print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}", file=sys.stderr)
    return out, n_truncated


def all_but_the_top(train_x: np.ndarray, d: int):
    mu = train_x.mean(axis=0, keepdims=True)
    centered = train_x - mu
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top = vt[:d]  # [d, dim]
    return mu, top


def apply_abtt(x: np.ndarray, mu: np.ndarray, top: np.ndarray) -> np.ndarray:
    c = x - mu
    c = c - (c @ top.T) @ top
    return c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-8, None)


def group_r2(x: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of total variance explained by group means (between-group / total)."""
    total = ((x - x.mean(0)) ** 2).sum()
    between = 0.0
    for g in np.unique(groups):
        m = groups == g
        between += m.sum() * ((x[m].mean(0) - x.mean(0)) ** 2).sum()
    return float(between / total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--abtt-d", type=int, default=5, help="number of top principal components removed")
    ap.add_argument("--diagnostics", action="store_true")
    args = ap.parse_args()

    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    out_dir = root / "artifacts" / "report_emb"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(root / "data/raw/train.csv", usecols=["StudyInstanceUID", "Report"])
    train = train.rename(columns={"StudyInstanceUID": "study_uid"})
    uids = train["study_uid"].tolist()

    raw_path = out_dir / "raw_bgem3.npy"
    if raw_path.exists() and (out_dir / "uids.txt").read_text().split() == uids:
        raw = np.load(raw_path)
        print(f"reusing {raw_path} {raw.shape}")
    else:
        raw, n_trunc = embed_reports(train["Report"].fillna("").tolist(), args.device, args.max_len, args.batch_size)
        np.save(raw_path, raw)
        (out_dir / "uids.txt").write_text("\n".join(uids))
        (out_dir / "meta.json").write_text(json.dumps({
            "model": MODEL_NAME, "revision": SAFETENSORS_REVISION, "license": "MIT", "pooling": "CLS + L2 normalize", "max_len": args.max_len,
            "n_reports": len(uids), "n_truncated_at_max_len": int(n_trunc)}, indent=1))
        print(f"embedded {raw.shape}, {n_trunc} reports truncated at {args.max_len} tokens")
    assert np.isfinite(raw).all()

    folds = pd.read_parquet(root / "artifacts/folds.parquet").set_index("study_uid")["fold"].loc[uids].to_numpy()
    uid_arr = np.array(uids)

    lang = labels = None
    if args.diagnostics:
        from analyze_acl_errors import guess_language
        lang = np.array([guess_language(t) for t in train["Report"].fillna("")])
        lab = pd.read_parquet(root / "artifacts/labels/labels_v2_final.parquet").set_index("study_uid")
        labels = np.stack([(lab.loc[uids, f"{t}__probability"].to_numpy() >= 0.5).astype(int) for t in TARGETS], axis=1)

    for k in range(5):
        tr = (folds != k) & (folds != -1)
        mu, top = all_but_the_top(raw[tr], args.abtt_d)
        target = apply_abtt(raw[tr], mu, top).astype(np.float32)
        np.savez(out_dir / f"fold{k}_target.npz", uids=uid_arr[tr], target=target)
        line = f"fold{k}: {int(tr.sum())} train targets {target.shape}"
        if args.diagnostics:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            va = folds == k
            r2_raw, r2_res = group_r2(raw[tr], lang[tr]), group_r2(target, lang[tr])
            aucs = {}
            for name, xtr, xva in (("raw", raw[tr], raw[va]),
                                   ("resid", target, apply_abtt(raw[va], mu, top))):
                per = []
                for i in range(len(TARGETS)):
                    if labels[tr, i].min() == labels[tr, i].max() or labels[va, i].min() == labels[va, i].max():
                        continue
                    clf = LogisticRegression(C=1.0, max_iter=300).fit(xtr, labels[tr, i])
                    per.append(roc_auc_score(labels[va, i], clf.predict_proba(xva)[:, 1]))
                aucs[name] = float(np.mean(per))
            line += (f" | language R2 raw {r2_raw:.3f} -> resid {r2_res:.3f} | "
                     f"report-embedding->label probe macro AUC (val fold) raw {aucs['raw']:.3f} resid {aucs['resid']:.3f}")
        print(line)
    (out_dir / "targets_meta.json").write_text(json.dumps({
        "abtt_d": args.abtt_d, "fit": "per fold, on that fold's train-split embeddings only",
        "d_selection": "smallest D with crude-language R2 <= 0.05 (scripts/sweep_abtt_d.py, chosen before any image result)"}, indent=1))
    print(f"wrote {out_dir}/fold[0-4]_target.npz (train-split studies only), D={args.abtt_d}")


if __name__ == "__main__":
    main()
