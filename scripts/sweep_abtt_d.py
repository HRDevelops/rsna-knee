# Copyright 2026 Husnain Rasool
# Licensed under the Apache License, Version 2.0 (see repository LICENSE).
"""exp016 target design: sweep the number of removed top principal components D (2026-09-20).

Selection rule fixed BEFORE running: pick the smallest D whose crude-language R^2 (share of embedding
variance explained by language-group means) is <= 0.05, i.e. about a third of the raw 0.15; ties broken by
higher report-embedding -> label probe macro AUC. Uses only report text and report-derived labels — no image
model result is involved, so this cannot be tuned to the outcome of the experiment it feeds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_acl_errors import guess_language  # noqa: E402
from build_report_embeddings import TARGETS, all_but_the_top, apply_abtt, group_r2  # noqa: E402

root = Path(os.environ["RSNA_ROOT"])
out = root / "artifacts" / "report_emb"
raw = np.load(out / "raw_bgem3.npy")
uids = (out / "uids.txt").read_text().split()
train = pd.read_csv(root / "data/raw/train.csv", usecols=["StudyInstanceUID", "Report"]).set_index("StudyInstanceUID").loc[uids]
lang = np.array([guess_language(t) for t in train["Report"].fillna("")])
folds = pd.read_parquet(root / "artifacts/folds.parquet").set_index("study_uid")["fold"].loc[uids].to_numpy()
lab = pd.read_parquet(root / "artifacts/labels/labels_v2_final.parquet").set_index("study_uid")
labels = np.stack([(lab.loc[uids, f"{t}__probability"].to_numpy() >= 0.5).astype(int) for t in TARGETS], axis=1)

print("D | fold | language R2 | probe macro AUC (val fold)")
for k in (0, 3):
    tr, va = (folds != k) & (folds != -1), folds == k
    for d in (0, 2, 3, 5, 7, 10):
        if d == 0:
            xtr, xva = raw[tr] - raw[tr].mean(0), raw[va] - raw[tr].mean(0)
            xtr /= np.linalg.norm(xtr, axis=1, keepdims=True); xva /= np.linalg.norm(xva, axis=1, keepdims=True)
        else:
            mu, top = all_but_the_top(raw[tr], d)
            xtr, xva = apply_abtt(raw[tr], mu, top), apply_abtt(raw[va], mu, top)
        aucs = []
        for i in range(len(TARGETS)):
            if labels[tr, i].min() == labels[tr, i].max() or labels[va, i].min() == labels[va, i].max():
                continue
            aucs.append(roc_auc_score(labels[va, i], LogisticRegression(C=1.0, max_iter=300).fit(xtr, labels[tr, i]).predict_proba(xva)[:, 1]))
        print(f"{d:>2} | {k} | {group_r2(xtr, lang[tr]):.3f} | {np.mean(aucs):.3f}", flush=True)
