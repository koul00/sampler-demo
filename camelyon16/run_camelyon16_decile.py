"""
run_camelyon16_decile.py

Cross-check of SAMPLER's core trick — per-feature decile aggregation across a
slide's tiles — on a different dataset and a different feature backbone than
the TCGA-BRCA / InceptionV3 demo in this repo.

Features here come from a LazySlide pipeline (PLIP, 512-dim) on a 10%
stratified sample of the CAMELYON16 training set (16 normal, 11 tumor):
https://github.com/koul00/lazyslide-wsi-pipeline

sampler_cohort27.npz holds, per slide, np.ravel(np.percentile(X, deciles, axis=0))
— SAMPLER.SAMPLERrep's one-liner — applied to the slide's PLIP tile features
instead of InceptionV3. X: (27, 5120), y: (27,) tumor=1/normal=0, names: (27,).

Evaluated with leave-one-out (N=27 is too small for a held-out split).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

d = np.load("sampler_cohort27.npz", allow_pickle=True)
X, y, names = d["X"], d["y"], d["names"]
print(f"X: {X.shape}  y counts (normal/tumor): {np.bincount(y)}")

loo = LeaveOneOut()
probs = np.zeros(len(y))
for train_idx, test_idx in loo.split(X):
    scaler = StandardScaler().fit(X[train_idx])
    Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
    clf = LogisticRegression(C=0.01, max_iter=5000)
    clf.fit(Xtr, y[train_idx])
    probs[test_idx] = clf.predict_proba(Xte)[:, 1]

auc = roc_auc_score(y, probs)
print(f"\nLeave-one-out AUC: {auc:.3f}")
for n, p, t in zip(names, probs, y):
    print(f"  {n:12s} true={t} score={p:.3f}")
