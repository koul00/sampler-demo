"""
run_decile.py

Evaluates cohort27_inceptionv3.npz (from extract_features.py): leave-one-out
logistic regression on the 61,440-dim SAMPLER-faithful representations
(InceptionV3, 3 scales, our 27-slide CAMELYON16 cohort). Same evaluation as
../plip/run_decile.py, so the two are directly comparable.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

d = np.load("cohort27_inceptionv3.npz", allow_pickle=True)
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
