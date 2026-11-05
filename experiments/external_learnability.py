"""Is each external cohort's label learnable from frozen MedSigLIP features at all? 5-fold cross-validated logistic regression
within the cohort (diagnostic only: no model of the paper is changed). Messidor-2 is the control.
python -m experiments.external_learnability --root ~/fundusshift
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="~/fundusshift")
a = ap.parse_args()
r = Path(a.root).expanduser()
e = pd.read_csv(r / "data/external/external.csv")
X = np.load(r / "data/external/features/mean3_last.npy").astype(np.float32)
out = {}
for s, pos in [("messidor2", 1), ("g1020", 2)]:
    m = (e.source == s).to_numpy()
    Xs, ys = X[m], (e.y.to_numpy()[m] == pos).astype(int)
    for C in (0.01, 0.1, 1.0):
        au = []
        for seed in range(5):
            p = np.zeros(len(ys))
            for trn, tst in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
                p[tst] = clf.fit(Xs[trn], ys[trn]).predict_proba(Xs[tst])[:, 1]
            au.append(roc_auc_score(ys, p))
        out[f"{s}_C{C}"] = dict(auroc_mean=float(np.mean(au)), auroc_sd=float(np.std(au)), n=int(len(ys)), pos=int(ys.sum()))
        print(s, C, round(np.mean(au), 3), round(np.std(au), 3), flush=True)
(r / "results/external").mkdir(parents=True, exist_ok=True)
json.dump(out, open(r / "results/external/learnability.json", "w"), indent=1)
