"""Reviewer checks from saved data, no training of new models beyond LANTERN's own heads.
projection: how much of the between-class signal the between-site projection removes, on development data.
gsources: glaucoma recall over the 50 rotations against the number of development sources that contain glaucoma.
ece_ci: per-source ECE after TS on the published split with bootstrap intervals over leakage groups.
python -m experiments.review_checks --bench data/bench --strata data/strata --runs runs --out results/review
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .calib import Calibrator, ece
from .data import load_table
from .strata_gate4 import nap

G_SOURCES = {"bangladesh_eye", "grape", "afio_composite", "fives", "hrf", "hygd", "papila"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "strata", "runs", "out"]:
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, classes = load_table(a.bench)
    sp, y, src, grp = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.group_id.to_numpy()
    M = np.load(Path(a.strata) / "mean3_medsiglip.npy")[:, 2].astype(np.float32)
    tr = sp == "train"
    X = (M - M[tr].mean(0)) / (M[tr].std(0) + 1e-6)

    # projection: share of each class-mean difference, and of site identity, that lies in the removed subspace
    rows = []
    for o in [None] + sorted(set(src[tr])):
        m = tr & (src != o) if o else tr
        if len(np.unique(y[m])) < 3:
            continue
        U = nap(X[m], y[m], src[m], 32)
        mu = [X[m & (y == c)].mean(0) for c in range(3)]
        r = dict(left_out=o or "none", k=U.shape[0])
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            d = mu[j] - mu[i]
            r[f"{classes[i]}{classes[j]}_removed"] = float(np.linalg.norm(U @ d) / np.linalg.norm(d))
        # random subspace of the same size, as a reference
        rng = np.random.default_rng(0)
        Q = np.linalg.qr(rng.standard_normal((X.shape[1], U.shape[0])))[0].T
        for i, j in [(0, 2)]:
            d = mu[j] - mu[i]
            r["NG_removed_random"] = float(np.linalg.norm(Q @ d) / np.linalg.norm(d))
        # glaucoma lives in two sources: how aligned is the G site direction with the N-G direction?
        g = [X[m & (y == 2) & (src == s)].mean(0) for s in sorted(set(src[m & (y == 2)]))]
        if len(g) == 2:
            dg, dng = g[1] - g[0], mu[2] - mu[0]
            r["cos_Gsite_NG"] = float(abs(dg @ dng) / np.linalg.norm(dg) / np.linalg.norm(dng))
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "projection.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())

    # glaucoma recall against glaucoma diversity on the development side
    lr = pd.read_csv("results/lantern_rotation.csv")
    rot = pd.read_csv("results/supplement/rotation.csv")
    rows = []
    for name, d in [("LANTERN", lr), ("MedSigLIP LP", rot[(rot.model == "medsiglip") & (rot.kind == "full_class")])]:
        d = d[d.split > 0].copy()
        d["g_dev"] = d.test_sources.map(lambda s: len(G_SOURCES - set(s.split("|"))))
        for k, q in d.groupby("g_dev"):
            rows.append(dict(method=name, g_dev_sources=k, rotations=len(q), recall_G=q.recall_G.mean(), test_bacc=q.test_bacc.mean()))
    g = pd.DataFrame(rows)
    g.to_csv(out / "glaucoma_sources.csv", index=False)
    print(g.round(3).to_string())

    # per-source ECE after TS with bootstrap over leakage groups, LANTERN seeds pooled
    ca, te = (sp == "calibration") & (t.cal_subset.to_numpy() == "cal_A"), sp == "test"
    P = []
    for s in range(3):
        Z = np.load(Path(a.runs) / "top_dev" / "last_reg" / f"logits_s{s}.npy").astype(np.float64)
        P.append(Calibrator("ts").fit(Z[ca], y[ca])(Z))
    rng = np.random.default_rng(0)
    rows = []
    for s in ["paraguay_dr", "hygd", "fives", "papila", "palm", "hrf", "afio_composite", "chaksu"]:
        m = np.flatnonzero((src == s) & (te | (sp == "test_ood")))
        ug, inv = np.unique(grp[m], return_inverse=True)
        point = np.mean([ece(p[m], y[m])[0] for p in P])
        bs = []
        for _ in range(1000):
            w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv]
            idx = np.repeat(m, w)
            bs.append(np.mean([ece(p[idx], y[idx])[0] for p in P]))
        rows.append(dict(source=s, n=len(m), groups=len(ug), ece_ts=point, lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
    e = pd.DataFrame(rows)
    e.to_csv(out / "ece_ci.csv", index=False)
    print(e.round(3).to_string())


if __name__ == "__main__":
    main()
