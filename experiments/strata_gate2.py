"""STRATA gate 2: does the disc-anchored spectrum add to the pooled feature? Leave-one-development-source-out; reads no test row.
Arms, all with the conditional-logit loss unless named: global (pooled feature), polar (+ all modes), polar_m0 (+ ring means only: the ablation of the angular part), ce_polar (polar with the usual loss alone).
python -m experiments.strata_gate2 --bench data/bench --features runs/features --spectrum data/strata/spectrum_medsiglip.npy --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from .data import balanced_weights, load_table
from .probe_calib import train_head
from .shoal import load_x, metrics
from .strata import train_strata


def project(S, fit, dev, dim, seed):
    """PCA fitted on the fold's training rows only; scaled to unit variance."""
    F = np.nan_to_num(S[dev].reshape(dev.sum(), -1).astype(np.float32))
    p = PCA(dim, svd_solver="randomized", random_state=seed).fit(F[fit[dev]])
    Z = p.transform(F).astype(np.float32)
    return Z / (Z[fit[dev]].std(0) + 1e-6)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--spectrum", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--dim", type=int, default=128)
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    X = load_x(a.features, "medsiglip", t)
    S = np.load(a.spectrum, mmap_mode="r")
    dev = np.isin(sp, ["train", "validation"])
    rows = []
    for s in sorted(set(src[sp == "train"])):
        tr, va, held = (sp == "train") & (src != s), (sp == "validation") & (src != s), dev & (src == s)
        pos = np.full(len(t), -1)
        pos[np.flatnonzero(dev)] = np.arange(dev.sum())
        views = {"polar": project(S, tr, dev, a.dim, 0), "polar_m0": project(S[:, :, :1], tr, dev, a.dim, 0)}
        w = balanced_weights(y[tr].tolist())
        for seed in a.seeds:
            def concat(view, m):
                return np.concatenate([X[m], views[view][pos[m]]], 1)
            heads = {"global": (train_strata(X[tr], y[tr], src[tr], w, X[va], y[va], a.lam, seed), X[held]),
                     "ce_polar": (train_head(concat("polar", tr), y[tr], w, concat("polar", va), y[va], "ls", seed), concat("polar", held))}
            for v in views:
                heads[v] = (train_strata(concat(v, tr), y[tr], src[tr], w, concat(v, va), y[va], a.lam, seed), concat(v, held))
            for arm, (h, Xh) in heads.items():
                with torch.no_grad():
                    Z = h(torch.tensor(Xh)).numpy()
                pred, yy = Z.argmax(1), y[held]
                rec = {f"recall_{c}": (pred[yy == k] == k).mean() if (yy == k).any() else np.nan for k, c in enumerate(classes)}
                rows.append(dict(seed=seed, source=s, arm=arm, n=int(held.sum()), **metrics(Z, yy), **rec))
            print(s, seed, flush=True)
    d = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    d.to_csv(Path(a.out) / "gate2.csv", index=False)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_N", "recall_D", "recall_G"]].mean().round(3).to_string())
    print(d.groupby(["arm", "seed"]).bacc.mean().unstack().round(3).to_string())
    print(d.pivot_table(index="source", columns="arm", values="bacc").round(3).to_string())
    print(d.pivot_table(index="source", columns="arm", values="recall_G").round(3).to_string())


if __name__ == "__main__":
    main()
