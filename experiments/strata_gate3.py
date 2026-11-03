"""STRATA gate 3 on the conditional-logit head: shrunken site prior (kappa), site temperature from feature distance (beta), and the polar spectrum on the G logit only.
Leave-one-development-source-out; reads no test row.
python -m experiments.strata_gate3 --bench data/bench --features runs/features --spectrum data/strata/spectrum_medsiglip.npy --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import balanced_weights, load_table
from .shoal import K, load_x, metrics
from .strata import train_strata
from .strata_gate2 import project


class GOnly(torch.nn.Module):
    """Pooled feature on every class; the extra block reaches the G logit only."""

    def __init__(self, d, extra):
        super().__init__()
        self.base, self.g, self.d = torch.nn.Linear(d, K), torch.nn.Linear(extra, 1, bias=False), d

    def forward(self, x):
        z = self.base(x[:, :self.d])
        return z + torch.cat([torch.zeros(len(x), K - 1), self.g(x[:, self.d:])], 1)


def shrunk_em(Z, kappa, iters=100):
    """EM for the site prior with a symmetric Dirichlet pull of total weight kappa; the head assumes a uniform prior."""
    p = torch.softmax(torch.tensor(Z), -1).numpy()
    pi = np.full(K, 1 / K)
    for _ in range(iters):
        q = p * pi
        q /= q.sum(1, keepdims=True)
        pi = (q.sum(0) + kappa / K) / (len(Z) + kappa)
    return Z + np.log(pi * K)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--spectrum", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    X = load_x(a.features, "medsiglip", t)
    S = np.load(a.spectrum, mmap_mode="r")
    dev = np.isin(sp, ["train", "validation"])
    pos = np.full(len(t), -1)
    pos[np.flatnonzero(dev)] = np.arange(dev.sum())
    rows = []
    for s in sorted(set(src[sp == "train"])):
        tr, va, held = (sp == "train") & (src != s), (sp == "validation") & (src != s), dev & (src == s)
        P = project(S, tr, dev, 128, 0)
        w = balanced_weights(y[tr].tolist())
        D = float((X[held] ** 2).mean())  # mean squared z-score: the diagonal Mahalanobis distance per dimension
        for seed in a.seeds:
            torch.manual_seed(seed)
            cat = lambda m: np.concatenate([X[m], P[pos[m]]], 1)
            g = train_strata(cat(tr), y[tr], src[tr], w, cat(va), y[va], 0.3, seed, head=GOnly(X.shape[1], P.shape[1]))
            h = train_strata(X[tr], y[tr], src[tr], w, X[va], y[va], 0.3, seed)
            with torch.no_grad():
                Zg, Z = g(torch.tensor(cat(held))).numpy(), h(torch.tensor(X[held])).numpy()
            arms = {"global": Z, "polar_g_only": Zg}
            for kappa in [30, 100, 300, 1000, 3000]:
                arms[f"kappa{kappa}"] = shrunk_em(Z, kappa)
            for beta in [.25, .5, 1]:
                arms[f"beta{beta}"] = Z / (1 + D) ** beta
            yy = y[held]
            for arm, z in arms.items():
                pred = z.argmax(1)
                rows.append(dict(seed=seed, source=s, arm=arm, D=D, **metrics(z, yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan))
            print(s, seed, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(Path(a.out) / "gate3.csv", index=False)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G"]].mean().round(3).to_string())
    print(d.pivot_table(index="source", columns="arm", values="bacc").round(3).to_string())


if __name__ == "__main__":
    main()
