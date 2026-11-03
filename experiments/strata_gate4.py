"""STRATA gate 4: what can join the conditional-logit head. Leave-one-development-source-out on MedSigLIP; reads no test row.
anchor: zero-shot text logits as a fixed site-free evidence, the head learns a residual. nap: project out the leading between-site directions within class.
aug: random site offset in feature space on the usual loss. bag: average of heads that each leave out one more site.
python -m experiments.strata_gate4 --bench data/bench --features runs/features --zeroshot runs/medsiglip --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .data import balanced_weights, load_table
from .shoal import K, load_x, metrics
from .strata import train_strata

BANKS = ["clip_ensemble", "cupl", "descriptive_single"]


class Anchor(torch.nn.Module):
    """The last K inputs are the text logits: output = gamma * text + linear(feature)."""

    def __init__(self, d, gamma):
        super().__init__()
        self.f, self.gamma, self.d = torch.nn.Linear(d, K), gamma, d
        torch.nn.init.zeros_(self.f.weight)
        torch.nn.init.zeros_(self.f.bias)

    def forward(self, x):
        return self.f(x[:, :self.d]) + self.gamma * x[:, self.d:]


def site_means(X, y, src):
    return np.stack([X[(src == s) & (y == k)].mean(0) - X[y == k].mean(0) for k in np.unique(y) for s in np.unique(src) if ((src == s) & (y == k)).any()])


def nap(X, y, src, k):
    """Orthonormal basis of the k leading within-class between-site directions."""
    return np.linalg.svd(site_means(X, y, src), full_matrices=False)[2][:k]


def anchor_logits(path, name, t, va):
    """Prompt bank by validation bAcc of the fold; centred per image, unit scale on the fold."""
    y = t.y.to_numpy()
    b = max(BANKS, key=lambda n: balanced_accuracy_score(y[va], np.load(Path(path) / f"logits_{name}_{n}.npy")[va].argmax(1)))
    Z = np.load(Path(path) / f"logits_{name}_{b}.npy").astype(np.float32)
    Z = Z - Z.mean(1, keepdims=True)
    return Z / Z[va].std()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--zeroshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="medsiglip")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    X = load_x(a.features, a.model, t)
    rows = []
    for s in sorted(set(src[sp == "train"])):
        tr, va, held = (sp == "train") & (src != s), (sp == "validation") & (src != s), np.isin(sp, ["train", "validation"]) & (src == s)
        w = balanced_weights(y[tr].tolist())
        T = anchor_logits(a.zeroshot, a.model, t, va)
        D = X.shape[1]
        off = site_means(X[tr], y[tr], src[tr])
        cat = np.concatenate([X, T], 1)
        fit = lambda XX, seed, **kw: train_strata(XX[tr], y[tr], src[tr], w, XX[va], y[va], 0.3, seed, **kw)
        for seed in a.seeds:
            arms = {"base": (fit(X, seed), X)}
            arms["text_only"] = (Anchor(D, 1.0).eval(), cat)
            for g in [0.5, 1.0, 2.0]:
                arms[f"anchor{g}"] = (fit(cat, seed, head=Anchor(D, g)), cat)
            for k in [8, 32, 128]:
                U = nap(X[tr], y[tr], src[tr], k)
                P = X - (X @ U.T) @ U
                arms[f"nap{k}"] = (fit(P, seed), P)
            aug = lambda x, rng: x + torch.tensor(off[rng.integers(len(off), size=len(x))] * rng.random((len(x), 1)), dtype=x.dtype)
            arms["aug"] = (fit(X, seed, aug=aug), X)
            res = {}
            for arm, (h, XX) in arms.items():
                with torch.no_grad():
                    res[arm] = h(torch.tensor(XX[held])).numpy()
            bag = []
            for o in sorted(set(src[tr])):
                m = tr & (src != o)
                if len(np.unique(y[m])) == K:
                    h = train_strata(X[m], y[m], src[m], balanced_weights(y[m].tolist()), X[va], y[va], 0.3, seed)
                    with torch.no_grad():
                        bag.append(h(torch.tensor(X[held])).numpy())
            res["bag"] = np.mean(bag, 0)
            yy = y[held]
            for arm, z in res.items():
                pred = z.argmax(1)
                rows.append(dict(seed=seed, source=s, arm=arm, **metrics(z, yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan))
            print(s, seed, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(Path(a.out) / "gate4.csv", index=False)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G"]].mean().round(3).to_string())
    print(d.pivot_table(index="source", columns="arm", values="bacc").round(3).to_string())


if __name__ == "__main__":
    main()
