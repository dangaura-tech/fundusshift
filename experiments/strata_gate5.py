"""STRATA gate 5: per-image correction between the site-cancelled head and the zero-shot text logits, after the paired full-source and source-excluded training of CoREx.
It reads one image at a time, unlike the batch network of experiments.shoal. Leave-one-development-source-out; reads no test row.
python -m experiments.strata_gate5 --bench data/bench --features runs/features --zeroshot runs/medsiglip --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import balanced_weights, load_table
from .shoal import K, load_x, metrics
from .strata import pair_loss, train_strata
from .strata_gate4 import anchor_logits


class Corrector(torch.nn.Module):
    """z = g * z_p + (1 - g) * z_c + delta, with a class-wise gate g; starts as the plain average."""

    def __init__(self, hidden=32, residual=True):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2 * K, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2 * K))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.residual = residual

    def forward(self, zp, zc):
        zp, zc = zp - zp.mean(1, keepdim=True), zc - zc.mean(1, keepdim=True)
        o = self.net(torch.cat([zp, zc], 1))
        g = torch.sigmoid(o[:, :K])
        return g * zp + (1 - g) * zc + (o[:, K:] if self.residual else 0)


def ce_brier(z, y):
    p = torch.softmax(z, 1)
    return torch.nn.functional.cross_entropy(z, y) + .1 * ((p - torch.nn.functional.one_hot(y, K)) ** 2).sum(1).mean()


def fit_corrector(pools, seed, use=("full", "excl"), residual=True, pairs=False, steps=1500, n=256):
    """pools: one dict per source with full, excl, text, y. Each step: one source, classes equally."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Corrector(residual=residual)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        p = pools[rng.integers(len(pools))]
        y = p["y"]
        cl = np.unique(y)
        c = rng.choice(cl, n)
        i = np.array([rng.choice(np.flatnonzero(y == k)) for k in c])
        yb = torch.tensor(y[i])
        loss = 0
        for u in use:
            z = net(torch.tensor(p[u][i]), torch.tensor(p["text"][i]))
            loss = loss + ce_brier(z, yb) / len(use)
            if pairs:
                j = rng.permutation(n)
                m = torch.tensor(y[i] != y[i][j])
                if m.any():
                    loss = loss + pair_loss(z[m], z[j][m], yb[m], yb[j][m]) / len(use)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


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
    dev = np.isin(sp, ["train", "validation"])
    rows = []

    def head(m, va, seed):
        h = train_strata(X[m], y[m], src[m], balanced_weights(y[m].tolist()), X[va], y[va], 0.3, seed)
        with torch.no_grad():
            return h(torch.tensor(X)).numpy()

    for s in sorted(set(src[sp == "train"])):
        tr, va, held = (sp == "train") & (src != s), (sp == "validation") & (src != s), dev & (src == s)
        T = anchor_logits(a.zeroshot, a.model, t, va)
        for seed in a.seeds:
            full = head(tr, va, seed)
            scale = full[va].std()
            pools = []
            for o in sorted(set(src[tr])):
                m = tr & (src != o)
                if len(np.unique(y[m])) < K:  # no glaucoma left without these two sources
                    continue
                excl = head(m, va & (src != o), seed)
                r = dev & (src == o)
                pools.append(dict(full=full[r] / scale, excl=excl[r] / scale, text=T[r], y=y[r]))
            P, C, yy = torch.tensor(full[held] / scale), torch.tensor(T[held]), y[held]
            nets = {"correx": fit_corrector(pools, seed), "excl_only": fit_corrector(pools, seed, use=("excl",)),
                    "full_only": fit_corrector(pools, seed, use=("full",)), "gate_only": fit_corrector(pools, seed, residual=False),
                    "correx_pairs": fit_corrector(pools, seed, pairs=True)}
            with torch.no_grad():
                res = {k: n(P, C).numpy() for k, n in nets.items()}
            res |= {"head": P.numpy(), "text": C.numpy(), "average": ((P + C) / 2).numpy()}
            for arm, z in res.items():
                pred = z.argmax(1)
                rows.append(dict(seed=seed, source=s, arm=arm, **metrics(z, yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan))
            print(s, seed, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(Path(a.out) / "gate5.csv", index=False)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G"]].mean().round(3).to_string())
    print(d.pivot_table(index="source", columns="arm", values="bacc").round(3).to_string())


if __name__ == "__main__":
    main()
