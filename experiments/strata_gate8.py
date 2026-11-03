"""STRATA gate 8: the CoREx correction cut down to fusion weights. Each bag head never saw its left-out site, so its logits there are out of site by construction; the weights of head and text banks are fitted on those rows only.
There is no residual and no class bias: with GRAPE held out the pools hold no glaucoma, and a free bias learns never to predict it (gate 5). Leave-one-development-source-out; reads no test row.
python -m experiments.strata_gate8 --bench data/bench --features runs/features --zeroshot runs/medsiglip --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_table
from .shoal import K, load_x, metrics
from .strata_gate4 import BANKS
from .strata_gate7 import one, scaled


def fit_weights(pools, per_class, shrink=.01, steps=400):
    """pools: (Z, y), Z of shape (n, branches, K). Weights start at head .5 and text .5 shared across banks, and are shrunk towards it."""
    b = pools[0][0].shape[1]
    w0 = torch.tensor([.5] + [.5 / (b - 1)] * (b - 1)).log()
    a = (w0.repeat(K, 1) if per_class else w0[None]).clone().requires_grad_()
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([a, t], lr=.03)
    for _ in range(steps):
        w = torch.softmax(a, 1)
        loss = shrink * ((a - w0) ** 2).sum()
        for Z, y in pools:
            z = (torch.tensor(Z) * w.T[None]).sum(1) * t.exp()
            present = np.unique(y)
            yt = torch.tensor(y)
            # classes count equally inside a site, and sites count equally
            loss = loss + torch.stack([torch.nn.functional.cross_entropy(z[yt == c], yt[yt == c]) for c in present]).mean() / len(pools)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(a, 1).detach(), t.exp().detach()


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
    rows, W = [], []
    for s in sorted(set(src[sp == "train"])):
        tr, va, held = (sp == "train") & (src != s), (sp == "validation") & (src != s), dev & (src == s)
        T = [scaled(np.load(Path(a.zeroshot) / f"logits_{a.model}_{b}.npy").astype(np.float32), va) for b in BANKS]
        others = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
        for seed in a.seeds:
            pools, bag = [], []
            for o in others:
                m = tr & (src != o)
                z = scaled(one(X, y, src, m, va & (src != o), "ce", 32, seed), va)
                bag.append(z[held])
                r = dev & (src == o)
                pools.append((np.stack([z[r]] + [b[r] for b in T], 1), y[r]))
            Zh = torch.tensor(np.stack([np.mean(bag, 0)] + [b[held] for b in T], 1))
            leader = .5 * Zh[:, 0] + .5 * Zh[:, 1:].mean(1)
            res = {"leader": leader}
            for name, pc in [("stack", False), ("stack_class", True)]:
                w, tau = fit_weights(pools, pc)
                res[name] = (Zh * w.T[None]).sum(1) * tau
                W.append(dict(seed=seed, source=s, arm=name, tau=float(tau), **{f"w{i}": v for i, v in enumerate(w.flatten().tolist())}))
            yy = y[held]
            for arm, z in res.items():
                z = z.numpy()
                pred = z.argmax(1)
                rows.append(dict(seed=seed, source=s, arm=arm, **metrics(z, yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan))
            print(s, seed, flush=True)
            pd.DataFrame(rows).to_csv(Path(a.out) / "gate8.csv", index=False)
    d = pd.DataFrame(rows)
    pd.DataFrame(W).to_csv(Path(a.out) / "gate8_weights.csv", index=False)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G"]].mean().round(3).to_string())
    print(d.pivot_table(index="source", columns="arm", values="bacc").round(3).to_string())
    print(pd.DataFrame(W).query("arm=='stack'").groupby("source")[["tau", "w0", "w1", "w2", "w3"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
