"""STRATA gate 1: linear head with a conditional-logit loss over pairs from one site, plus lambda * the usual loss, on frozen features.
Leave-one-development-source-out only; reads no test row. The baseline is the same head with the usual loss alone (lambda = inf).
python -m experiments.strata --bench data/bench --features runs/features --out results/strata --models medsiglip
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .calib import loss_fn
from .data import balanced_weights, load_table
from .probe_calib import train_head
from .shoal import load_x, metrics


def pair_loss(Zi, Zj, a, b):
    """-log P(y_i = a, y_j = b | {a, b}, site). Any per-site constant added to a class's logits cancels."""
    r = torch.arange(len(a))
    return torch.nn.functional.softplus(-((Zi[r, a] - Zi[r, b]) - (Zj[r, a] - Zj[r, b]))).mean()


def pair_table(y, src):
    """{(a, b): [(indices of a, indices of b), one entry per site holding both]}."""
    out = {}
    for s in np.unique(src):
        cl = np.unique(y[src == s])
        for a in cl:
            for b in cl[cl > a]:
                out.setdefault((int(a), int(b)), []).append((np.flatnonzero((src == s) & (y == a)), np.flatnonzero((src == s) & (y == b))))
    return out


def train_strata(X, y, src, w, Xv, yv, lam, seed, epochs=40, bs=256, lr=1e-3, wd=1e-4, head=None, aug=None):
    """Each step draws a class pair uniformly, then a site holding it, so the rare glaucoma pairs are not swamped."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = head or torch.nn.Linear(X.shape[1], int(y.max()) + 1)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    f = loss_fn("ls")
    Xt, yt, Xvt = torch.tensor(X), torch.tensor(y), torch.tensor(Xv)
    table = pair_table(y, src)
    types = sorted(table)
    best, state = -1.0, None
    for _ in range(epochs):
        idx = rng.choice(len(y), size=len(y), replace=True, p=w / w.sum())
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            ta, tb, A, B = [], [], [], []
            for _ in range(bs // 2):
                t = types[rng.integers(len(types))]
                ia, ib = table[t][rng.integers(len(table[t]))]
                A.append(ia[rng.integers(len(ia))]); B.append(ib[rng.integers(len(ib))])
                ta.append(t[0]); tb.append(t[1])
            opt.zero_grad()
            (pair_loss(head(Xt[A]), head(Xt[B]), torch.tensor(ta), torch.tensor(tb)) + lam * f(head(aug(Xt[b], rng) if aug else Xt[b]), yt[b])).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            v = balanced_accuracy_score(yv, head(Xvt).argmax(1).numpy())
        head.train()
        if v > best:
            best, state = v, {k: p.clone() for k, p in head.state_dict().items()}
    head.load_state_dict(state)
    return head.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=["medsiglip"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--lams", type=float, nargs="*", default=[0.1, 0.3, 1.0, 3.0])
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, classes = load_table(a.bench)
    assert classes == ["N", "D", "G"]
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    rows = []
    for name in a.models:
        X = load_x(a.features, name, t)
        for seed in a.seeds:
            for s in sorted(set(src[sp == "train"])):
                tr, va = (sp == "train") & (src != s), (sp == "validation") & (src != s)
                held = np.isin(sp, ["train", "validation"]) & (src == s)
                w = balanced_weights(y[tr].tolist())
                heads = {"ce": train_head(X[tr], y[tr], w, X[va], y[va], "ls", seed)}
                for lam in a.lams:
                    heads[f"strata_l{lam}"] = train_strata(X[tr], y[tr], src[tr], w, X[va], y[va], lam, seed)
                    # ablation: same loss and sampling, pairs drawn across sites, so nothing cancels
                    heads[f"cross_l{lam}"] = train_strata(X[tr], y[tr], np.full(tr.sum(), "all"), w, X[va], y[va], lam, seed)
                for arm, h in heads.items():
                    with torch.no_grad():
                        Z = h(torch.tensor(X[held])).numpy()
                    pred, yy = Z.argmax(1), y[held]
                    rec = {f"recall_{c}": (pred[yy == k] == k).mean() if (yy == k).any() else np.nan for k, c in enumerate(classes)}
                    rows.append(dict(model=name, seed=seed, source=s, arm=arm, n=int(held.sum()), **metrics(Z, yy), **rec))
                print(name, seed, s, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "gate1.csv", index=False)
    print(d.groupby(["model", "arm"])[["bacc", "ece", "nll", "recall_N", "recall_D", "recall_G"]].mean().round(3).to_string())
    print(d.pivot_table(index="source", columns="arm", values="bacc").round(3).to_string())


if __name__ == "__main__":
    main()
