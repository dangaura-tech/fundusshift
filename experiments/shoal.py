"""SHOAL (Source-Held-Out Adaptive Logits): a set network reads an unlabelled batch from one site and outputs a temperature and a class bias for that batch.

Its training data are cross-fitted: each development source is scored by a head that never saw it, so the set network learns from real source shift.
dev: leave-one-development-source-out comparison with baselines and ablations; reads no test row.
final: applies the config frozen in --config; writes logits per site for experiments.full_eval (runs/shoal).
python -m experiments.shoal dev --bench data/bench --features runs/features --out results/shoal --models medsiglip
python -m experiments.shoal final --bench data/bench --features runs/features --config docs/shoal_config.json --out runs/shoal
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .data import balanced_weights, load_table
from .probe_calib import train_head

CFG = {"hidden": 32, "steps": 1500, "lr": 1e-3, "wd": 1e-4, "reg": 0.01, "alpha": 1.0, "nmin": 32, "nmax": 256, "mix": 1}
K = 3


def tokens(Z):
    """Per-image input: centred logits, probabilities, entropy."""
    p = torch.softmax(Z, -1)
    return torch.cat([Z - Z.mean(-1, keepdim=True), p, -(p * p.clamp_min(1e-9).log()).sum(-1, keepdim=True)], -1)


class SetNet(torch.nn.Module):
    """Pools the batch by mean and std; starts as the identity (T = 1, bias = 0)."""

    def __init__(self, hidden=32, use_t=True, use_b=True):
        super().__init__()
        d = 2 * K + 1
        self.phi = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
        self.rho = torch.nn.Sequential(torch.nn.Linear(2 * hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, K + 1))
        torch.nn.init.zeros_(self.rho[-1].weight)
        torch.nn.init.zeros_(self.rho[-1].bias)
        self.use_t, self.use_b = use_t, use_b

    def params(self, Z):
        h = self.phi(tokens(Z))
        o = self.rho(torch.cat([h.mean(0), h.std(0, unbiased=False)]))
        return (o[0] if self.use_t else o[0] * 0), (o[1:] - o[1:].mean() if self.use_b else o[1:] * 0)

    def forward(self, Z):
        logt, b = self.params(Z)
        return Z / logt.exp() + b, logt, b


def episode(y, rng, cfg):
    """Indices of one batch from one source, with a prior drawn over its classes."""
    cl = np.unique(y)
    pi = rng.dirichlet(np.full(len(cl), cfg["alpha"]))
    n = int(rng.integers(cfg["nmin"], cfg["nmax"] + 1))
    c = rng.choice(cl, n, p=pi)
    return np.array([rng.choice(np.flatnonzero(y == k)) for k in c])


def fit_setnet(pools, seed, cfg=CFG, use_t=True, use_b=True):
    """pools: list of (logits, labels), one per source."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SetNet(cfg["hidden"], use_t, use_b)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    for _ in range(cfg["steps"]):
        # a synthetic site: 1 to cfg["mix"] sources pooled, which widens the few real ones
        take = rng.choice(len(pools), rng.integers(1, cfg["mix"] + 1), replace=False)
        y = np.concatenate([pools[j][1] for j in take])
        z = torch.tensor(np.concatenate([pools[j][0] for j in take]))
        i = episode(y, rng, cfg)
        out, logt, b = net(z[i])
        loss = torch.nn.functional.cross_entropy(out, torch.tensor(y[i])) + cfg["reg"] * (logt ** 2 + (b ** 2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


def apply_net(net, Z):
    with torch.no_grad():
        return net(torch.tensor(Z))[0].numpy()


def fit_t(Z, y):
    grid = np.exp(np.linspace(np.log(.05), np.log(20), 200))
    z, yt = torch.tensor(Z), torch.tensor(y)
    return float(grid[int(np.argmin([torch.nn.functional.cross_entropy(z / g, yt).item() for g in grid]))])


def em_prior(Z, train_prior, iters=100):
    """Saerens et al. 2002, starting from the training prior; returns adjusted logits."""
    p = torch.softmax(torch.tensor(Z), -1).numpy()
    pi = train_prior.copy()
    for _ in range(iters):
        q = p * pi / train_prior
        q /= q.sum(1, keepdims=True)
        pi = q.mean(0)
    return Z + np.log(pi.clip(1e-6) / train_prior)


def metrics(Z, y):
    p = torch.softmax(torch.tensor(Z), -1).numpy()
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, 16)
    ece = sum(abs((pred == y)[m].mean() - conf[m].mean()) * m.mean() for lo, hi in zip(edges, edges[1:]) if (m := (conf > lo) & (conf <= hi)).any())
    nll = -np.log(p[np.arange(len(y)), y].clip(1e-9)).mean()
    return {"bacc": balanced_accuracy_score(y, pred), "ece": ece, "nll": nll}


def batches(n, size, seed):
    """Disjoint batches in random order; size 0 means the whole site."""
    if not size or size >= n:
        return [np.arange(n)]
    i = np.random.default_rng(seed).permutation(n)
    return [b for b in np.array_split(i, max(1, n // size))]


def by_batch(net, Z, size, seed=0):
    out = np.empty_like(Z)
    for b in batches(len(Z), size, seed):
        out[b] = apply_net(net, Z[b])
    return out


def crossfit(X, t, sources, seed):
    """Out-of-source logits for train and validation rows of each development source."""
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    pools, heads = {}, {}
    for s in sources:
        tr = (sp == "train") & (src != s)
        va = (sp == "validation") & (src != s)
        h = train_head(X[tr], y[tr], balanced_weights(y[tr].tolist()), X[va], y[va], "ls", seed)
        m = np.isin(sp, ["train", "validation"]) & (src == s)
        with torch.no_grad():
            pools[s] = (h(torch.tensor(X[m])).numpy(), y[m])
        heads[s] = h
    return pools, heads


def load_x(features, name, t):
    X = np.load(Path(features) / f"{name}.npy").astype(np.float32)
    tr = t.split.to_numpy() == "train"
    return (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)


def dev(a):
    """Every arm is fitted without the held-out source, which is scored by a head that never saw it. No test row is read."""
    t, classes = load_table(a.bench)
    assert classes == ["N", "D", "G"]
    t = t.copy()
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    sources = sorted(set(src[sp == "train"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in a.models:
        X = load_x(a.features, name, t)
        for seed in a.seeds:
            pools, _ = crossfit(X, t, sources, seed)
            for s in sources:
                Zs, ys = pools[s]
                others = [pools[o] for o in sources if o != s]
                Zo, yo = np.concatenate([p[0] for p in others]), np.concatenate([p[1] for p in others])
                T = fit_t(Zo, yo)
                pri = np.full(K, 1 / K)  # the head is class-weighted, so its logits assume a uniform prior
                # calibration rows of the other sources scored in-sample style: no source shift seen, the ablation of cross-fitting
                tr = (sp == "train") & (src != s)
                va = (sp == "validation") & (src != s)
                h = train_head(X[tr], y[tr], balanced_weights(y[tr].tolist()), X[va], y[va], "ls", seed)
                with torch.no_grad():
                    Zin = [(h(torch.tensor(X[(sp == "calibration") & (src == o)])).numpy(), y[(sp == "calibration") & (src == o)]) for o in sources if o != s]
                nets = {"shoal": fit_setnet(others, seed), "shoal_t_only": fit_setnet(others, seed, use_b=False),
                        "shoal_b_only": fit_setnet(others, seed, use_t=False), "shoal_no_crossfit": fit_setnet(Zin, seed)}
                for reg in a.grid_reg:
                    for mix in a.grid_mix:
                        nets[f"shoal_r{reg}_m{mix}"] = fit_setnet(others, seed, {**CFG, "reg": reg, "mix": mix})
                # a held-out source's prior is drawn the same way the arms will meet it: natural and resampled
                rng = np.random.default_rng(1000 + seed)
                views = {"natural": np.arange(len(ys))}
                for j in range(a.resamples):
                    views[f"prior{j}"] = episode_fixed(ys, rng)
                for v, idx in views.items():
                    Z, yy = Zs[idx], ys[idx]
                    for size in a.batch_sizes:
                        arms = {"raw": Z, "ts_pooled": Z / T, "em_ts": concat_batches(lambda b: em_prior(Z[b] / T, pri), len(Z), Z.shape, size)}
                        for k, n in nets.items():
                            arms[k] = by_batch(n, Z, size)
                        for arm, zz in arms.items():
                            rows.append(dict(model=name, seed=seed, source=s, view=v, batch=size, arm=arm, n=len(yy), **metrics(zz, yy)))
                print(name, seed, s, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "dev_loso.csv", index=False)
    s = d.groupby(["model", "batch", "arm"])[["bacc", "ece", "nll"]].mean().round(4)
    s.to_csv(out / "dev_summary.csv")
    print(s.to_string())


def episode_fixed(y, rng):
    """A held-out view with a random prior and every class of the source kept."""
    cl = np.unique(y)
    pi = rng.dirichlet(np.ones(len(cl)))
    n = len(y)
    c = rng.choice(cl, n, p=pi)
    return np.array([rng.choice(np.flatnonzero(y == k)) for k in c])


def concat_batches(fn, n, shape, size):
    out = np.empty(shape, np.float32)
    for b in batches(n, size, 0):
        out[b] = fn(b)
    return out


def final(a):
    """Applies the frozen config. Rows are grouped by split and source, as a site would meet them."""
    t, classes = load_table(a.bench)
    assert classes == ["N", "D", "G"]
    cfg = json.loads(Path(a.config).read_text())
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    sources = sorted(set(src[sp == "train"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, va = sp == "train", sp == "validation"
    for name in cfg["models"]:
        X = load_x(a.features, name, t)
        for seed in cfg["seeds"]:
            f = out / f"logits_{name}_s{seed}.npy"
            if f.exists():
                continue
            pools, _ = crossfit(X, t, sources, seed)
            net = fit_setnet(list(pools.values()), seed, {**CFG, **cfg.get("setnet", {})})
            h = train_head(X[tr], y[tr], balanced_weights(y[tr].tolist()), X[va], y[va], "ls", seed)
            with torch.no_grad():
                Z = h(torch.tensor(X)).numpy()
            res = Z.copy()
            for k in sorted(set(zip(sp, src))):
                m = np.flatnonzero((sp == k[0]) & (src == k[1]))
                res[m] = by_batch(net, Z[m], cfg["batch"], seed)
            np.save(f, res)
            print("[ok]", f.name, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "final"])
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config")
    ap.add_argument("--models", nargs="*", default=["medsiglip"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[64, 0])
    ap.add_argument("--resamples", type=int, default=3)
    ap.add_argument("--grid-reg", type=float, nargs="*", default=[])
    ap.add_argument("--grid-mix", type=int, nargs="*", default=[1])
    a = ap.parse_args(argv)
    (dev if a.cmd == "dev" else final)(a)


if __name__ == "__main__":
    main()
