"""Gate 15: two modules that each fixed one failure. Training on an image and its mirror lowered false G; the stronger head regularisation of gate 12 raised recall G. Here together. Leave-one-development-source-out; reads no test row.
python -m experiments.strata_gate15 --bench data/bench --features runs/features --zeroshot runs/medsiglip --strata data/strata --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import balanced_weights, load_table
from .probe_calib import train_head
from .shoal import K, metrics
from .strata_gate4 import BANKS, nap
from .strata_gate7 import scaled

RECIPES = {"base": {}, "mid": dict(epochs=20, bs=128, lr=5e-4, wd=1e-3), "reg": dict(epochs=12, bs=64, lr=3e-4, wd=1e-2)}


def fit(X, y, src, m, va, seed, kw):
    U = nap(X[m], y[m], src[m], 32)
    P = X - (X @ U.T) @ U
    h = train_head(P[m], y[m], balanced_weights(y[m].tolist()), P[va], y[va], "ls", seed, **kw)
    with torch.no_grad():
        return h(torch.tensor(P)).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "features", "zeroshot", "strata", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)
    t, _ = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr0, dv, n = sp == "train", np.isin(sp, ["train", "validation"]), len(t)
    R, M = np.load(Path(a.features) / "medsiglip.npy").astype(np.float32), np.load(Path(a.strata) / "mirror_medsiglip.npy").astype(np.float32)
    mu, sd = R[tr0].mean(0), R[tr0].std(0) + 1e-6
    X2, y2, src2 = np.concatenate([(R - mu) / sd, (M - mu) / sd]), np.concatenate([y, y]), np.concatenate([src, src])
    left = np.load(Path(a.strata) / "disc.npy")[:, 0] < .5
    out, rows = Path(a.out) / "gate15", []
    out.mkdir(parents=True, exist_ok=True)
    for s in sorted(set(src[tr0])):
        tr, va, held = tr0 & (src != s), (sp == "validation") & (src != s), dv & (src == s)
        T = np.mean([.5 * (scaled(np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32), va) + scaled(np.load(Path(a.strata) / f"mirror_logits_{b}.npy").astype(np.float32), va)) for b in BANKS], 0)[held]
        others = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
        yy, ll, N = y[held], left[held], y[held] == 0
        for seed in a.seeds:
            for name, kw in RECIPES.items():
                zs = []
                for o in others:
                    m, vo = tr & (src != o), va & (src != o)
                    z = fit(X2, y2, src2, np.concatenate([m, m]), np.concatenate([vo, vo]), seed, kw)
                    zs.append(.5 * (scaled(z[:n], va)[held] + scaled(z[n:], va)[held]))
                z = .5 * np.mean(zs, 0) + .5 * T
                np.save(out / f"{name}_{s}_s{seed}.npy", z)
                pred = z.argmax(1)
                rows.append(dict(seed=seed, source=s, arm=name, **metrics(z.astype(np.float32), yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan, false_G=(pred[yy != 2] == 2).mean() if (yy != 2).any() else np.nan, N_G_left=(pred[N & ll] == 2).mean() if (N & ll).any() else np.nan, N_G_right=(pred[N & ~ll] == 2).mean() if (N & ~ll).any() else np.nan))
            print(s, seed, flush=True)
            pd.DataFrame(rows).to_csv(Path(a.out) / "gate15.csv", index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G", "false_G", "N_G_left", "N_G_right"]].mean().round(3).sort_values("bacc").to_string())
    print(d.pivot_table(index="source", columns="arm", values="bacc").round(3).T.to_string())
    print(d.pivot_table(index="seed", columns="arm", values="bacc").round(3).to_string())


if __name__ == "__main__":
    main()
