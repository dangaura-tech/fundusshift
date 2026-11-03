"""Gate 12: why the mean-token control of experiments.rptoken reached .760 while the same input reached .688 through train_head. Same pipeline, training recipe varied. Leave-one-development-source-out; reads no test row.
python -m experiments.strata_gate12 --bench data/bench --features runs/features --zeroshot runs/medsiglip --tokens data/strata --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import balanced_weights, load_table
from .probe_calib import train_head
from .shoal import K, load_x, metrics
from .strata_gate4 import BANKS, nap
from .strata_gate7 import scaled
from .strata_gate10 import mean_tokens

RECIPES = {
    "base": dict(cls=False, kw={}),
    "reg": dict(cls=False, kw=dict(epochs=12, bs=64, lr=3e-4, wd=1e-2)),
}


def fit(X, y, src, m, va, proj, seed, r):
    P = X
    if proj:
        U = nap(X[m], y[m], src[m], proj)
        P = X - (X @ U.T) @ U
    w = 1 / np.bincount(y[m], minlength=K)[y[m]] if r["cls"] else balanced_weights(y[m].tolist())
    h = train_head(P[m], y[m], w, P[va], y[va], "ls", seed, **r["kw"])
    with torch.no_grad():
        return h(torch.tensor(P)).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "features", "zeroshot", "tokens", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)
    t, _ = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr0 = sp == "train"
    v = mean_tokens(a.tokens)[:, 2]
    inputs = {"pooled": load_x(a.features, "medsiglip", t), "last": (v - v[tr0].mean(0)) / (v[tr0].std(0) + 1e-6)}
    rows = []
    for s in sorted(set(src[tr0])):
        tr, va, held = tr0 & (src != s), (sp == "validation") & (src != s), np.isin(sp, ["train", "validation"]) & (src == s)
        Tm = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32), va) for b in BANKS], 0)[held]
        others = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
        yy = y[held]
        for seed in a.seeds:
            for name, X in inputs.items():
                for rn, r in RECIPES.items():
                    single = scaled(fit(X, y, src, tr, va, 0, seed, r), va)[held]
                    bn = np.mean([scaled(fit(X, y, src, tr & (src != o), va & (src != o), 32, seed, r), va)[held] for o in others], 0)
                    for pipe, z in [("single", single), ("bag_nap", bn), ("single+text", .5 * single + .5 * Tm), ("bag_nap+text", .5 * bn + .5 * Tm)]:
                        pred = z.argmax(1)
                        rows.append(dict(seed=seed, source=s, arm=f"{name}_{rn}_{pipe}", **metrics(z, yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan, false_G=(pred[yy != 2] == 2).mean() if (yy != 2).any() else np.nan))
            print(s, seed, flush=True)
            pd.DataFrame(rows).to_csv(Path(a.out) / "gate12.csv", index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G", "false_G"]].mean().round(3).sort_values("bacc").to_string())
    print(d[d.arm.str.endswith("bag_nap+text")].pivot_table(index="source", columns="arm", values="bacc").round(3).T.to_string())


if __name__ == "__main__":
    main()
