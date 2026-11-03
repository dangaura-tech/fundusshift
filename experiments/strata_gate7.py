"""STRATA gate 7: the full pipeline against the same pipeline with the site-cancelling loss removed. A pipeline: loss, optional between-site projection, optional bag of heads that each leave out one more site, late fusion with text.
Leave-one-development-source-out; reads no test row.
python -m experiments.strata_gate7 --bench data/bench --features runs/features --zeroshot runs/medsiglip --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import balanced_weights, load_table
from .probe_calib import train_head
from .shoal import K, load_x, metrics
from .strata import train_strata
from .strata_gate4 import BANKS, anchor_logits, nap


def scaled(Z, va):
    Z = Z - Z.mean(1, keepdims=True)
    return Z / Z[va].std()


def one(X, y, src, m, va, loss, proj, seed):
    """Logits for every row from one head fitted on rows m."""
    P = X
    if proj:
        U = nap(X[m], y[m], src[m], proj)
        P = X - (X @ U.T) @ U
    w = balanced_weights(y[m].tolist())
    if loss == "ce":
        h = train_head(P[m], y[m], w, P[va], y[va], "ls", seed)
    else:
        h = train_strata(P[m], y[m], src[m] if loss == "strata" else np.full(m.sum(), "all"), w, P[va], y[va], .3, seed)
    with torch.no_grad():
        return h(torch.tensor(P)).numpy()


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
        T = anchor_logits(a.zeroshot, a.model, t, va)
        Tm = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_{a.model}_{b}.npy").astype(np.float32), va) for b in BANKS], 0)
        subs = [tr & (src != o) for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
        for seed in a.seeds:
            res = {}
            for loss in ["ce", "cross", "strata"]:
                for proj in [0, 32]:
                    single = scaled(one(X, y, src, tr, va, loss, proj, seed), va)[held]
                    bagged = np.mean([scaled(one(X, y, src, m, va & np.isin(src, np.unique(src[m])), loss, proj, seed), va)[held] for m in subs], 0)
                    for name, z in [("single", single), ("bag", bagged)]:
                        key = f"{loss}_{name}{'_nap' if proj else ''}"
                        res[key] = z
                        res[key + "+text"] = .5 * z + .5 * T[held]
                        res[key + "+allbanks"] = .5 * z + .5 * Tm[held]
            yy = y[held]
            for arm, z in res.items():
                pred = z.argmax(1)
                rows.append(dict(seed=seed, source=s, arm=arm, **metrics(z, yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan))
            print(s, seed, flush=True)
            pd.DataFrame(rows).to_csv(Path(a.out) / "gate7.csv", index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G"]].mean().round(3).sort_values("bacc").to_string())
    print(d[d.arm.str.contains("bag") & d.arm.str.endswith("+text")].pivot_table(index="source", columns="arm", values="bacc").round(3).to_string())


if __name__ == "__main__":
    main()
