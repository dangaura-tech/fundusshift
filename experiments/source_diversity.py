"""Source diversity versus training-set size on frozen features (linear probe, logistic regression C = 0.05).

Pools: K training sources and N images per class, 5 random pools per cell (one when K = all). Tasks: D vs N, G vs N, 3-class.
A pool must contain every class of the task. G vs N can use few K because only Bangladesh and GRAPE publish glaucoma images in development.
Metric: balanced accuracy on test and on Chákṣu (classes of the task only), mean over pools with a bootstrap interval over pools.
python -m experiments.source_diversity --bench data/bench --features runs/features --out results/source_diversity
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from .data import load_table

TASKS = {"D_vs_N": ["N", "D"], "G_vs_N": ["N", "G"], "3class": ["N", "D", "G"]}


def sample_pool(tr, sources, classes, n, rng):
    """N images per class, spread as evenly as the chosen sources allow, one per leakage group."""
    idx = []
    for c in classes:
        cell = tr[(tr.label == c) & tr.source.isin(sources)].groupby("group_id").head(1)
        srcs = [s for s in sources if (cell.source == s).any()]
        if not srcs:
            return None
        got = []
        left = n
        for i, s in enumerate(sorted(srcs, key=lambda s: (cell.source == s).sum())):
            pool = cell[cell.source == s].index.to_numpy()
            take = min(len(pool), left // (len(srcs) - i))
            got += rng.choice(pool, take, replace=False).tolist()
            left -= take
        if left > 0:
            return None
        idx += got
    return np.array(idx)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=["dinov2_b14", "clip_b16", "siglip2_l16", "convnext_t", "resnet50", "vit_b16_in21k"])
    ap.add_argument("--pools", type=int, default=5)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, classes = load_table(a.bench)
    tr = t[t.split == "train"]
    rows = []
    for name in a.models:
        X = np.load(Path(a.features) / f"{name}.npy").astype(np.float32)
        X = (X - X[tr.index].mean(0)) / (X[tr.index].std(0) + 1e-6)
        for task, cl in TASKS.items():
            pool_src = sorted(tr[tr.label.isin(cl)].source.unique())
            tmask = t.label.isin(cl).to_numpy()
            y = t.label.map({c: i for i, c in enumerate(cl)}).to_numpy()
            for K in [1, 2, 4, len(pool_src)]:
                if K > len(pool_src):
                    continue
                combos = [c for c in combinations(pool_src, K) if all(tr[tr.source.isin(c)].label.eq(k).any() for k in cl)]
                if not combos:
                    continue
                rng = np.random.default_rng(0)
                picks = [combos[i] for i in rng.choice(len(combos), min(1 if K == len(pool_src) else a.pools, len(combos)), replace=False)]
                for N in [75, 150, 300]:
                    for p, c in enumerate(picks):
                        idx = sample_pool(tr, c, cl, N, np.random.default_rng(1000 * p + N))
                        if idx is None:
                            continue
                        clf = LogisticRegression(C=0.05, max_iter=500).fit(X[idx], y[idx])
                        row = {"model": name, "task": task, "K": K, "N": N, "pool": p, "sources": "+".join(c)}
                        for s in ["test", "test_ood"]:
                            m = (t.split == s).to_numpy() & tmask
                            row[s] = balanced_accuracy_score(y[m], clf.predict(X[m])) if len(set(y[m])) > 1 else np.nan
                        rows.append(row)
                print(name, task, "K", K, flush=True)
            pd.DataFrame(rows).to_csv(out / "pools.csv", index=False)
    d = pd.DataFrame(rows)
    d.groupby(["model", "task", "K", "N"])[["test", "test_ood"]].agg(["mean", "std", "count"]).round(4).to_csv(out / "summary.csv")


if __name__ == "__main__":
    main()
