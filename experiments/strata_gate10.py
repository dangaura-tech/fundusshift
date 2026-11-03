"""Gate 10: the leading pipeline on other pooled inputs. The reverse-pyramid idea of RPAttn is kept at low capacity, as a linear head over the mean tokens of three depths.
Inputs: the encoder's pooled output, the mean of last-layer tokens, the means of three depths side by side, and the pooled output beside the last-layer mean. Leave-one-development-source-out; reads no test row.
python -m experiments.strata_gate10 --bench data/bench --features runs/features --zeroshot runs/medsiglip --tokens data/strata --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_table
from .shoal import K, load_x, metrics
from .strata_gate4 import BANKS
from .strata_gate7 import one, scaled


def mean_tokens(path):
    """(N, 3, 384): grid mean per depth, cached. Holds every row; rows are chosen by the caller."""
    f = Path(path) / "mean3_medsiglip.npy"
    if not f.exists():
        S = np.load(Path(path) / "tokens_medsiglip.npy", mmap_mode="r")
        np.save(f, np.concatenate([np.asarray(S[i:i + 512]).astype(np.float32).mean((2, 3)) for i in range(0, len(S), 512)]))
    return np.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--zeroshot", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="medsiglip")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr0 = sp == "train"
    std = lambda v: (v - v[tr0].mean(0)) / (v[tr0].std(0) + 1e-6)
    pooled = load_x(a.features, a.model, t)
    M = mean_tokens(a.tokens)
    last = std(M[:, 2])
    inputs = {"pooled": pooled, "last": last, "depths": std(M.reshape(len(M), -1)), "pooled_last": np.concatenate([pooled, last], 1)}
    rows = []
    for s in sorted(set(src[tr0])):
        tr, va, held = tr0 & (src != s), (sp == "validation") & (src != s), np.isin(sp, ["train", "validation"]) & (src == s)
        Tm = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_{a.model}_{b}.npy").astype(np.float32), va) for b in BANKS], 0)[held]
        others = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
        yy = y[held]
        for seed in a.seeds:
            for name, X in inputs.items():
                single = scaled(one(X, y, src, tr, va, "ce", 0, seed), va)[held]
                bag = np.mean([scaled(one(X, y, src, tr & (src != o), va & (src != o), "ce", 0, seed), va)[held] for o in others], 0)
                bn = np.mean([scaled(one(X, y, src, tr & (src != o), va & (src != o), "ce", 32, seed), va)[held] for o in others], 0)
                for pipe, z in [("single", single), ("bag", bag), ("bag_nap", bn)]:
                    for arm, zz in [(f"{name}_{pipe}", z), (f"{name}_{pipe}+text", .5 * z + .5 * Tm)]:
                        pred = zz.argmax(1)
                        rows.append(dict(seed=seed, source=s, arm=arm, **metrics(zz, yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan, false_G=(pred[yy != 2] == 2).mean() if (yy != 2).any() else np.nan))
            print(s, seed, flush=True)
            pd.DataFrame(rows).to_csv(Path(a.out) / "gate10.csv", index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G", "false_G"]].mean().round(3).sort_values("bacc").to_string())
    print(d[d.arm.str.endswith("bag_nap+text")].pivot_table(index="source", columns="arm", values="bacc").round(3).to_string())
    print(d[d.arm.str.endswith("bag_nap+text")].pivot_table(index="seed", columns="arm", values="bacc").round(3).to_string())


if __name__ == "__main__":
    main()
