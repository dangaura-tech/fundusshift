"""STRATA gate 6: late fusion of a head and the zero-shot text logits, w * head + (1 - w) * text. Does the site-cancelling loss still matter once text is added?
Leave-one-development-source-out; reads no test row.
python -m experiments.strata_gate6 --bench data/bench --features runs/features --zeroshot runs/medsiglip --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import balanced_weights, load_table
from .probe_calib import train_head
from .shoal import load_x, metrics
from .strata import train_strata
from .strata_gate4 import BANKS, anchor_logits


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
        Tm = np.mean([(lambda Z: (Z - Z.mean(1, keepdims=True)) / Z[va].std())(np.load(Path(a.zeroshot) / f"logits_{a.model}_{b}.npy").astype(np.float32)) for b in BANKS], 0)
        for seed in a.seeds:
            heads = {"ce": train_head(X[tr], y[tr], w, X[va], y[va], "ls", seed),
                     "cross": train_strata(X[tr], y[tr], np.full(tr.sum(), "all"), w, X[va], y[va], .3, seed),
                     "strata": train_strata(X[tr], y[tr], src[tr], w, X[va], y[va], .3, seed)}
            res = {"text": T[held], "text_allbanks": Tm[held]}
            for n, h in heads.items():
                with torch.no_grad():
                    Z = h(torch.tensor(X)).numpy()
                Z = Z - Z.mean(1, keepdims=True)
                Z = Z / Z[va].std()
                res[n] = Z[held]
                for wt in [.3, .5, .7]:
                    res[f"{n}+text_w{wt}"] = wt * Z[held] + (1 - wt) * T[held]
                res[f"{n}+allbanks_w0.5"] = .5 * Z[held] + .5 * Tm[held]
            yy = y[held]
            for arm, z in res.items():
                pred = z.argmax(1)
                rows.append(dict(seed=seed, source=s, arm=arm, **metrics(z, yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan))
            print(s, seed, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(Path(a.out) / "gate6.csv", index=False)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G"]].mean().round(3).to_string())
    print(d.groupby(["arm", "seed"]).bacc.mean().unstack().round(3).to_string())
    k = ["text", "ce+text_w0.5", "cross+text_w0.5", "strata+text_w0.5", "strata"]
    print(d.pivot_table(index="source", columns="arm", values="bacc")[k].round(3).to_string())


if __name__ == "__main__":
    main()
