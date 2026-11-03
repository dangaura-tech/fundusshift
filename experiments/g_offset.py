"""The leading pipeline's main failure is N read as G on sites with no glaucoma. Class offsets fixed for all images, chosen on the other development sites only, so the held-out site's labels play no part. Development rows only.
python -m experiments.g_offset --bench data/bench --zeroshot runs/medsiglip --logits results/strata/gate11
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc

from .data import load_table
from .shoal import metrics
from .strata_gate4 import BANKS
from .strata_gate7 import scaled


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "zeroshot", "logits"]:
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    t, _ = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    sites = sorted(set(src[sp == "train"]))
    L, rows, G = Path(a.logits), [], np.linspace(-2, 1, 13)
    for seed in range(3):
        Z, Y = {}, {}
        for s in sites:
            va, held = (sp == "validation") & (src != s), np.isin(sp, ["train", "validation"]) & (src == s)
            Tm = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32), va) for b in BANKS], 0)[held]
            Z[s], Y[s] = .5 * np.load(L / f"medsiglip_{s}_s{seed}.npy") + .5 * Tm, y[held]
        table = {(d, g): {s: bacc(Y[s], (Z[s] + np.array([0, d, g])).argmax(1)) for s in sites} for d in np.linspace(-1, 1, 9) for g in G}
        for s in sites:
            d, g = max(table, key=lambda k: np.mean([table[k][o] for o in sites if o != s]))
            for arm, off in [("leader", (0, 0)), ("offset_nested", (d, g))]:
                z = Z[s] + np.array([0, off[0], off[1]])
                pred, yy = z.argmax(1), Y[s]
                rows.append(dict(seed=seed, source=s, arm=arm, d=off[0], g=off[1], **metrics(z.astype(np.float32), yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan, false_G=(pred[yy != 2] == 2).mean() if (yy != 2).any() else np.nan))
    r = pd.DataFrame(rows)
    print(r.groupby("arm")[["bacc", "ece", "nll", "recall_G", "false_G"]].mean().round(3).to_string())
    print(r.pivot_table(index="source", columns="arm", values="bacc").round(3).to_string())
    print(r[r.arm == "offset_nested"].groupby("source")[["d", "g", "recall_G", "false_G"]].mean().round(2).to_string())
    print(r.groupby(["arm", "seed"]).bacc.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
