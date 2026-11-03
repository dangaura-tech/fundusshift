"""Where the leading pipeline fails, by held-out site: confusion, class recall, ceiling under class offsets chosen with labels, and errors shared by every encoder. Development rows only.
python -m experiments.error_analysis --bench data/bench --zeroshot runs/medsiglip --logits results/strata/gate11 --out results/strata/error_analysis.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc

from .strata_gate4 import BANKS
from .strata_gate7 import scaled
from .data import load_table

ENC = ["medsiglip", "medsiglip_last", "retfound_dinov2", "retfound_mae", "dinov2_b14", "siglip2_l16", "biomedclip", "bioclip2_l14", "convnext_t"]


def oracle(z, y, grid=np.linspace(-4, 4, 33)):
    best = (bacc(y, z.argmax(1)), 0., 0.)
    for a in grid:
        for b in grid:
            v = bacc(y, (z + np.array([a, b, 0.])).argmax(1))
            if v > best[0]:
                best = (v, a, b)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "zeroshot", "logits", "out"]:
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    L, rows = Path(a.logits), []
    for s in sorted(set(src[sp == "train"])):
        va, held = (sp == "validation") & (src != s), np.isin(sp, ["train", "validation"]) & (src == s)
        Tm = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32), va) for b in BANKS], 0)[held]
        yy = y[held]
        head = np.mean([np.load(L / f"medsiglip_{s}_s{i}.npy") for i in range(3)], 0)
        z = .5 * head + .5 * Tm
        pred = z.argmax(1)
        wrong = np.stack([(np.mean([np.load(L / f"{e}_{s}_s{i}.npy") for i in range(3)], 0) * .5 + .5 * Tm).argmax(1) != yy for e in ENC])
        orc = oracle(z, yy)
        print(f"\n== {s}  n={len(yy)}  bAcc {bacc(yy, pred):.3f}  head {bacc(yy, head.argmax(1)):.3f}  text {bacc(yy, Tm.argmax(1)):.3f}  oracle {orc[0]:.3f} (N {orc[1]:+.2f}, D {orc[2]:+.2f})")
        print(pd.crosstab(pd.Series(yy, name="true").map(dict(enumerate(classes))), pd.Series(pred, name="pred").map(dict(enumerate(classes)))).to_string())
        r = dict(source=s, n=len(yy), bacc=bacc(yy, pred), head=bacc(yy, head.argmax(1)), text=bacc(yy, Tm.argmax(1)), oracle=orc[0], off_N=orc[1], off_D=orc[2], err=(pred != yy).mean(), shared_all=wrong.all(0).mean(), head_text_disagree=(head.argmax(1) != Tm.argmax(1)).mean())
        for i, c in enumerate(classes):
            m = yy == i
            r[f"n_{c}"], r[f"recall_{c}"] = int(m.sum()), (pred[m] == i).mean() if m.any() else np.nan
            r[f"shared_{c}"] = wrong[:, m].all(0).mean() if m.any() else np.nan
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(d.round(3).T.to_string())
    print(d.mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
