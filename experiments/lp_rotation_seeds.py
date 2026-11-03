"""The MedSigLIP linear probe of experiments.supplement.rotation with two more training seeds, on the same 50 splits and the same development re-split, so the
LANTERN comparison has three seeds on both sides. The re-split uses the split index as before; only the head's seed changes.
python -m experiments.lp_rotation_seeds --bench data/bench --features runs/features --out results/supplement/rotation_lp_seeds.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .calib import Calibrator, ece
from .data import balanced_weights, load_table
from .probe_calib import train_head
from .supplement import draw_splits


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "features", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    keep = t.split.isin(["train", "validation", "calibration", "test"]).to_numpy()
    base_sp, y, src, groups = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.group_id.to_numpy()
    F = np.load(Path(a.features) / "medsiglip.npy").astype(np.float32)
    rows = []
    for k, test in enumerate(draw_splits(t, 50)):
        rng = np.random.default_rng(k)
        te = keep & np.isin(src, test)
        dev = np.flatnonzero(keep & ~np.isin(src, test))
        if k == 0:
            sp, sub = base_sp.copy(), t.cal_subset.to_numpy().astype(object).copy()
        else:
            sp, sub = np.full(len(t), "other", dtype=object), np.full(len(t), "", dtype=object)
            sp[te] = "test"
            ug = np.unique(groups[dev])
            u = rng.random(len(ug))
            side = dict(zip(ug, np.where(u < .8, "train", np.where(u < .9, "validation", "calibration"))))
            half = dict(zip(ug, rng.integers(0, 2, len(ug))))
            for i in dev:
                sp[i] = side[groups[i]]
                if sp[i] == "calibration":
                    sub[i] = "cal_A" if half[groups[i]] else "cal_B"
        tr, va, ca = sp == "train", sp == "validation", (sp == "calibration") & (sub == "cal_A")
        X = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-6)
        for seed in a.seeds:
            h = train_head(X[tr], y[tr], balanced_weights(y[tr].tolist()), X[va], y[va], "ce", k + 1000 * seed)
            with torch.no_grad():
                Z = h(torch.tensor(X)).numpy()
            P = Calibrator("ts").fit(Z[ca], y[ca])(Z)
            pred = P.argmax(1)
            per = [balanced_accuracy_score(y[te & (src == s)], pred[te & (src == s)]) for s in test]
            r = dict(split=k, seed=seed, test_bacc=balanced_accuracy_score(y[te], pred[te]), test_bacc_source_equal=float(np.mean(per)), ece_ts=ece(P[te], y[te])[0])
            for c, cn in enumerate(classes):
                r[f"recall_{cn}"] = float((pred[te & (y == c)] == c).mean())
            rows.append(r)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(k, flush=True)


if __name__ == "__main__":
    main()
