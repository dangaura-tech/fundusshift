"""LANTERN on the 50 random 7/7 source splits of experiments.supplement.rotation (split 0 is the published one), same rows, so it pairs with the probes in results/supplement/rotation.csv.
python -m experiments.lantern_rotation --bench data/bench --features runs/features --zeroshot runs/medsiglip --strata data/strata --out results/lantern_rotation.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .calib import Calibrator, ece
from .data import balanced_weights, load_table
from .full_eval import coverage_threshold, quantile
from .probe_calib import train_head
from .shoal import K
from .strata_gate4 import BANKS, nap
from .strata_gate7 import scaled
from .supplement import draw_splits


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "features", "zeroshot", "strata", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--splits", type=int, default=50)
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    keep = t.split.isin(["train", "validation", "calibration", "test"]).to_numpy()
    base_sp, y, src, groups = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.group_id.to_numpy()
    M = np.load(Path(a.strata) / "mean3_medsiglip.npy")[:, 2].astype(np.float32)
    B = [np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32) for b in BANKS]
    f = Path(a.out)
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {r["split"] for r in rows}
    ch = t.split.to_numpy() == "test_ood"
    for k, test in enumerate(draw_splits(t, a.splits)):
        if k in done:
            continue
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
        tr, va, ca, cb = sp == "train", sp == "validation", (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B")
        X = (M - M[tr].mean(0)) / (M[tr].std(0) + 1e-6)
        sites = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
        zs = []
        for o in sites:
            m, vo = tr & (src != o), va & (src != o)
            U = nap(X[m], y[m], src[m], 32)
            Q = X - (X @ U.T) @ U
            h = train_head(Q[m], y[m], balanced_weights(y[m].tolist()), Q[vo], y[vo], "ls", k, epochs=12, bs=64, lr=3e-4, wd=1e-2)
            with torch.no_grad():
                zs.append(scaled(h(torch.tensor(Q)).numpy(), va))
        Z = .5 * np.mean(zs, 0) + .5 * np.mean([scaled(b, va) for b in B], 0)
        P = Calibrator("ts").fit(Z[ca], y[ca])(Z)
        Pr = Calibrator("none").fit(Z[ca], y[ca])(Z)
        pred, conf = P.argmax(1), P.max(1)
        cbi = np.flatnonzero(cb)
        side = np.random.default_rng(k).integers(0, 2, len(cbi)).astype(bool)
        b1, b2 = cbi[side], cbi[~side]
        thr, q = coverage_threshold(conf[b1], .8), quantile(1 - P[b1, y[b1]], .10)

        def read(m):
            ok = conf[m] >= thr
            return float(ok.mean()), float((pred[m][ok] != y[m][ok]).mean()) if ok.any() else np.nan, float(((1 - P[np.flatnonzero(m), y[m]]) <= q).mean())
        mb2 = np.zeros(len(y), bool)
        mb2[b2] = True
        ci, ri, li = read(mb2)
        ce_, re_, le_ = read(te)
        per = [balanced_accuracy_score(y[te & (src == s)], pred[te & (src == s)]) for s in test]
        r = dict(split=k, test_sources="|".join(test), n_heads=len(sites), val_bacc=balanced_accuracy_score(y[va], pred[va]), test_bacc=balanced_accuracy_score(y[te], pred[te]), test_bacc_source_equal=float(np.mean(per)), worst_source=float(np.min(per)), ece_raw=ece(Pr[te], y[te])[0], ece_ts=ece(P[te], y[te])[0], cov_internal=ci, risk_internal=ri, lac_internal=li, cov_test=ce_, risk_test=re_, lac_test=le_, chaksu_bacc=balanced_accuracy_score(y[ch], pred[ch]))
        for c, cn in enumerate(classes):
            r[f"recall_{cn}"] = float((pred[te & (y == c)] == c).mean()) if (te & (y == c)).any() else np.nan
        rows.append(r)
        pd.DataFrame(rows).to_csv(f, index=False)
        print(k, {x: round(r[x], 3) for x in ["test_bacc", "test_bacc_source_equal", "ece_ts", "chaksu_bacc", "recall_G"]}, flush=True)


if __name__ == "__main__":
    main()
