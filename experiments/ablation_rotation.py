"""LANTERN ablation on the 50 source rotations of experiments.supplement (split 0 is the published one). Each arm removes one component; same splits and recipe as experiments.lantern_rotation.
python -m experiments.ablation_rotation --bench data/bench --strata data/strata --zeroshot runs/medsiglip --out results/ablation_rotation --worker 0 --workers 1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .calib import Calibrator, ece
from .scope import load, pipeline
from .supplement import draw_splits

ARMS = {"full": {}, "no_bag": dict(bag=False), "no_projection": dict(proj=False), "no_text": dict(text=False), "single_head": dict(bag=False, proj=False, text=False), "text_only": dict(head=False)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "strata", "zeroshot", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    t, M, Tb = load(a)
    keep = t.split.isin(["train", "validation", "calibration", "test"]).to_numpy()
    base_sp, y, src, groups, scope = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.group_id.to_numpy(), t.normal_scope.to_numpy()
    ch = base_sp == "test_ood"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"w{a.worker}.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["split"], r["arm"]) for r in rows}
    for k, test in enumerate(draw_splits(t, 50)):
        if k % a.workers != a.worker:
            continue
        rng = np.random.default_rng(k)
        te = keep & np.isin(src, test)
        dv = np.flatnonzero(keep & ~np.isin(src, test))
        if k == 0:
            sp, sub = base_sp.copy(), t.cal_subset.to_numpy().astype(object).copy()
        else:
            sp, sub = np.full(len(t), "other", dtype=object), np.full(len(t), "", dtype=object)
            sp[te] = "test"
            ug = np.unique(groups[dv])
            u = rng.random(len(ug))
            side = dict(zip(ug, np.where(u < .8, "train", np.where(u < .9, "validation", "calibration"))))
            half = dict(zip(ug, rng.integers(0, 2, len(ug))))
            for i in dv:
                sp[i] = side[groups[i]]
                if sp[i] == "calibration":
                    sub[i] = "cal_A" if half[groups[i]] else "cal_B"
        tr, va, ca = sp == "train", sp == "validation", (sp == "calibration") & (sub == "cal_A")
        X = (M - M[tr].mean(0)) / (M[tr].std(0) + 1e-6)
        for arm, kw in ARMS.items():
            if (k, arm) in done:
                continue
            Z = pipeline(X, y, scope, src, tr, va, Tb, "exact", k, **kw)
            P = Calibrator("ts").fit(Z[ca], y[ca])(Z)
            pred = P.argmax(1)
            per = [balanced_accuracy_score(y[te & (src == s)], pred[te & (src == s)]) for s in test]
            r = dict(split=k, arm=arm, test_bacc=balanced_accuracy_score(y[te], pred[te]), test_bacc_source_equal=float(np.mean(per)), ece_ts=ece(P[te], y[te])[0], chaksu_bacc=balanced_accuracy_score(y[ch], pred[ch]))
            for c, cn in enumerate(["N", "D", "G"]):
                r[f"recall_{cn}"] = float((pred[te & (y == c)] == c).mean()) if (te & (y == c)).any() else np.nan
            r["false_G"] = float((pred[te & (y != 2)] == 2).mean())
            rows.append(r)
            pd.DataFrame(rows).to_csv(f, index=False)
        print(k, flush=True)


if __name__ == "__main__":
    main()
