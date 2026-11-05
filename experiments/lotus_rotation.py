"""Rotation rows for LOTUS (and the initial LANTERN design) in the format of results/lantern_rotation.csv, read from the seed-0 rotation logits
that experiments.revision_runs saved (<logits>/{no_projection,full}_k{k}.npy). The Cal-B halves are drawn as in experiments.supplement, so
internal and test promises are comparable with the probe rows of results/supplement/rotation.csv.
python -m experiments.lotus_rotation --bench data/bench --logits results/revision/rot_logits --out results/lotus
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .calib import Calibrator, ece
from .data import load_table
from .full_eval import coverage_threshold, quantile


def draw_splits(t, n, seed=0):
    """Same partitions as experiments.supplement.draw_splits (kept here so this script needs no deep-learning stack)."""
    use = t[t.split.isin(["train", "validation", "calibration", "test"])]
    classes_of = use.groupby("source").y.agg(lambda v: set(v)).to_dict()
    sources = sorted(classes_of)
    rng = np.random.default_rng(seed)
    fixed = tuple(sorted(t[t.split == "test"].source.unique()))
    out, seen = [fixed], {fixed}
    while len(out) < n + 1:
        test = tuple(sorted(rng.choice(sources, 7, replace=False).tolist()))
        dev = [s for s in sources if s not in test]
        if test in seen or set().union(*[classes_of[s] for s in dev]) != {0, 1, 2} or set().union(*[classes_of[s] for s in test]) != {0, 1, 2}:
            continue
        seen.add(test)
        out.append(test)
    return out


def assign(t, k, test, rng_seed):
    """Same split labels as experiments.revision_runs.assign: k = 0 is the published split, otherwise an 80/10/10 group split of the development side."""
    keep = t.split.isin(["train", "validation", "calibration", "test"]).to_numpy()
    src, groups = t.source.to_numpy(), t.group_id.to_numpy()
    te = keep & np.isin(src, test)
    if k == 0:
        return t.split.to_numpy().copy(), t.cal_subset.to_numpy().astype(object).copy(), te
    rng = np.random.default_rng(rng_seed)
    dv = np.flatnonzero(keep & ~np.isin(src, test))
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
    return sp, sub, te


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "logits", "out"]:
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    y, src = t.y.to_numpy(), t.source.to_numpy()
    ch = t.split.to_numpy() == "test_ood"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = draw_splits(t, 50)
    for arm, name in [("no_projection", "lotus"), ("full", "lantern")]:
        rows = []
        for k in range(1, len(splits)):
            test = splits[k]
            sp, sub, te = assign(t, k, test, k)
            va, ca, cb = sp == "validation", (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B")
            Z = np.load(Path(a.logits) / f"{arm}_k{k}.npy").astype(np.float32)
            P, Pr = Calibrator("ts").fit(Z[ca], y[ca])(Z), Calibrator("none").fit(Z[ca], y[ca])(Z)
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
            r = dict(split=k, test_sources="|".join(test), val_bacc=balanced_accuracy_score(y[va], pred[va]), test_bacc=balanced_accuracy_score(y[te], pred[te]), test_bacc_source_equal=float(np.mean(per)), worst_source=float(np.min(per)),
                     ece_raw=ece(Pr[te], y[te])[0], ece_ts=ece(P[te], y[te])[0], cov_internal=ci, risk_internal=ri, lac_internal=li, cov_test=ce_, risk_test=re_, lac_test=le_, chaksu_bacc=balanced_accuracy_score(y[ch], pred[ch]))
            for c, cn in enumerate(classes):
                r[f"recall_{cn}"] = float((pred[te & (y == c)] == c).mean()) if (te & (y == c)).any() else np.nan
            rows.append(r)
        pd.DataFrame(rows).to_csv(out / f"rotation_{name}.csv", index=False)
        d = pd.DataFrame(rows)
        print(name, d[["test_bacc", "ece_ts", "risk_internal", "risk_test", "lac_internal", "lac_test", "cov_internal", "cov_test"]].mean().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main()
