"""TS, FTS, selective thresholds and conformal sets of the token pipeline, checked on held-out development sites: fitted on calibration rows of the other sites only. No test row is scored.
python -m experiments.last_postprocess --bench data/bench --logits results/strata/last_upgrade --out results/strata/last_postprocess.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .calib import Calibrator, ece, nll
from .data import load_table
from .full_eval import conformal, coverage_threshold, set_metrics, sgr_threshold


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "logits", "out"]:
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    t, _ = load_table(a.bench)
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    rows = []
    for s in sorted(set(src[sp == "train"])):
        held = np.isin(sp, ["train", "validation"]) & (src == s)
        ca, cb = (sp == "calibration") & (sub == "cal_A") & (src != s), (sp == "calibration") & (sub == "cal_B") & (src != s)
        for seed in range(3):
            Z = np.load(Path(a.logits) / f"d384_plain_k32_wd0.01_{s}_s{seed}.npy")
            for ph in ["none", "ts", "fts"]:
                c = Calibrator(ph).fit(Z[ca], y[ca])
                P = c(Z)
                conf, pred = P[held].max(1), P[held].argmax(1)
                yy = y[held]
                r = dict(source=s, seed=seed, posthoc=ph, T=c.T, ece=ece(P[held], yy)[0], nll=nll(P[held], yy))
                pb, ok = P[cb].max(1), P[cb].argmax(1) == y[cb]
                for op, th in [("cov80", coverage_threshold(pb, .8)), ("cov90", coverage_threshold(pb, .9)), ("risk10", sgr_threshold(pb, ok))]:
                    acc = conf >= th
                    r[f"{op}_cov"], r[f"{op}_risk"] = acc.mean(), (pred[acc] != yy[acc]).mean() if acc.any() else np.nan
                    for cl, nm in [(1, "D"), (2, "G")]:
                        r[f"{op}_{nm}toN"] = (acc & (yy == cl) & (pred == 0)).sum() / (yy == cl).sum() if (yy == cl).any() else np.nan
                S = conformal(P[cb], y[cb], P[held], yy, .10)
                for me, q in S.items():
                    m = set_metrics(q, yy)
                    r[f"{me}_cov"], r[f"{me}_size"] = m["coverage"], m["size"]
                rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    c = ["T", "ece", "nll", "cov90_cov", "cov90_risk", "cov80_cov", "cov80_risk", "cov80_GtoN", "cov80_DtoN", "risk10_cov", "risk10_risk", "LAC_cov", "LAC_size", "ClassLAC_cov", "ClassLAC_size", "APS_cov", "APS_size"]
    print(d.groupby("posthoc")[c].mean().round(3).T.to_string())
    print(d[d.posthoc == "ts"].groupby("source")[["ece", "cov90_cov", "risk10_risk", "LAC_cov", "ClassLAC_cov"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
