"""Calibrated normal certificate for LANTERN, after the healthy certificate of MaizeTrace (panel d): temperature and thresholds come from a leave-one-source-out (LOSO) reference
instead of the in-distribution Cal split. Each development row is scored by a LANTERN fitted without its source, so the reference has already met a source shift.
Rules, all fitted on the reference: T by NLL; tau_c, the confidence giving coverage c; tau_N, the smallest p(N) at which disease images answered "normal" stay at most r of all disease images;
an image is answered iff conf >= tau_c, and a normal answer also needs p(N) >= tau_N. LAC quantile likewise. The Cal-B arm keeps the current protocol.
python -m experiments.certify --bench data/bench --strata data/strata --zeroshot runs/medsiglip --out results/certify --worker 0 --workers 1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .calib import Calibrator, ece
from .full_eval import coverage_threshold, quantile, sgr_threshold
from .scope import N, load, pipeline
from .supplement import draw_splits

COVS, FR = (.8, .9), .02


def tau_normal(P, y, r):
    """Smallest threshold on p(N) such that disease images predicted N with p(N) >= it are at most r of disease images."""
    dis = y != N
    s = np.sort(P[dis & (P.argmax(1) == N), N])[::-1]
    k = int(np.floor(r * dis.sum()))
    return float(s[k]) + 1e-9 if k < len(s) else 0.


def decide(P, y, tc, tn):
    conf, pred = P.max(1), P.argmax(1)
    acc = (conf >= tc) & ((pred != N) | (P[:, N] >= tn))
    dis = y != N
    return dict(cov=float(acc.mean()), risk=float((pred[acc] != y[acc]).mean()) if acc.any() else np.nan,
                fr=float((acc & dis & (pred == N)).sum() / max(dis.sum(), 1)), GtoN=float((acc & (y == 2) & (pred == N)).sum() / max((y == 2).sum(), 1)))


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
    f = out / f"rot2_w{a.worker}.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {r["split"] for r in rows}
    for k, test in enumerate(draw_splits(t, 50)):
        if k % a.workers != a.worker or k in done:
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
        tr, va, ca, cb = sp == "train", sp == "validation", (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B")
        X = (M - M[tr].mean(0)) / (M[tr].std(0) + 1e-6)
        Z = pipeline(X, y, scope, src, tr, va, Tb, "exact", k)
        # LOSO reference: rows of each development source scored by a LANTERN fitted without it
        R, Ry = [], []
        for o in sorted(set(src[tr])):
            tro, vao = tr & (src != o), va & (src != o)
            if len(np.unique(y[tro])) < 3:
                continue
            Zo = pipeline(X, y, scope, src, tro, vao, Tb, "exact", k)
            m = np.isin(sp, ["train", "validation", "calibration"]) & (src == o)
            R.append(Zo[m])
            Ry.append(y[m])
        R, Ry = np.concatenate(R), np.concatenate(Ry)
        r = dict(split=k, test_sources="|".join(test), n_ref=len(Ry))
        for arm, (Zf, yf) in {"calB": (None, None), "loso": (R, Ry)}.items():
            c = Calibrator("ts").fit(Z[ca], y[ca]) if arm == "calB" else Calibrator("ts").fit(Zf, yf)
            P = c(Z)
            Pf, yfit = (P[cb], y[cb]) if arm == "calB" else (c(Zf), yf)
            r[f"{arm}_T"] = c.T
            r[f"{arm}_ece_test"] = ece(P[te], y[te])[0]
            tn = tau_normal(Pf, yfit, FR)
            q = quantile(1 - Pf[np.arange(len(yfit)), yfit], .10)
            r[f"{arm}_lac_test"] = float(((1 - P[np.flatnonzero(te), y[te]]) <= q).mean())
            r[f"{arm}_lac_chaksu"] = float(((1 - P[np.flatnonzero(ch), y[ch]]) <= q).mean())
            for cv in COVS:
                tc = coverage_threshold(Pf.max(1), cv)
                for cert in [False, True]:
                    nm = f"{arm}_c{int(cv * 100)}{'_cert' if cert else ''}"
                    for view, m in [("test", te), ("chaksu", ch)]:
                        for kk, vv in decide(P[m], y[m], tc, tn if cert else 0.).items():
                            r[f"{nm}_{view}_{kk}"] = vv
                    r[f"{nm}_fit_cov"] = decide(Pf, yfit, tc, tn if cert else 0.)["cov"]
            # risk target: SGR bound of 10% (delta 0.05) on the fitting set; reported with and without the normal certificate
            ok = Pf.argmax(1) == yfit
            ts = sgr_threshold(Pf.max(1), ok, risk=.10)
            r[f"{arm}_sgr_thr"] = ts
            for cert in [False, True]:
                nm = f"{arm}_r10{'_cert' if cert else ''}"
                for view, m in [("test", te), ("chaksu", ch)]:
                    for kk, vv in decide(P[m], y[m], ts, tn if cert else 0.).items():
                        r[f"{nm}_{view}_{kk}"] = vv
            # risk at matched coverage: the same model, so this isolates threshold placement from ranking
            for cv in (.5, .7, .9):
                conf_t = P[te].max(1)
                k_ = max(int(round(cv * te.sum())), 1)
                o = np.argsort(-conf_t)[:k_]
                r[f"{arm}_matched{int(cv * 100)}_risk"] = float((P[te].argmax(1)[o] != y[te][o]).mean())
        rows.append(r)
        pd.DataFrame(rows).to_csv(f, index=False)
        print(k, {x: round(r[x], 3) for x in ["calB_c80_test_cov", "loso_c80_test_cov", "calB_c80_test_risk", "loso_c80_test_risk", "loso_c80_cert_test_fr", "calB_c80_test_fr"]}, flush=True)


if __name__ == "__main__":
    main()
