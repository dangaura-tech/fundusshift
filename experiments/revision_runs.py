"""Revision runs, one script for every experiment that needs the HPC features (CPU only, from saved features).

rotation  Full factorial LANTERN ablation (bag x projection x text, plus text only) on the published split (k = 0) and the 50 source rotations,
          seeds 0-2. One row per (split, seed, arm) with pooled and per-source metrics, macro-F1, AUROC, NLL, Brier, AURC, matched-coverage risk
          and the Cal-B promises (internal against test). Saves published-split logits for every arm and seed, and float16 rotation logits for
          the full and projection-free arms (seed 0), for per-source reliability diagrams and risk-coverage intervals.
region    Leave-one-region-out: each region's non-OOD sources form the test side; full LANTERN, projection-free LANTERN and the class-weighted
          MedSigLIP linear probe, seeds 0-2.
certify   The leave-one-source-out reference of experiments.certify, for the projection-free arm.

python -m experiments.revision_runs rotation --bench data/bench --strata data/strata --zeroshot runs/medsiglip --out results/revision --worker 0 --workers 1
python -m experiments.revision_runs region --bench data/bench --strata data/strata --zeroshot runs/medsiglip --features runs/features --out results/revision
python -m experiments.revision_runs certify --bench data/bench --strata data/strata --zeroshot runs/medsiglip --out results/revision --worker 0 --workers 1
"""
import argparse
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .calib import Calibrator, ece
from .data import balanced_weights
from .full_eval import coverage_threshold, quantile
from .probe_calib import train_head
from .scope import load, pipeline
from .supplement import draw_splits

ARMS = {
    "full": {},                                                    # bag + projection + text
    "no_projection": dict(proj=False),                             # bag + text
    "no_bag": dict(bag=False),                                     # single + projection + text
    "no_text": dict(text=False),                                   # bag + projection
    "bag_only": dict(proj=False, text=False),                      # bag
    "proj_only": dict(bag=False, text=False),                      # single + projection
    "single_text": dict(bag=False, proj=False),                    # single + text
    "single_head": dict(bag=False, proj=False, text=False),        # single
    "text_only": dict(head=False),                                 # text
}
SAVE_ROT = ("full", "no_projection")
K = 3


def assign(t, k, test, rng_seed):
    """Split labels for one partition: k = 0 is the published split, otherwise test sources vs an 80/10/10 group split of the rest."""
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


def aurc(P, y):
    conf, ok = P.max(1), P.argmax(1) == y
    err = ~ok[np.argsort(-conf)]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def auroc(P, y):
    cls = np.unique(y)
    try:
        if len(cls) == K:
            return float(roc_auc_score(y, P, multi_class="ovr", labels=list(range(K))))
        if len(cls) == 2:
            return float(roc_auc_score(y == cls[1], P[:, cls[1]] / (P[:, cls[0]] + P[:, cls[1]])))
    except ValueError:
        pass
    return np.nan


def evaluate(Z, y, src, sp, sub, te, ch, test):
    """Every metric of one fitted model: TS on Cal-A, thresholds and quantiles on half of Cal-B, internal estimates on the other half."""
    ca, cb, va = (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B"), sp == "validation"
    cal = Calibrator("ts").fit(Z[ca], y[ca])
    P, Pr = cal(Z), Calibrator("none").fit(Z[ca], y[ca])(Z)
    pred, conf = P.argmax(1), P.max(1)
    cbi = np.flatnonzero(cb)
    half = np.random.default_rng(0).integers(0, 2, len(cbi)).astype(bool)
    b1, b2 = cbi[half], cbi[~half]
    thr, q = coverage_threshold(conf[b1], .8), quantile(1 - P[b1, y[b1]], .10)

    def promise(m):
        ok = conf[m] >= thr
        return float(ok.mean()), float((pred[m][ok] != y[m][ok]).mean()) if ok.any() else np.nan, float(((1 - P[np.flatnonzero(m), y[m]]) <= q).mean())
    mb2 = np.zeros(len(y), bool)
    mb2[b2] = True
    ci, ri, li = promise(mb2)
    cte, rte, lte = promise(te)
    yt, Pt = y[te], P[te]
    r = dict(T=cal.T, val_bacc=balanced_accuracy_score(y[va], pred[va]), test_bacc=balanced_accuracy_score(yt, Pt.argmax(1)),
             test_macro_f1=f1_score(yt, Pt.argmax(1), average="macro", labels=sorted(np.unique(yt))), test_auroc=auroc(Pt, yt),
             ece_raw=ece(Pr[te], yt)[0], ece_ts=ece(Pt, yt)[0], ace_ts=ece(Pt, yt, adaptive=True)[0], nll_ts=float(-np.log(Pt[np.arange(len(yt)), yt].clip(1e-12)).mean()),
             brier_ts=float(((Pt - np.eye(K)[yt]) ** 2).sum(1).mean()), aurc_ts=aurc(Pt, yt),
             cov_internal=ci, risk_internal=ri, lac_internal=li, cov_test=cte, risk_test=rte, lac_test=lte,
             false_G=float((Pt.argmax(1)[yt != 2] == 2).mean()), GtoN=float((Pt.argmax(1)[yt == 2] == 0).mean()) if (yt == 2).any() else np.nan,
             DtoN=float((Pt.argmax(1)[yt == 1] == 0).mean()) if (yt == 1).any() else np.nan,
             chaksu_bacc=balanced_accuracy_score(y[ch], pred[ch]), chaksu_recall_G=float((pred[ch & (y == 2)] == 2).mean()), chaksu_ece_ts=ece(P[ch], y[ch])[0],
             chaksu_auroc=auroc(P[ch], y[ch]))
    for c, cn in enumerate("NDG"):
        r[f"recall_{cn}"] = float((Pt.argmax(1)[yt == c] == c).mean()) if (yt == c).any() else np.nan
    o = np.argsort(-Pt.max(1))
    for cv in (.5, .7, .9):
        n_ = max(int(round(cv * len(yt))), 1)
        r[f"matched{int(cv * 100)}_risk"] = float((Pt.argmax(1)[o[:n_]] != yt[o[:n_]]).mean())
    per = []
    for s in test:
        m = te & (src == s)
        b = balanced_accuracy_score(y[m], pred[m])
        per.append(b)
        r[f"src_bacc:{s}"], r[f"src_ece:{s}"], r[f"src_n:{s}"] = b, ece(P[m], y[m])[0], int(m.sum())
    r["test_bacc_source_equal"], r["worst_source"] = float(np.mean(per)), float(np.min(per))
    return r


def rotation(a):
    t, M, Tb = load(a)
    y, src, scope = t.y.to_numpy(), t.source.to_numpy(), t.normal_scope.to_numpy()
    ch = t.split.to_numpy() == "test_ood"
    out = Path(a.out)
    (out / "rotation").mkdir(parents=True, exist_ok=True)
    (out / "final").mkdir(parents=True, exist_ok=True)
    (out / "rot_logits").mkdir(parents=True, exist_ok=True)
    f = out / "rotation" / f"w{a.worker}.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["split"], r["seed"], r["arm"]) for r in rows}
    splits = draw_splits(t, 50)
    tasks = [(k, sd, arm) for k in range(len(splits)) for sd in a.seeds for arm in a.arms]
    for i, (k, sd, arm) in enumerate(tasks):
        if i % a.workers != a.worker or (k, sd, arm) in done:
            continue
        test = splits[k]
        sp, sub, te = assign(t, k, test, k)
        tr, va = sp == "train", sp == "validation"
        X = (M - M[tr].mean(0)) / (M[tr].std(0) + 1e-6)
        Z = pipeline(X, y, scope, src, tr, va, Tb, "exact", k + 1000 * sd, **ARMS[arm]).astype(np.float32)
        if k == 0:
            np.save(out / "final" / f"{arm}_s{sd}.npy", Z)
        elif sd == 0 and arm in SAVE_ROT:
            np.save(out / "rot_logits" / f"{arm}_k{k}.npy", Z.astype(np.float16))
        r = dict(split=k, seed=sd, arm=arm, test_sources="|".join(test), **evaluate(Z, y, src, sp, sub, te, ch, test))
        rows.append(r)
        pd.DataFrame(rows).to_csv(f, index=False)
        print(k, sd, arm, round(r["test_bacc"], 3), round(r["ece_ts"], 3), flush=True)


def region(a):
    t, M, Tb = load(a)
    y, src, scope, reg = t.y.to_numpy(), t.source.to_numpy(), t.normal_scope.to_numpy(), t.region.to_numpy()
    ch = t.split.to_numpy() == "test_ood"
    F = np.load(Path(a.features) / "medsiglip.npy").astype(np.float32)
    use = t.split.isin(["train", "validation", "calibration", "test"]).to_numpy()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "region.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["region"], r["seed"], r["arm"]) for r in rows}
    for j, R in enumerate(sorted(set(reg[use]))):
        test = sorted(set(src[use & (reg == R)]))
        dev = use & ~np.isin(src, test)
        if len(np.unique(y[dev])) < K:
            print("skip", R, "development side lacks a class", flush=True)
            continue
        sp, sub, te = assign(t, 1, test, 100 + j)
        tr, va = sp == "train", sp == "validation"
        X = (M - M[tr].mean(0)) / (M[tr].std(0) + 1e-6)
        Xf = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-6)
        for sd in a.seeds:
            for arm in ["full", "no_projection", "medsiglip_lp"]:
                if (R, sd, arm) in done:
                    continue
                if arm == "medsiglip_lp":
                    head = train_head(Xf[tr], y[tr], balanced_weights(y[tr].tolist()), Xf[va], y[va], "ce", 100 + j + 1000 * sd)
                    with torch.no_grad():
                        Z = head(torch.tensor(Xf)).numpy()
                else:
                    Z = pipeline(X, y, scope, src, tr, va, Tb, "exact", 100 + j + 1000 * sd, **ARMS[arm])
                r = dict(region=R, seed=sd, arm=arm, test_sources="|".join(test), n_test=int(te.sum()), **evaluate(Z.astype(np.float32), y, src, sp, sub, te, ch, test))
                rows.append(r)
                pd.DataFrame(rows).to_csv(f, index=False)
                print(R, sd, arm, round(r["test_bacc"], 3), round(r["ece_ts"], 3), flush=True)


def certify(a):
    from . import certify as C
    C.pipeline = functools.partial(pipeline, proj=False)   # every LANTERN fit inside certify, including the LOSO reference, drops the projection
    C.main(["--bench", a.bench, "--strata", a.strata, "--zeroshot", a.zeroshot, "--out", str(Path(a.out) / "certify_no_projection"), "--worker", str(a.worker), "--workers", str(a.workers)])


def main(argv=None):
    torch.set_num_threads(int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', 2)))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rotation", "region", "certify"])
    for k in ["bench", "strata", "zeroshot", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--features", default="runs/features")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    a = ap.parse_args(argv)
    {"rotation": rotation, "region": region, "certify": certify}[a.cmd](a)


if __name__ == "__main__":
    main()
