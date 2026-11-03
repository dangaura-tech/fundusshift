"""One evaluator over every saved logits file: views, frozen thresholds, false reassurance, conformal sets, calibration budget, OOD scores.

Protocol (fixed before reading any output): temperature on Cal-A; thresholds and conformal quantiles on Cal-B; test only reported.
Raw (none), TS and FTS go through identical code: every view, metric, threshold, matched coverage and conformal table exists for all three.
An ensemble averages the seeds' probabilities of that same post-hoc state. Thresholds and quantiles are refit on Cal-B per state.
Calibration metrics in every view: ECE-15, MCE-15 (Naeini et al. 2015), ACE-15 (Nixon et al. 2019, classwise equal-count ranges),
top-label adaptive ECE, classwise ECE, NLL, Brier, AURC, E-AURC, selective risk at 80 / 90 % coverage.
Views: pooled test, difficulty-weighted test, equal-source mean, each held-out source, Chákṣu (test_ood).
Difficulty weights: each class gets 1/3; within a class each (source, class) cell is weighted by the mean error of a fixed panel of
six CE linear probes (dinov2_b14, resnet50, clip_b16, convnext_t, vit_b16_in21k, siglip2_b16; 3 seeds). They use test labels, so the view is exploratory.
Thresholds: cov80 / cov90 = smallest MSP threshold with Cal-B coverage >= c; risk10 = SGR (Geifman & El-Yaniv 2017), one-sided
Clopper-Pearson bound <= 10 %, delta = 0.05 split over the binary-search steps; infeasible = accepts nothing.
False reassurance: accepted images of a disease class predicted normal, count and rate over all images of that class.
Conformal: LAC, APS (top class always in), class-conditional LAC; alpha 0.10 and 0.05; finite-sample quantile on Cal-B.
python -m experiments.full_eval --bench data/bench --runs runs --features runs/features --out results/full_eval
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.covariance import LedoitWolf

from .calib import POSTHOC, Calibrator, classwise_ece, ece, nll, selective, softmax
from .data import load_table

PANEL = ["dinov2_b14", "resnet50", "clip_b16", "convnext_t", "vit_b16_in21k", "siglip2_b16"]
SEEDS = re.compile(r"_s(\d+)$")


def discover(runs):
    """Yields (setting, model, config, seed, path) for every saved logits file."""
    runs = Path(runs)
    out = []
    for d in ["zeroshot", "flair", "medsiglip"]:
        for f in sorted((runs / d).glob("logits_*.npy")):
            for b in ["clip_ensemble", "cupl", "descriptive_single"]:
                if f.stem.endswith("_" + b):
                    out.append(("zero-shot", f.stem[7:-len(b) - 1], b, 0, f))
    for f in sorted((runs / "coop").glob("logits_*_s?.npy")):
        out.append(("coop", f.stem[7:-3], "ce", int(f.stem[-1]), f))
    for d, setting in [("coop_ablation", None), ("lp16", "16-shot-probe"), ("linear_probe", "linear-probe"), ("mlp_probe", "mlp-probe"), ("tipadapter", None), ("tipadapter_bacc", None)]:
        for f in sorted((runs / d).glob("logits_*_s?.npy")) if (runs / d).exists() else []:
            st, seed = f.stem[7:-3], int(f.stem[-1])
            if d in ("coop_ablation", "tipadapter", "tipadapter_bacc"):
                arm, rest = st.split("_", 1)
                model, _, cfg = (rest.rpartition("_") if d == "coop_ablation" else (rest, "", ""))
                if d == "coop_ablation" and rest.split("_")[-2:] [0] == "zs":
                    model, cfg = rest.rsplit("_", 2)[0], "_".join(rest.split("_")[-2:])
                if d == "coop_ablation" and arm == "coop" and cfg == "ce":  # the same run as runs/coop
                    continue
                out.append((f"{arm}-acc" if d == "tipadapter" else arm, model, cfg, seed, f))  # benchmark rule = bAcc; official accuracy rule kept as sensitivity
            else:
                m = re.match(r"(.+)_(none|class|class_source)_([a-z]+)$", st)
                out.append((setting, m[1], f"{m[2]}/{m[3]}", seed, f))
    for f in sorted((runs / "finetune").glob("*/logits.npy")) if (runs / "finetune").exists() else []:
        m = re.match(r"(.+)_([a-z]+)_s(\d+)$", f.parent.name)
        out.append(("finetune", m[1], m[2], int(m[3]), f))
    for f in sorted((runs / "finetune_nll").glob("*/logits.npy")) if (runs / "finetune_nll").exists() else []:
        m = re.match(r"(.+)_([a-z]+)_s(\d+)$", f.parent.name)
        out.append(("finetune-nll", m[1], m[2], int(m[3]), f))
    for f in sorted((runs / "robust").glob("*/logits.npy")) if (runs / "robust").exists() else []:
        m = re.match(r"(.+?)_(erm|groupdro|coral)(_swad)?_s(\d+)$", f.parent.name)
        out.append(("swad" if m[3] else m[2], m[1], "ce", int(m[4]), f))
    for f in sorted((runs / "wiseft").glob("*/logits_*.npy")) if (runs / "wiseft").exists() else []:
        m = re.match(r"(.+)_s(\d+)$", f.parent.name)
        out.append(("wise-ft", m[1], f.stem[7:], int(m[2]), f))
    for f in sorted((runs / "vlm").glob("logits_*.npy")) if (runs / "vlm").exists() else []:
        if np.isnan(np.load(f)).any():  # checkpoint of a model still running
            print(f"skip unfinished {f.name}")
            continue
        out.append(("mcq", f.stem[7:], "avg6", 0, f))
    return out


def load_logits(path, setting, t):
    """Logits aligned to the full table; rows the predictor never scored are NaN."""
    Z = np.load(path).astype(np.float32)
    if setting == "mcq":
        m = t.split.isin(["calibration", "test", "test_ood", "test_openset"]).to_numpy()
        full = np.full((len(t), 3), np.nan, np.float32)
        full[m] = Z.mean(1)
        return full
    return Z


def difficulty_weights(t, runs, out):
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    te = sp == "test"
    err = np.zeros(te.sum())
    n = 0
    for m in PANEL:
        for s in range(3):
            f = Path(runs) / "linear_probe" / f"logits_{m}_none_ce_s{s}.npy"
            if f.exists():
                err += (np.load(f)[te].argmax(1) != y[te])
                n += 1
    if not n:
        raise SystemExit("difficulty panel logits are missing")
    err /= n
    d = pd.DataFrame({"source": src[te], "y": y[te], "err": err})
    cell = d.groupby(["y", "source"]).agg(err=("err", "mean"), n=("err", "size")).reset_index()
    cell["weight"] = cell.groupby("y").err.transform(lambda e: e / e.sum() / 3)
    cell.to_csv(Path(out) / "difficulty_weights.csv", index=False)
    key = {(a, b): w / c for a, b, w, c in zip(cell.y, cell.source, cell.weight, cell.n)}
    w = np.zeros(len(t))
    w[te] = [key[(a, b)] for a, b in zip(y[te], src[te])]
    return w


def ace_classwise(p, y, w=None, bins=15):
    """Nixon et al. 2019: per class, images sorted by p_k and cut into equal-count (or equal-weight) ranges; mean |frequency - mean p_k|."""
    w = np.ones(len(y)) if w is None else w
    gaps = []
    for k in range(p.shape[1]):
        o = np.argsort(p[:, k], kind="stable")
        c = np.cumsum(w[o]) / w.sum()
        cut = np.minimum((c * bins - 1e-9).astype(int), bins - 1)
        for b in range(bins):
            m = o[cut == b]
            if len(m):
                gaps.append(abs(np.average(y[m] == k, weights=w[m]) - np.average(p[m, k], weights=w[m])))
    return float(np.mean(gaps))


def weighted(p, y, w):
    w = w / w.sum()
    pred, conf = p.argmax(1), p.max(1)
    ece_, mce_ = 0.0, 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            g = abs((w[m] * (pred[m] == y[m])).sum() - (w[m] * conf[m]).sum())
            ece_ += g
            mce_ = max(mce_, g / w[m].sum())
    return {"w_acc": float((w * (pred == y)).sum()), "w_nll": float(-(w * np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))).sum()),
            "w_brier": float((w * ((p - np.eye(p.shape[1])[y]) ** 2).sum(1)).sum()), "w_ece": float(ece_), "w_mce": float(mce_), "w_ace": ace_classwise(p, y, w)}


def view_metrics(p, y):
    pred = p.argmax(1)
    present = sorted(set(y))
    r = {"n": len(y), "acc": float((pred == y).mean()), "bacc": float(balanced_accuracy_score(y, pred)),
         "macro_f1": float(f1_score(y, pred, average="macro", labels=present, zero_division=0)), "nll": nll(p, y),
         "brier": float(((p - np.eye(3)[y]) ** 2).sum(1).mean())}
    r["ece"], r["mce"] = ece(p, y)
    r["ace"] = ace_classwise(p, y)
    r["ace_top"] = ece(p, y, adaptive=True)[0]
    r["cw_ece"] = classwise_ece(p, y)
    r.update(selective(p, y))
    for c in present:
        r[f"recall_{c}"] = float((pred[y == c] == c).mean())
    if len(present) == 3:
        r["auroc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    return r


def coverage_threshold(conf, c):
    return float(np.sort(conf)[::-1][max(int(np.ceil(c * len(conf))) - 1, 0)])


def sgr_threshold(conf, correct, risk=0.10, delta=0.05):
    """Geifman & El-Yaniv 2017 binary search; returns +inf when no threshold is feasible."""
    order = np.argsort(-conf)
    err = ~correct[order]
    k = int(np.ceil(np.log2(len(conf))))
    lo, hi, best = 0, len(conf) - 1, np.inf
    for _ in range(k):
        mid = (lo + hi + 1) // 2
        m, e = mid + 1, int(err[:mid + 1].sum())
        bound = 1.0 if e == m else beta_dist.ppf(1 - delta / k, e + 1, m - e)
        if bound <= risk:
            best, lo = float(conf[order][mid]), mid
        else:
            hi = mid - 1
        if lo >= hi:
            break
    return best


def operating(p, y, thr, classes):
    conf, pred = p.max(1), p.argmax(1)
    acc = conf >= thr
    N = classes.index("N")
    r = {"coverage": float(acc.mean()), "risk": float((pred[acc] != y[acc]).mean()) if acc.any() else np.nan}
    for c in range(3):
        if c != N and (y == c).any():
            bad = acc & (y == c) & (pred == N)
            r[f"{classes[c]}_to_N"] = int(bad.sum())
            r[f"{classes[c]}_to_N_rate"] = float(bad.sum() / (y == c).sum())
    return r


def matched(p, y, frac, classes):
    k = max(int(round(frac * len(y))), 1)
    idx = np.argsort(-p.max(1), kind="stable")[:k]
    pred = p[idx].argmax(1)
    N = classes.index("N")
    r = {"sel_risk": float((pred != y[idx]).mean())}
    for c in range(3):
        if c != N and (y == c).any():
            r[f"{classes[c]}_to_N_rate"] = float(((pred == N) & (y[idx] == c)).sum() / (y == c).sum())
    return r


def quantile(s, a):
    return float(np.sort(s)[min(int(np.ceil((len(s) + 1) * (1 - a))) - 1, len(s) - 1)])


def lac_sets(p, q):
    return (1 - p) <= q


def aps_scores(p):
    o = np.argsort(-p, 1)
    c = np.cumsum(np.take_along_axis(p, o, 1), 1)
    s = np.empty_like(p)
    np.put_along_axis(s, o, c, 1)
    return s


def conformal(pb, yb, p, y, a):
    q = quantile(1 - pb[np.arange(len(yb)), yb], a)
    out = {"LAC": lac_sets(p, q)}
    sb = aps_scores(pb)[np.arange(len(yb)), yb]
    qa = quantile(sb, a)
    so = aps_scores(p)
    S = so <= qa
    S[np.arange(len(p)), p.argmax(1)] = True
    out["APS"] = S
    M = np.zeros_like(p, dtype=bool)
    for c in range(3):
        M[:, c] = (1 - p[:, c]) <= quantile(1 - pb[yb == c, c], a)
    out["ClassLAC"] = M
    return out


def set_metrics(S, y):
    size = S.sum(1)
    single = size == 1
    return {"coverage": float(S[np.arange(len(y)), y].mean()), "size": float(size.mean()), "singleton": float(single.mean()),
            "singleton_acc": float(S[single].argmax(1)[S[single].argmax(1) == y[single]].size / single.sum()) if single.any() else np.nan}


def posthoc_states(Z, y, ca):
    """{state: (probabilities, T, gamma)}, the only place where raw and calibrated predictions are made."""
    out = {}
    for ph in POSTHOC:
        c = Calibrator(ph).fit(Z[ca], y[ca])
        out[ph] = (c(Z), c.T, c.gamma)
    return out


def evaluate_states(states, t, weights, classes, meta, rows):
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    cb = (sp == "calibration") & (sub == "cal_B")
    te, oo, va = sp == "test", sp == "test_ood", sp == "validation"
    views = {"test": te, "chaksu": oo, **{f"src:{s}": te & (src == s) for s in np.unique(src[te])}}
    # Test sources absent from FLAIR's published pretraining list; the others were seen by FLAIR.
    views["test_no_flair_overlap"] = te & np.isin(src, ["afio_composite", "hygd", "palm"])
    for ph, (P, T, gamma) in states.items():
        base = {**meta, "posthoc": ph, "T": T, "gamma": gamma}
        rec = dict(base, view="test", **view_metrics(P[te], y[te]), **weighted(P[te], y[te], weights[te]))
        if np.isfinite(P[va]).all():  # used only to pick representatives; MCQ never scored validation
            rows["views"].append(dict(base, view="validation", **view_metrics(P[va], y[va])))
        per = {}
        for v, m in views.items():
            if v != "test":
                r = view_metrics(P[m], y[m])
                rows["views"].append(dict(base, view=v, **r))
                if v.startswith("src:"):
                    per[v] = r
        for k in ["acc", "bacc", "ece", "mce", "ace", "nll", "brier", "aurc"]:
            rec[f"eq_{k}"] = float(np.mean([r[k] for r in per.values()]))
        rows["views"].append(rec)
        conf, ok = P[cb].max(1), P[cb].argmax(1) == y[cb]
        thr = {"cov80": coverage_threshold(conf, .8), "cov90": coverage_threshold(conf, .9), "risk10": sgr_threshold(conf, ok)}
        for name, th in thr.items():
            for v, m in views.items():
                rows["ops"].append(dict(base, op=name, thr=th, view=v, **operating(P[m], y[m], th, classes)))
        for frac in [.5, .7, .9, 1.0]:
            for v, m in views.items():
                rows["matched"].append(dict(base, frac=frac, view=v, **matched(P[m], y[m], frac, classes)))
        for a in [.10, .05]:
            for method, S in conformal(P[cb], y[cb], P, y, a).items():
                for v, m in views.items():
                    rows["conformal"].append(dict(base, alpha=a, method=method, view=v, **set_metrics(S[m], y[m])))


def calibration_budget(Z, t, meta, rows, repeats=100):
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    ca = np.flatnonzero((sp == "calibration") & (sub == "cal_A"))
    te = sp == "test"
    rng = np.random.default_rng(0)
    oracle = Calibrator("ts").fit(Z[te], y[te]).T
    full = Calibrator("ts").fit(Z[ca], y[ca])
    rows.append(dict(meta, policy="full", size=len(ca), T=full.T, T_oracle=oracle, test_nll=nll(full(Z[te]), y[te])))
    srcs = np.unique(src[ca])
    for size in [50, 100, 200, 500, 1000]:
        if size > len(ca):
            continue
        for policy in ["natural", "source_balanced"]:
            Ts, ls = [], []
            for _ in range(repeats):
                if policy == "natural":
                    idx = rng.choice(ca, size, replace=False)
                else:
                    pr = np.array([1 / (src[ca] == src[i]).sum() for i in ca])
                    idx = rng.choice(ca, size, replace=False, p=pr / pr.sum())
                c = Calibrator("ts").fit(Z[idx], y[idx])
                Ts.append(c.T)
                ls.append(nll(c(Z[te]), y[te]))
            rows.append(dict(meta, policy=policy, size=size, T=float(np.median(Ts)), T_q25=float(np.quantile(Ts, .25)), T_q75=float(np.quantile(Ts, .75)),
                             T_oracle=oracle, test_nll=float(np.median(ls))))
    for s in srcs:
        idx = ca[src[ca] == s]
        c = Calibrator("ts").fit(Z[idx], y[idx])
        rows.append(dict(meta, policy=f"only:{s}", size=len(idx), T=c.T, T_oracle=oracle, test_nll=nll(c(Z[te]), y[te])))


def ood_scores(Z, feats, t, meta, rows):
    """Referral scores: higher = more likely out of scope. Detection of open-set images against the held-out test images."""
    sp, y = t.split.to_numpy(), t.y.to_numpy()
    te, op, va = sp == "test", sp == "test_openset", sp == "validation"
    s = {"MSP": -softmax(Z).max(1), "MaxLogit": -Z.max(1), "Energy": -np.log(np.exp(Z - Z.max(1, keepdims=True)).sum(1)) - Z.max(1)}
    if feats is not None:
        F = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        bank = F[va]
        q = np.r_[np.flatnonzero(te), np.flatnonzero(op)]
        knn = np.zeros(len(F))
        for i in range(0, len(q), 2048):
            sim = F[q[i:i + 2048]] @ bank.T
            knn[q[i:i + 2048]] = 1 - np.sort(sim, 1)[:, -50]
        s["KNN"] = knn
        cov = LedoitWolf().fit(np.concatenate([feats[va & (y == c)] - feats[va & (y == c)].mean(0) for c in range(3)]))
        mu = np.stack([feats[va & (y == c)].mean(0) for c in range(3)])
        prec = cov.precision_
        s["Mahalanobis"] = np.min([np.einsum("ij,jk,ik->i", feats - m, prec, feats - m) for m in mu], 0)
    lab = np.r_[np.zeros(te.sum()), np.ones(op.sum())]
    for name, v in s.items():
        x = np.r_[v[te], v[op]]
        fpr = float((x[lab == 0] >= np.quantile(x[lab == 1], 0.05)).mean())
        rows.append(dict(meta, score=name, auroc=float(roc_auc_score(lab, x)), fpr95=fpr))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-models", nargs="*", default=["dinov2_b14", "clip_b16", "biomedclip", "siglip2_l16", "resnet50"])
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, classes = load_table(a.bench)
    weights = difficulty_weights(t, a.runs, out)
    preds = discover(a.runs)
    print(len(preds), "predictors", flush=True)
    rows = defaultdict(list)
    groups = defaultdict(list)
    y, sp, sub = t.y.to_numpy(), t.split.to_numpy(), t.cal_subset.to_numpy()
    ca = (sp == "calibration") & (sub == "cal_A")
    for setting, model, cfg, seed, path in preds:
        Z = load_logits(path, setting, t)
        meta = dict(setting=setting, model=model, config=cfg, seed=seed)
        states = posthoc_states(Z, y, ca)
        evaluate_states(states, t, weights, classes, meta, rows)
        groups[(setting, model, cfg)].append(states)
        if cfg in ("none/ce", "ce", "clip_ensemble") or setting == "mcq":
            calibration_budget(Z, t, meta, rows["budget"]) if model in a.budget_models or setting == "mcq" else None
            f = Path(a.features) / f"{model}.npy"
            ood_scores(Z, np.load(f).astype(np.float32) if setting == "linear-probe" and f.exists() else None, t, meta, rows["ood"])
        print(setting, model, cfg, seed, flush=True)
    for (setting, model, cfg), ps in groups.items():
        if len(ps) > 1:
            ens = {ph: (np.mean([q[ph][0] for q in ps], 0), np.nan, np.nan) for ph in POSTHOC}
            evaluate_states(ens, t, weights, classes, dict(setting=f"{setting}+ens", model=model, config=cfg, seed=-1), rows)
    for k, v in rows.items():
        pd.DataFrame(v).to_csv(out / f"{k}.csv", index=False)
        print(k, len(v))


if __name__ == "__main__":
    main()
