"""Follow-up experiments on the 50 source rotations (CPU only, from saved features and the logits of experiments.revision_runs).

nested  Selection without held-out data. For every rotation and every LOTUS ablation arm, a leave-one-development-source-out score on the
        development side only (seed 0): each development source in turn is held out, the arm is fitted on the others, and the held source's
        train+validation rows are scored. The arm with the highest equal-source score is the one a practitioner would pick; its test
        metrics are read from results/revision/rotation (three seeds), so no test row is used for the choice.
recipe  Input versus recipe. The 2 x 2 of {pooled embedding, mean patch tokens} x {linear-probe recipe (class-weighted CE),
        LOTUS head recipe (label smoothing, class-balanced sampling, AdamW)}, plus LOTUS on pooled features, on every rotation and the
        published split (k = 0), seeds 0-2. Every fit is also scored after label-free EM prior correction (Saerens et al.) per test source.
        Seed-0 logits of the pooled probe are saved for the screen mode.
prior   EM prior correction applied to the saved seed-0 LOTUS rotation logits (no refitting).
screen  Screening operating points: thresholds on P(G) and on 1 - P(N) set on Cal-B for 90% and 95% specificity, read on test and Chaksu,
        for LOTUS, the initial design and the pooled linear probe, published split (seeds 0-2) and rotations (seed 0).

python -m experiments.extra_runs nested --bench data/bench --strata data/strata --zeroshot runs/medsiglip --out results/extra --worker 0 --workers 1
python -m experiments.extra_runs recipe --bench data/bench --strata data/strata --zeroshot runs/medsiglip --features runs/features --out results/extra --worker 0 --workers 1
python -m experiments.extra_runs prior  --bench data/bench --strata data/strata --zeroshot runs/medsiglip --revision results/revision --out results/extra
python -m experiments.extra_runs screen --bench data/bench --strata data/strata --zeroshot runs/medsiglip --revision results/revision --out results/extra
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .calib import Calibrator, ece
from .data import balanced_weights
from .probe_calib import train_head
from .revision_runs import ARMS, K, assign, evaluate
from .scope import load, pipeline
from .supplement import draw_splits


def standardize(F, tr):
    return ((F - F[tr].mean(0)) / (F[tr].std(0) + 1e-6)).astype(np.float32)


def append(f, rows):
    pd.DataFrame(rows).to_csv(f, index=False)


def resume(f, keys):
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    return rows, {tuple(r[k] for k in keys) for r in rows}


# ---------------------------------------------------------------- nested selection
def nested(a):
    t, M, Tb = load(a)
    y, src, scope = t.y.to_numpy(), t.source.to_numpy(), t.normal_scope.to_numpy()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "nested" / f"w{a.worker}.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    rows, done = resume(f, ["split", "arm", "held"])
    splits = draw_splits(t, 50)
    tasks = [(k, arm) for k in range(1, len(splits)) for arm in a.arms]
    for i, (k, arm) in enumerate(tasks):
        if i % a.workers != a.worker:
            continue
        sp, sub, te = assign(t, k, splits[k], k)
        tr0, va0 = sp == "train", sp == "validation"
        X = standardize(M, tr0)
        for s in sorted(set(src[tr0])):
            if (k, arm, s) in done:
                continue
            tr, va = tr0 & (src != s), va0 & (src != s)
            held = (tr0 | va0) & (src == s)
            if len(np.unique(y[tr])) < K or not va.any():
                continue
            Z = pipeline(X, y, scope, src, tr, va, Tb, "exact", k, **ARMS[arm])
            pred = Z[held].argmax(1)
            rows.append(dict(split=k, arm=arm, held=s, n=int(held.sum()), bacc=balanced_accuracy_score(y[held], pred),
                             recall_G=float((pred[y[held] == 2] == 2).mean()) if (y[held] == 2).any() else np.nan))
            append(f, rows)
        print("nested", k, arm, flush=True)


# ---------------------------------------------------------------- EM prior correction
def em_prior(P, prior, iters=100, tol=1e-6):
    """Saerens et al. (2002): re-estimate class priors on unlabelled rows and reweight the posteriors."""
    pi = prior.copy()
    for _ in range(iters):
        Q = P * (pi / prior)
        Q /= Q.sum(1, keepdims=True)
        new = Q.mean(0)
        if np.abs(new - pi).max() < tol:
            break
        pi = new
    Q = P * (pi / prior)
    return Q / Q.sum(1, keepdims=True)


def em_metrics(Z, y, src, sp, sub, te, ch, test):
    """TS on Cal-A, then EM per test source (and on Chaksu) with the training prior; label-free."""
    ca, tr = (sp == "calibration") & (sub == "cal_A"), sp == "train"
    P = Calibrator("ts").fit(Z[ca], y[ca])(Z)
    prior = np.bincount(y[tr], minlength=K) / tr.sum()
    Q = P.copy()
    for s in test:
        m = te & (src == s)
        Q[m] = em_prior(P[m], prior)
    Q[ch] = em_prior(P[ch], prior)
    pred = Q.argmax(1)
    per = [balanced_accuracy_score(y[te & (src == s)], pred[te & (src == s)]) for s in test]
    r = dict(em_test_bacc=balanced_accuracy_score(y[te], pred[te]), em_test_bacc_source_equal=float(np.mean(per)), em_worst_source=float(np.min(per)),
             em_ece=ece(Q[te], y[te])[0], em_chaksu_bacc=balanced_accuracy_score(y[ch], pred[ch]))
    for c, cn in enumerate("NDG"):
        r[f"em_recall_{cn}"] = float((pred[te & (y == c)] == c).mean()) if (te & (y == c)).any() else np.nan
    return r


# ---------------------------------------------------------------- input versus recipe
RECIPE = ["lp_pooled", "lp_tokens", "head_pooled", "lotus_pooled"]


def recipe(a):
    t, M, Tb = load(a)
    y, src, scope = t.y.to_numpy(), t.source.to_numpy(), t.normal_scope.to_numpy()
    ch = t.split.to_numpy() == "test_ood"
    F = np.load(Path(a.features) / "medsiglip.npy").astype(np.float32)
    out = Path(a.out)
    (out / "recipe").mkdir(parents=True, exist_ok=True)
    (out / "lp_logits").mkdir(parents=True, exist_ok=True)
    f = out / "recipe" / f"w{a.worker}.csv"
    rows, done = resume(f, ["split", "seed", "arm"])
    splits = draw_splits(t, 50)
    tasks = [(k, sd, arm) for k in range(len(splits)) for sd in a.seeds for arm in RECIPE]
    for i, (k, sd, arm) in enumerate(tasks):
        if i % a.workers != a.worker or (k, sd, arm) in done:
            continue
        test = splits[k]
        sp, sub, te = assign(t, k, test, k)
        tr, va = sp == "train", sp == "validation"
        X = standardize(F if arm != "lp_tokens" else M, tr)
        seed = k + 1000 * sd
        if arm.startswith("lp_"):
            head = train_head(X[tr], y[tr], balanced_weights(y[tr].tolist()), X[va], y[va], "ce", seed)
            with torch.no_grad():
                Z = head(torch.tensor(X)).numpy()
        elif arm == "head_pooled":
            Z = pipeline(X, y, scope, src, tr, va, Tb, "exact", seed, bag=False, proj=False, text=False)
        else:
            Z = pipeline(X, y, scope, src, tr, va, Tb, "exact", seed, proj=False)
        Z = Z.astype(np.float32)
        if arm == "lp_pooled" and sd == 0:
            np.save(out / "lp_logits" / f"k{k}.npy", Z.astype(np.float16))
        r = dict(split=k, seed=sd, arm=arm, test_sources="|".join(test), **evaluate(Z, y, src, sp, sub, te, ch, test), **em_metrics(Z, y, src, sp, sub, te, ch, test))
        rows.append(r)
        append(f, rows)
        print(k, sd, arm, round(r["test_bacc"], 3), round(r["em_test_bacc"], 3), flush=True)


def prior(a):
    t, _, _ = load(a)
    y, src = t.y.to_numpy(), t.source.to_numpy()
    ch = t.split.to_numpy() == "test_ood"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = draw_splits(t, 50)
    rows = []
    for k in range(1, len(splits)):
        sp, sub, te = assign(t, k, splits[k], k)
        for arm in ["no_projection", "full"]:
            p = Path(a.revision) / "rot_logits" / f"{arm}_k{k}.npy"
            if p.exists():
                rows.append(dict(split=k, arm=arm, **em_metrics(np.load(p).astype(np.float32), y, src, sp, sub, te, ch, splits[k])))
    append(out / "prior_lotus.csv", rows)
    print(pd.DataFrame(rows).groupby("arm").mean(numeric_only=True).round(3).to_string())


# ---------------------------------------------------------------- screening operating points
def screen_rows(Z, y, sp, sub, te, ch):
    ca, cb = (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B")
    P = Calibrator("ts").fit(Z[ca], y[ca])(Z)
    out = []
    for task, s, pos in [("glaucoma", P[:, 2], y == 2), ("any_disease", 1 - P[:, 0], y != 0)]:
        for spec in (.90, .95):
            neg_cb = cb & ~pos
            if not neg_cb.any():
                continue
            thr = np.quantile(s[neg_cb], spec)
            r = dict(task=task, target_spec=spec)
            for name, m in [("test", te), ("chaksu", ch)]:
                p_, n_ = m & pos, m & ~pos
                r[f"{name}_sens"] = float((s[p_] > thr).mean()) if p_.any() else np.nan
                r[f"{name}_spec"] = float((s[n_] <= thr).mean()) if n_.any() else np.nan
                if p_.any() and n_.any():   # oracle: sensitivity at exactly the target specificity, threshold set on the target itself
                    r[f"{name}_sens_at_spec_oracle"] = float((s[p_] > np.quantile(s[n_], spec)).mean())
            out.append(r)
    return out


def screen(a):
    t, _, _ = load(a)
    y = t.y.to_numpy()
    ch = t.split.to_numpy() == "test_ood"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = draw_splits(t, 50)
    rows = []
    rev, ext = Path(a.revision), Path(a.out)
    for k in range(len(splits)):
        sp, sub, te = assign(t, k, splits[k], k)
        cands = []
        if k == 0:
            for sd in (0, 1, 2):
                cands += [("lotus", sd, rev / "final" / f"no_projection_s{sd}.npy"), ("initial_design", sd, rev / "final" / f"full_s{sd}.npy")]
        else:
            cands += [("lotus", 0, rev / "rot_logits" / f"no_projection_k{k}.npy"), ("initial_design", 0, rev / "rot_logits" / f"full_k{k}.npy")]
        cands.append(("linear_probe", 0, ext / "lp_logits" / f"k{k}.npy"))
        for model, sd, p in cands:
            if p.exists():
                for r in screen_rows(np.load(p).astype(np.float32), y, sp, sub, te, ch):
                    rows.append(dict(split=k, seed=sd, model=model, **r))
    d = pd.DataFrame(rows)
    append(out / "screen.csv", rows)
    if len(d):
        d["published"] = d.split == 0
        print(d.groupby(["published", "task", "target_spec", "model"]).mean(numeric_only=True).drop(columns=["split", "seed"]).round(3).to_string())


def main(argv=None):
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 2)))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["nested", "recipe", "prior", "screen"])
    for k in ["bench", "strata", "zeroshot", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--features", default="runs/features")
    ap.add_argument("--revision", default="results/revision")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    a = ap.parse_args(argv)
    {"nested": nested, "recipe": recipe, "prior": prior, "screen": screen}[a.cmd](a)


if __name__ == "__main__":
    main()
