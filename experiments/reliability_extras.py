"""Analyses over saved logits that need no training. All use the representatives chosen on validation (see paired_tests).

halving: split Cal-B by leakage group 100 times; thresholds and LAC quantile fitted on one half, read on the other half (same sources) and on test / Chakṣu.
  If the target holds on the other half but not on test, the failure comes from the new sources, not from too few Cal-B images.
rank: score the representatives under fixed source / class mixtures, difficulty-weight variants and random mixtures. These use test labels, so they are exploratory.
diagnose: MbLS margin activity against CE, and the open-set scores of the MCQ models.
python -m experiments.reliability_extras all --bench data/bench --runs runs --views results/full_eval/views.csv --out results/reliability
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .calib import softmax
from .data import load_table
from .full_eval import PANEL, coverage_threshold, discover, load_logits, operating, posthoc_states, quantile, sgr_threshold, weighted
from .paired_tests import representatives


def rep_files(runs, views):
    v = pd.read_csv(views, low_memory=False)
    v["config"] = v.config.fillna("")
    reps = representatives(v)
    want = set(zip(reps.setting, reps.model, reps.config))
    out = {}
    for setting, model, cfg, seed, f in discover(runs):
        if (setting, model, str(cfg)) in want:
            out.setdefault(setting, []).append((model, str(cfg), seed, f))
    return reps, out


def halving(t, classes, files, out, repeats=100):
    sp, y, sub, groups = t.split.to_numpy(), t.y.to_numpy(), t.cal_subset.to_numpy(), t.group_id.to_numpy()
    ca = (sp == "calibration") & (sub == "cal_A")
    cb = np.flatnonzero((sp == "calibration") & (sub == "cal_B"))
    ug, inv = np.unique(groups[cb], return_inverse=True)
    targets = {"test": sp == "test", "chaksu": sp == "test_ood"}
    rows = []
    for setting, runs in files.items():
        for model, cfg, seed, f in runs:
            states = posthoc_states(load_logits(f, setting, t), y, ca)
            for ph, (P, _, _) in states.items():
                conf, ok = P.max(1), P.argmax(1) == y
                rng = np.random.default_rng(seed)
                for rep in range(repeats):
                    side = rng.integers(0, 2, len(ug))[inv].astype(bool)
                    a, b = cb[side], cb[~side]
                    thr = {"cov80": coverage_threshold(conf[a], .8), "cov90": coverage_threshold(conf[a], .9), "risk10": sgr_threshold(conf[a], ok[a])}
                    q = quantile(1 - P[a, y[a]], .10)
                    views = {"other_half": b, **{k: np.flatnonzero(m) for k, m in targets.items()}}
                    for vn, idx in views.items():
                        base = dict(setting=setting, model=model, config=cfg, seed=seed, posthoc=ph, rep=rep, view=vn)
                        for name, th in thr.items():
                            rows.append(dict(base, op=name, thr=th, **operating(P[idx], y[idx], th, classes)))
                        rows.append(dict(base, op="lac90", coverage=float(((1 - P[idx, y[idx]]) <= q).mean()), size=float(((1 - P[idx]) <= q).sum(1).mean())))
        print("halving", setting, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "halving.csv.gz", index=False)
    d = d.groupby(["setting", "model", "config", "posthoc", "view", "op", "rep"]).mean(numeric_only=True).reset_index()  # seed mean within a split
    g = d.groupby(["setting", "model", "config", "posthoc", "view", "op"])
    s = g[["coverage", "risk", "D_to_N_rate", "G_to_N_rate", "size"]].mean()
    for c in ["coverage", "risk"]:
        s[c + "_q05"], s[c + "_q95"] = g[c].quantile(.05), g[c].quantile(.95)
    s.reset_index().to_csv(out / "halving_summary.csv", index=False)


def rank(t, classes, files, runs, out, draws=2000):
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    te = sp == "test"
    yt, st = y[te], src[te]
    cells = sorted({(a, b) for a, b in zip(yt, st)})
    n = np.array([((yt == a) & (st == b)).sum() for a, b in cells], float)
    cy = np.array([a for a, _ in cells])
    cs = np.array([b for _, b in cells])
    panels = {m: [np.load(Path(runs) / "linear_probe" / f"logits_{m}_none_ce_s{s}.npy")[te].argmax(1) for s in range(3)] for m in PANEL}
    err = {m: np.array([np.mean([(p[(yt == a) & (st == b)] != a).mean() for p in ps]) for a, b in cells]) for m, ps in panels.items()}
    mean_err = np.mean(list(err.values()), 0)

    def by_class(raw):
        w = np.zeros(len(cells))
        for c in np.unique(cy):
            w[cy == c] = raw[cy == c] / raw[cy == c].sum() / len(np.unique(cy))
        return w

    schemes = {"pooled": n / n.sum(), "balanced_acc": by_class(n), "equal_cell": by_class(np.ones(len(cells)))}
    es = np.zeros(len(cells))
    for s in np.unique(cs):
        es[cs == s] = 1 / (cs == s).sum() / len(np.unique(cs))
    schemes["equal_source"] = es
    for g in [.5, 1, 2]:
        schemes[f"difficulty_g{g}"] = by_class(mean_err ** g + 1e-12)
    for m in PANEL:
        schemes[f"difficulty_without_{m}"] = by_class(np.mean([e for k, e in err.items() if k != m], 0) + 1e-12)
    for s in np.unique(cs):
        schemes[f"without_{s}"] = by_class(np.where(cs == s, 0, n))
    names, recalls, probs = [], [], {}
    ca = (sp == "calibration") & (t.cal_subset.to_numpy() == "cal_A")
    for setting, rs in files.items():
        r, ph = [], {"none": [], "ts": []}
        for _, _, _, f in rs:
            Z = load_logits(f, setting, t)
            states = posthoc_states(Z, y, ca)
            pred = Z[te].argmax(1)
            r.append([(pred[(yt == a) & (st == b)] == a).mean() for a, b in cells])
            for k in ph:
                ph[k].append(states[k][0][te])
        names.append(setting)
        recalls.append(np.mean(r, 0))
        probs[setting] = ph
    R = np.array(recalls)
    rows = []
    for sn, w in schemes.items():
        score = R @ w
        order = (-score).argsort().argsort() + 1
        per = np.array([w[i] / n[i] for i in range(len(cells))])
        img = np.zeros(len(yt))
        for i, (a, b) in enumerate(cells):
            img[(yt == a) & (st == b)] = per[i]
        for j, s in enumerate(names):
            rec = dict(scheme=sn, setting=s, score=score[j], rank=int(order[j]))
            for k in ["none", "ts"]:
                rec[f"w_ece_{k}"] = float(np.mean([weighted(P, yt, img)["w_ece"] for P in probs[s][k]]))
            rows.append(rec)
    pd.DataFrame(rows).to_csv(out / "rank_schemes.csv", index=False)
    rng = np.random.default_rng(0)
    base = R @ schemes["balanced_acc"]
    top, rho, ranks = np.zeros(len(names)), [], []
    for _ in range(draws):
        w = by_class(rng.dirichlet(np.ones(len(cells))))
        sc = R @ w
        top[sc.argmax()] += 1
        rho.append(spearmanr(base, sc).statistic)
        ranks.append((-sc).argsort().argsort() + 1)
    ranks = np.array(ranks)
    pd.DataFrame({"setting": names, "balanced_acc": base, "top_share": top / draws, "rank_q05": np.quantile(ranks, .05, 0), "rank_median": np.median(ranks, 0), "rank_q95": np.quantile(ranks, .95, 0)}).sort_values("balanced_acc", ascending=False).to_csv(out / "rank_random_mixtures.csv", index=False)
    print("spearman with balanced accuracy over random mixtures: median", np.median(rho), "q05", np.quantile(rho, .05), flush=True)
    pd.DataFrame({"y": cy, "source": cs, "n": n, "panel_error": mean_err}).to_csv(out / "rank_cells.csv", index=False)


def diagnose(t, classes, runs, out):
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr, te = sp == "train", sp == "test"
    rows = []
    preds = {}
    for setting, model, cfg, seed, f in discover(runs):
        if setting == "finetune" and cfg in ("ce", "mbls"):
            Z = np.load(f)
            preds[(model, cfg, seed)] = Z[te].argmax(1)
            gap = Z[tr].max(1, keepdims=True) - Z[tr]
            hinge = np.clip(gap - 10, 0, None).sum(1)
            rows.append(dict(model=model, loss=cfg, seed=seed, train_active=float((hinge > 0).mean()), train_mean_penalty=float(.1 * hinge.mean()), train_max_gap_mean=float(np.sort(gap, 1)[:, -1].mean()), train_max_gap_q99=float(np.quantile(np.sort(gap, 1)[:, -1], .99)),
                             test_max_gap_mean=float((Z[te].max(1) - Z[te].min(1)).mean())))
    pd.DataFrame(rows).to_csv(out / "mbls_margin.csv", index=False)
    ag = []
    for (m, c, s), p in preds.items():
        if c == "ce" and (m, "mbls", s) in preds:
            other = (m, "ce", (s + 1) % 3)
            ag.append(dict(model=m, seed=s, ce_vs_mbls=float((p == preds[(m, "mbls", s)]).mean()), ce_vs_ce_other_seed=float((p == preds[other]).mean()) if other in preds else np.nan))
    pd.DataFrame(ag).to_csv(out / "mbls_agreement.csv", index=False)
    rows, dist = [], []
    m = t.split.isin(["calibration", "test", "test_ood", "test_openset"]).to_numpy()
    op = sp == "test_openset"
    for f in sorted((Path(runs) / "vlm").glob("logits_*.npy")):
        raw = np.load(f)
        if np.isnan(raw).any():
            continue
        assert len(raw) == m.sum(), (f.name, len(raw), m.sum())
        Z = load_logits(f, "mcq", t)
        tag = f.stem[7:]
        msp = softmax(Z[te | op]).max(1)
        full = np.full(len(t), np.nan)
        full[te | op] = -msp
        for name, mask in [("all_test", te), *[(f"class_{classes[c]}", te & (y == c)) for c in range(3)], *[(f"src_{s}", te & (src == s)) for s in np.unique(src[te])]]:
            lab = np.r_[np.zeros(mask.sum()), np.ones(op.sum())]
            rows.append(dict(model=tag, reference=name, auroc=float(roc_auc_score(lab, np.r_[full[mask], full[op]]))))
        for s in np.unique(src[op]):
            mask = op & (src == s)
            lab = np.r_[np.zeros(te.sum()), np.ones(mask.sum())]
            rows.append(dict(model=tag, reference=f"open_{s}_n{mask.sum()}", auroc=float(roc_auc_score(lab, np.r_[full[te], full[mask]]))))
        for name, mask in [("test", te), ("openset", op)]:
            P = softmax(Z[mask])
            dist.append(dict(model=tag, split=name, msp_median=float(np.median(P.max(1))), msp_q10=float(np.quantile(P.max(1), .1)), msp_q90=float(np.quantile(P.max(1), .9)), **{f"pred_{classes[c]}": float((P.argmax(1) == c).mean()) for c in range(3)},
                             vote_agree_6orders=float((raw[np.isin(np.flatnonzero(m), np.flatnonzero(mask))].argmax(2) == raw[np.isin(np.flatnonzero(m), np.flatnonzero(mask))].mean(1).argmax(1)[:, None]).mean())))
    pd.DataFrame(rows).to_csv(out / "mcq_openset_auroc.csv", index=False)
    pd.DataFrame(dist).to_csv(out / "mcq_openset_scores.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["halving", "rank", "diagnose", "all"])
    ap.add_argument("--bench", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, classes = load_table(a.bench)
    reps, files = rep_files(a.runs, a.views)
    reps.to_csv(out / "representatives.csv", index=False)
    if a.task in ("diagnose", "all"):
        diagnose(t, classes, a.runs, out)
    if a.task in ("rank", "all"):
        rank(t, classes, files, a.runs, out)
    if a.task in ("halving", "all"):
        halving(t, classes, files, out)


if __name__ == "__main__":
    main()
