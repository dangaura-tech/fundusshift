"""Dataset validation on frozen features and metadata (logistic regression C = 0.05, (source, class)-balanced, as in data_exps).

V1 rotation     : leave one source out, and random choices of 7 held-out sources; where does the released test fall?
V2 NCC          : duplicate pairs and cross-split pairs per threshold; dev images lost and test effect at a lower threshold
V3 label noise  : confident learning on grouped out-of-fold dev predictions, with a synthetic-flip control; retrain without flags
V4 label meaning: N from DR-only vs full-scope sources, DeepDRiD poor-quality images, Chákṣu by number of expert votes
V5 shift        : per test source, label-shift vs conditional part of the accuracy gap; dev-vs-source AUC within class
V6 confounding  : quality-matched classes within source: covariate balance, trivial AUC, head on matched sets
V7 sample size  : source-class cells with the Wilson half-width they allow
V8 quality      : quality per source; correctness vs quality within source
python -m experiments.data_validation --bench data/bench --features runs/features --out results/data_validation
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from .data import balanced_weights, load_table
from .data_exps import cells, fit_eval

QUALITY = ["fov_fraction", "brightness", "contrast", "sharpness", "dark_clip", "bright_clip", "mean_r", "mean_g", "mean_b", "proc_width", "proc_height"]
RELEASED_TEST = ["afio_composite", "fives", "hrf", "hygd", "palm", "papila", "paraguay_dr"]


def bacc_present(y, pred):
    """Balanced accuracy over the classes present in y."""
    return balanced_accuracy_score(y, pred)


def fit_pred(X, y, tr, te, keys):
    b, clf, sc = fit_eval(X, y, tr, te, keys)
    return clf.predict(sc.transform(X[te]))


def eq_source(y, pred, src):
    return float(np.mean([bacc_present(y[src == s], pred[src == s]) for s in np.unique(src)]))


def v1_rotation(X, t, n_parts, seed=0):
    y, src, g, keys = t.y.to_numpy(), t.source.to_numpy(), t.group_id.to_numpy(), cells(t)
    lab = y >= 0
    sources = sorted(set(src[lab]) - {"chaksu"})
    has = {s: set(y[lab & (src == s)]) for s in sources}
    train_only = np.isin(t.split.to_numpy(), ["train", "test", "test_ood"]) & lab

    def run(held):
        te = lab & np.isin(src, held)
        drop = np.isin(g, g[te])
        tr = np.flatnonzero(train_only & ~drop & ~np.isin(src, held) & (src != "chaksu"))
        pred = fit_pred(X, y, tr, np.flatnonzero(te), keys)
        ys, ss = y[te], src[te]
        return {"bacc": bacc_present(ys, pred), "eq_bacc": eq_source(ys, pred, ss), "worst": min(bacc_present(ys[ss == s], pred[ss == s]) for s in held),
                "recall_G": float((pred[ys == 2] == 2).mean()) if (ys == 2).any() else np.nan, "n_train": len(tr), "n_test": int(te.sum())}

    loso = [dict(kind="leave_one_source_out", held=s, **run([s])) for s in sources]
    rows = [dict(kind="released_test", held="+".join(RELEASED_TEST), **run(RELEASED_TEST))]
    rng, seen = np.random.default_rng(seed), set()
    while len(seen) < n_parts:
        held = tuple(sorted(rng.choice(sources, 7, replace=False)))
        dev = [s for s in sources if s not in held]
        if held in seen or set().union(*(has[s] for s in held)) != {0, 1, 2} or set().union(*(has[s] for s in dev)) != {0, 1, 2}:
            continue
        seen.add(held)
        rows.append(dict(kind="random_partition", held="+".join(held), **run(list(held))))
        print("  partition", len(seen), rows[-1]["bacc"], flush=True)
    return pd.DataFrame(loso + rows)


def v2_ncc(t, bench, models):
    pairs = pd.read_csv(Path(bench) / "dup_candidates.csv", dtype={"image_id_a": str, "image_id_b": str})
    sp = t.set_index("image_id").split
    pairs["sa"], pairs["sb"] = pairs.image_id_a.map(sp), pairs.image_id_b.map(sp)
    both = pairs.dropna(subset=["sa", "sb"])
    dev = {"train", "validation", "calibration"}
    side = lambda s: s.map(lambda x: "dev" if x in dev else "held_out")
    rows = []
    for th in [.5, .55, .6, .65, .7, .75, .8, .9]:
        p, q = pairs[pairs.ncc >= th], both[both.ncc >= th]
        cross = q[q.sa != q.sb]
        dvh = q[side(q.sa) != side(q.sb)]
        rows.append({"ncc_threshold": th, "pairs_scored": len(p), "pairs_cross_source": int((p.source_a != p.source_b).sum()), "label_disagree": int((p.label_a != p.label_b).sum()),
                     "released_pairs": len(q), "released_cross_split": len(cross), "released_dev_vs_held_out": len(dvh), "released_images_in_dev_vs_held_out": len(set(dvh.image_id_a) | set(dvh.image_id_b))})
    # stricter threshold: remove dev images linked to held-out images at NCC >= th, refit
    y, keys, s = t.y.to_numpy(), cells(t), t.split.to_numpy()
    te, tr0 = np.flatnonzero(s == "test"), s == "train"
    effect = []
    for name, X in models.items():
        base = bacc_present(y[te], fit_pred(X, y, np.flatnonzero(tr0), te, keys))
        for th in [.5, .6]:
            q = both[both.ncc >= th]
            q = q[side(q.sa) != side(q.sb)]
            ids = set(q.image_id_a[q.sa.isin(dev)]) | set(q.image_id_b[q.sb.isin(dev)])
            keep = tr0 & ~t.image_id.isin(ids).to_numpy()
            effect.append({"model": name, "ncc_threshold": th, "train_removed": int(tr0.sum() - keep.sum()), "test_bacc_released": base, "test_bacc_stricter": bacc_present(y[te], fit_pred(X, y, np.flatnonzero(keep), te, keys))})
    return pd.DataFrame(rows), pd.DataFrame(effect)


def oof_proba(X, y, idx, groups, keys):
    P = np.zeros((len(idx), 3))
    for tr, te in GroupKFold(5).split(idx, groups=groups[idx]):
        sc = StandardScaler().fit(X[idx[tr]])
        clf = LogisticRegression(C=.05, max_iter=500).fit(sc.transform(X[idx[tr]]), y[idx[tr]], sample_weight=balanced_weights(keys[idx[tr]]))
        P[te] = clf.predict_proba(sc.transform(X[idx[te]]))
    return P


def confident_flags(P, y):
    th = np.array([P[y == c, c].mean() for c in range(3)])
    above = P >= th
    other = above.copy()
    other[np.arange(len(y)), y] = False
    return other.any(1) & (P[np.arange(len(y)), y] < th[y])


def v3_label_noise(models, t, flip_rate=.05, seed=0):
    y, keys, g, s = t.y.to_numpy(), cells(t), t.group_id.to_numpy(), t.split.to_numpy()
    dev = np.flatnonzero((np.isin(s, ["train", "validation", "calibration"])) & (y >= 0))
    te = np.flatnonzero(s == "test")
    Ps = {m: oof_proba(X, y, dev, g, keys) for m, X in models.items()}
    P = np.mean(list(Ps.values()), 0)
    flag = confident_flags(P, y[dev])
    votes = np.sum([confident_flags(p, y[dev]) for p in Ps.values()], 0)
    d = t.iloc[dev].assign(flag=flag, votes=votes, p_given=P[np.arange(len(dev)), y[dev]], suggest=P.argmax(1))
    # synthetic flips, same detector as above: how well does it find known wrong labels?
    rng = np.random.default_rng(seed)
    yf = y.copy()
    flip = rng.choice(dev, int(flip_rate * len(dev)), replace=False)
    yf[flip] = (yf[flip] + rng.integers(1, 3, len(flip))) % 3
    keys_f = cells(t.assign(y=yf))
    Pf = {m: oof_proba(X, yf, dev, g, keys_f) for m, X in models.items()}
    known = np.isin(dev, flip)
    control = []
    for arm, fl in [("ensemble", confident_flags(np.mean(list(Pf.values()), 0), yf[dev])), ("all_models_agree", np.sum([confident_flags(p, yf[dev]) for p in Pf.values()], 0) == len(models))]:
        control.append({"detector": arm, "flip_rate": flip_rate, "recall_of_flips": float(fl[known].mean()), "precision": float(known[fl].mean()), "flag_rate": float(fl.mean()),
                        "flag_rate_without_flips": float(flag.mean() if arm == "ensemble" else (votes == len(models)).mean())})
    retrain = []
    consensus = votes == len(models)
    for m, X in models.items():
        tr = (s == "train") & (y >= 0)
        for name, f in [("all_models_agree", consensus), ("ensemble", flag)]:
            keep = tr.copy()
            keep[dev[f]] = False
            pa, pb = fit_pred(X, y, np.flatnonzero(tr), te, keys), fit_pred(X, y, np.flatnonzero(keep), te, keys)
            retrain.append({"model": m, "drop": name, "train_removed": int(tr.sum() - keep.sum()), "test_bacc_all": bacc_present(y[te], pa), "test_bacc_cleaned": bacc_present(y[te], pb),
                            "recall_G_all": float((pa[y[te] == 2] == 2).mean()), "recall_G_cleaned": float((pb[y[te] == 2] == 2).mean())})
    rate = d.groupby(["source", "label"]).agg(n=("flag", "size"), flag_rate=("flag", "mean"), all_models_agree=("votes", lambda v: float((v == len(models)).mean()))).reset_index()
    return d[d.flag][["image_id", "source", "label", "native_label", "votes", "p_given", "suggest"]], rate, pd.DataFrame(control), pd.DataFrame(retrain)


def v4_label_meaning(X, t, name):
    y, s, src, keys, nat = t.y.to_numpy(), t.split.to_numpy(), t.source.to_numpy(), cells(t), t.native_label.to_numpy()
    tr, te = (s == "train") & (y >= 0), np.flatnonzero(s == "test")
    scope = t.normal_scope.to_numpy()
    arms = {"all": tr, "N from full-scope sources only": tr & ~((y == 0) & (scope == "dr_only")), "N from DR-only sources only": tr & ~((y == 0) & (scope == "full")),
            "drop DeepDRiD poor_quality": tr & ~((src == "deepdrid") & t.native_label.str.contains("poor_quality").to_numpy())}
    rows = []
    for arm, m in arms.items():
        p = fit_pred(X, y, np.flatnonzero(m), te, keys)
        ys = y[te]
        rows.append({"model": name, "train": arm, "n_train": int(m.sum()), "n_train_N": int((y[m] == 0).sum()), "test_bacc": bacc_present(ys, p), **{f"recall_{c}": float((p[ys == i] == i).mean()) for i, c in enumerate("NDG")},
                     "recall_N_full_scope_test": float((p[(ys == 0) & (scope[te] == "full")] == 0).mean()), "recall_N_dr_only_test": float((p[(ys == 0) & (scope[te] == "dr_only")] == 0).mean())})
    ch = np.flatnonzero(s == "test_ood")
    p = fit_pred(X, y, np.flatnonzero(tr), ch, keys)
    d = t.iloc[ch].assign(pred=p)
    d["votes"] = d.native_label.str.extract(r"votes_(\d)of5").astype(float)
    d["device"] = d.native_label.str.extract(r"device_(\w+)")
    votes = d.groupby(["label", "votes"]).apply(lambda g: pd.Series({"n": len(g), "pred_N": (g.pred == 0).mean(), "pred_D": (g.pred == 1).mean(), "pred_G": (g.pred == 2).mean()}), include_groups=False).reset_index().assign(model=name)
    return pd.DataFrame(rows), votes


def v5_shift(X, t, name):
    y, s, src, keys = t.y.to_numpy(), t.split.to_numpy(), t.source.to_numpy(), cells(t)
    tr, va = np.flatnonzero((s == "train") & (y >= 0)), np.flatnonzero((s == "validation") & (y >= 0))
    _, clf, sc = fit_eval(X, y, tr, va, keys)
    pred = clf.predict(sc.transform(X))
    rec_val = np.array([(pred[va][y[va] == c] == c).mean() for c in range(3)])
    prior_dev = np.bincount(y[tr], minlength=3) / len(tr)
    acc_val = float((prior_dev * rec_val).sum())
    rows = []
    for so in sorted(set(src[np.isin(s, ["test", "test_ood"])])):
        m = np.flatnonzero((src == so) & (y >= 0) & np.isin(s, ["test", "test_ood"]))
        prior = np.bincount(y[m], minlength=3) / len(m)
        acc_label = float((prior * rec_val).sum())
        acc = float((pred[m] == y[m]).mean())
        aucs = []
        for c in np.unique(y[m]):
            a, b = m[y[m] == c], tr[y[tr] == c]
            if len(a) < 10:
                continue
            b = np.random.default_rng(0).choice(b, min(len(b), 2000), replace=False)
            idx = np.concatenate([a, b])
            lab = np.r_[np.ones(len(a)), np.zeros(len(b))]
            prob = cross_val_predict(LogisticRegression(C=.05, max_iter=300, class_weight="balanced"), StandardScaler().fit_transform(X[idx]), lab, cv=3, method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(lab, prob))
        rows.append({"model": name, "source": so, "tv_prior": .5 * float(np.abs(prior - prior_dev).sum()), "acc_val": acc_val, "acc_val_at_source_prior": acc_label, "acc_source": acc, "gap_label_shift": acc_val - acc_label, "gap_conditional": acc_label - acc,
                     "domain_auc_within_class": float(np.mean(aucs)) if aucs else np.nan})
    return pd.DataFrame(rows)


def match_classes(q, y, rng, caliper=.2):
    """Indices kept after 1:1 propensity matching of every class to the smallest class present (within one source)."""
    cl, n = np.unique(y, return_counts=True)
    if len(cl) < 2:
        return np.arange(len(y))
    ref = cl[n.argmin()]
    Z = StandardScaler().fit_transform(q)
    keep = {int(i) for i in np.flatnonzero(y == ref)}
    used_ref = None
    for c in cl[cl != ref]:
        pair = np.isin(y, [ref, c])
        ps = LogisticRegression(max_iter=500).fit(Z[pair], y[pair] == c).decision_function(Z)
        cal = caliper * ps[pair].std()
        ctrl = list(rng.permutation(np.flatnonzero(y == c)))
        matched_ref = set()
        for r in rng.permutation(np.flatnonzero(y == ref)):
            if not ctrl:
                break
            j = int(np.argmin(np.abs(ps[ctrl] - ps[r])))
            if abs(ps[ctrl[j]] - ps[r]) <= cal:
                keep.add(int(ctrl.pop(j)))
                matched_ref.add(int(r))
        used_ref = matched_ref if used_ref is None else used_ref & matched_ref
    keep -= {int(i) for i in np.flatnonzero(y == ref)} - (used_ref or set())
    return np.array(sorted(keep))


def v6_confounding(models, t):
    y, s, src, keys = t.y.to_numpy(), t.split.to_numpy(), t.source.to_numpy(), cells(t)
    Q = t[QUALITY].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    rng = np.random.default_rng(0)
    tr = (s == "train") & (y >= 0)
    rows, sets = [], {}
    for so in sorted(set(src[y >= 0])):
        m = np.flatnonzero((src == so) & (y >= 0) & (s != "calibration") & (s != "validation"))
        if len(np.unique(y[m])) < 2:
            continue
        k = m[match_classes(Q[m], y[m], rng)]
        sets[so] = k
        for sub, idx in [("all", m), ("matched", k)]:
            if len(np.unique(y[idx])) < 2 or np.bincount(y[idx]).max() < 10:
                continue
            aucs, smds = [], []
            for c in np.unique(y[idx]):
                lab = y[idx] == c
                if lab.sum() < 5 or (~lab).sum() < 5:
                    continue
                prob = cross_val_predict(RandomForestClassifier(200, min_samples_leaf=3, random_state=0, n_jobs=-1), Q[idx], lab, cv=3, method="predict_proba")[:, 1]
                aucs.append(roc_auc_score(lab, prob))
                sd = Q[idx].std(0) + 1e-9
                smds.append(np.abs(Q[idx][lab].mean(0) - Q[idx][~lab].mean(0)) / sd)
            rows.append({"source": so, "split": s[m[0]], "subset": sub, "n": len(idx), "classes": "".join("NDG"[c] for c in np.unique(y[idx])), "trivial_auc_mean": float(np.mean(aucs)), "max_abs_smd": float(np.max(smds))})
    head = []
    keep_tr = np.zeros(len(t), bool)
    for so, k in sets.items():
        keep_tr[k] = True
    dev_sources = set(src[tr])
    keep_tr |= tr & np.isin(src, [so for so in dev_sources if so not in sets])
    keep_tr &= tr
    test_sets = np.concatenate([sets[so] for so in sets if so not in dev_sources and s[sets[so][0]] == "test"])
    test_all = np.flatnonzero((s == "test") & np.isin(src, [so for so in sets if so not in dev_sources]))
    te = np.flatnonzero(s == "test")
    for name, X in models.items():
        pa = fit_pred(X, y, np.flatnonzero(tr), te, keys)
        pb = fit_pred(X, y, np.flatnonzero(keep_tr), te, keys)
        pall, pmatch = fit_pred(X, y, np.flatnonzero(tr), test_all, keys), fit_pred(X, y, np.flatnonzero(tr), test_sets, keys)
        head.append({"model": name, "n_train": int(tr.sum()), "n_train_matched": int(keep_tr.sum()), "test_bacc": bacc_present(y[te], pa), "test_bacc_train_matched": bacc_present(y[te], pb),
                     "multiclass_test_bacc": bacc_present(y[test_all], pall), "multiclass_test_matched_bacc": bacc_present(y[test_sets], pmatch), "n_multiclass_test": len(test_all), "n_multiclass_test_matched": len(test_sets)})
    return pd.DataFrame(rows), pd.DataFrame(head)


def v7_sample_size(t):
    rows = []
    for (so, sp, lab), g in t[t.y >= 0].groupby(["source", "split", "label"]):
        n = len(g)
        rows.append({"source": so, "split": sp, "label": lab, "n": n, "wilson_halfwidth_at_p50": 1.96 * .5 / np.sqrt(n + 1.96 ** 2) * np.sqrt(1 + 1.96 ** 2 / n) if n else np.nan, "patient_groups": g.group_id.nunique()})
    return pd.DataFrame(rows)


def v8_quality(X, t, name):
    y, s, src, keys = t.y.to_numpy(), t.split.to_numpy(), t.source.to_numpy(), cells(t)
    Q = t[QUALITY].apply(pd.to_numeric, errors="coerce")
    med = Q.assign(source=src, split=s).groupby("source")[QUALITY].median().round(3).reset_index()
    te = np.flatnonzero(np.isin(s, ["test", "test_ood"]) & (y >= 0))
    pred = fit_pred(X, y, np.flatnonzero((s == "train") & (y >= 0)), te, keys)
    ok = pred == y[te]
    rows = []
    for so in np.unique(src[te]):
        m = src[te] == so
        for q in ["fov_fraction", "brightness", "contrast", "sharpness"]:
            v = Q[q].to_numpy()[te][m]
            r = spearmanr(v, ok[m]) if ok[m].std() > 0 and np.nanstd(v) > 0 else (np.nan, np.nan)
            rows.append({"model": name, "source": so, "quality": q, "n": int(m.sum()), "spearman_with_correct": float(r[0]), "p": float(r[1])})
    return med, pd.DataFrame(rows)


def v12_licence_subsets(X, t, name):
    """Train without collections whose terms may limit redistribution, in case the release has to change."""
    y, s, src, keys = t.y.to_numpy(), t.split.to_numpy(), t.source.to_numpy(), cells(t)
    tr, te, ch = (s == "train") & (y >= 0), np.flatnonzero(s == "test"), np.flatnonzero(s == "test_ood")
    rows = []
    for arm, drop in [("all", []), ("without DDR (NC-SA)", ["ddr"]), ("without DDR and DeepDRiD (NC or SA)", ["ddr", "deepdrid"])]:
        m = tr & ~np.isin(src, drop)
        p, q = fit_pred(X, y, np.flatnonzero(m), te, keys), fit_pred(X, y, np.flatnonzero(m), ch, keys)
        rows.append({"model": name, "train": arm, "n_train": int(m.sum()), "test_bacc": bacc_present(y[te], p), "chaksu_bacc": bacc_present(y[ch], q), **{f"recall_{c}": float((p[y[te] == i] == i).mean()) for i, c in enumerate("NDG")}})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["dinov2_b14", "resnet50", "medsiglip"])
    ap.add_argument("--parts", type=int, default=30)
    ap.add_argument("--only", nargs="*", default=["v7", "v8", "v2", "v4", "v5", "v6", "v3", "v1"])
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, _ = load_table(a.bench)
    models = {m: np.load(Path(a.features) / f"{m}.npy").astype(np.float32) for m in a.models}

    def save(d, name):
        d.to_csv(out / f"{name}.csv", index=False)
        print("saved", name, len(d), flush=True)

    if "v7" in a.only:
        save(v7_sample_size(t), "v7_sample_size")
    if "v8" in a.only:
        r = [v8_quality(X, t, m) for m, X in models.items()]
        save(r[0][0], "v8_quality_by_source")
        save(pd.concat([x[1] for x in r]), "v8_quality_vs_correct")
    if "v2" in a.only:
        d, e = v2_ncc(t, a.bench, models)
        save(d, "v2_ncc_threshold")
        save(e, "v2_ncc_effect")
    if "v4" in a.only:
        r = [v4_label_meaning(X, t, m) for m, X in models.items()]
        save(pd.concat([x[0] for x in r]), "v4_label_scope")
        save(pd.concat([x[1] for x in r]), "v4_chaksu_votes")
    if "v5" in a.only:
        save(pd.concat([v5_shift(X, t, m) for m, X in models.items()]), "v5_shift")
    if "v6" in a.only:
        d, h = v6_confounding(models, t)
        save(d, "v6_matching")
        save(h, "v6_matched_heads")
    if "v3" in a.only:
        fl, rate, ctl, re = v3_label_noise(models, t)
        save(fl, "v3_flagged_dev")
        save(rate, "v3_flag_rate")
        save(ctl, "v3_control")
        save(re, "v3_retrain")
    if "v1" in a.only:
        save(pd.concat([v1_rotation(X, t, a.parts).assign(model=m) for m, X in models.items()]), "v1_rotation")
    if "v12" in a.only:
        save(pd.concat([v12_licence_subsets(X, t, m) for m, X in models.items()]), "v12_licence_subsets")
    (out / "run_config.json").write_text(json.dumps(vars(a), indent=1))


if __name__ == "__main__":
    main()
