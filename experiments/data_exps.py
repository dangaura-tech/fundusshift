"""Data-centric experiments for the paper (frozen features + linear heads, so they are cheap and backbone-agnostic).

E1 split protocol  : image-level random split vs group-aware random split vs source-held-out (ours) on the same pool
E2 known reuse     : FIVES glaucoma images that are GRAPE copies, scored by a head trained with vs without GRAPE copies
E3 source identity : how well frozen features identify the collection (grouped CV), per backbone
E4 leave-region-out: train on all regions but one, test on that region
E5 label noise     : development with vs without the noisy Bangladesh collection
E6 domain distance : per test collection, distance to development (mean-embedding cosine, linear MMD) vs balanced acc.
E7 data scaling    : 5-100 % of training groups -> validation vs test balanced accuracy
python -m experiments.data_exps --bench data/bench --features runs/features --out runs/data_exps --models dinov2_b14 resnet50_in1k
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ShuffleSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from .calib import bootstrap_ci
from .data import balanced_weights, load_table


def cells(t):
    """(source, class) cell of each row, for balanced_weights."""
    return (t.source + "|" + t.y.astype(str)).to_numpy()


def fit_eval(X, y, tr, te, keys=None, C=0.05, seed=0):
    sc = StandardScaler().fit(X[tr])
    w = balanced_weights(keys[tr]) if keys is not None else None
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(sc.transform(X[tr]), y[tr], sample_weight=w)
    pred = clf.predict(sc.transform(X[te]))
    return balanced_accuracy_score(y[te], pred), clf, sc


def ci(y, pred, groups, n_boot=500):
    """95 % CI of balanced accuracy, resampling leakage groups."""
    lo, hi = bootstrap_ci(lambda i: balanced_accuracy_score(y[i], pred[i]), groups, n_boot)
    return {"bacc_lo": lo, "bacc_hi": hi}


def scored(X, y, tr, te, keys, groups):
    b, clf, sc = fit_eval(X, y, tr, te, keys)
    return {"bacc": b, **ci(y[te], clf.predict(sc.transform(X[te])), groups[te])}


def e1_split_protocols(X, t, seeds=(0, 1, 2)):
    core = (t.y >= 0).to_numpy() & ~t.split.isin(["test_ood"]).to_numpy()
    idx = np.flatnonzero(core)
    y, keys = t.y.to_numpy(), cells(t)
    rows = []
    for seed in seeds:
        # (a) image-level random 80/20 over the pooled dev+test collections
        tr, te = next(ShuffleSplit(1, test_size=0.2, random_state=seed).split(idx))
        rows.append({"protocol": "random image split (pooled)", "seed": seed,
                     "bacc": fit_eval(X, y, idx[tr], idx[te], keys, seed=seed)[0]})
        # (b) group-aware random split (patients / duplicates kept together) over the same pool
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(idx, groups=t.group_id.to_numpy()[idx]))
        rows.append({"protocol": "group-aware random split (pooled)", "seed": seed,
                     "bacc": fit_eval(X, y, idx[tr], idx[te], keys, seed=seed)[0]})
        # (c) FundusShift: development collections -> held-out collections
        sp = t.split.to_numpy()
        rows.append({"protocol": "source-held-out (FundusShift test)", "seed": seed,
                     "bacc": fit_eval(X, y, np.flatnonzero(sp == "train"), np.flatnonzero(sp == "test"), keys, seed=seed)[0]})
    return pd.DataFrame(rows)


def e2_known_reuse(X, t, bench):
    """Score FIVES-G images that are copies of GRAPE images with a head trained on train (clean) and on
    train + the GRAPE copies that FundusShift removed (leaky)."""
    s = pd.read_csv(Path(bench) / "splits_final.csv", dtype=str, keep_default_na=False)
    removed = set(s.image_id[s.unused_reason == "duplicate_of_test_side_image"])
    pairs = pd.read_csv(Path(bench) / "near_duplicates.csv", dtype=str)
    fg = pairs[pairs.source_a.isin(["fives", "grape"]) & pairs.source_b.isin(["fives", "grape"]) & (pairs.source_a != pairs.source_b)]
    fives_copied = set(fg.image_id_a[fg.source_a == "fives"]) | set(fg.image_id_b[fg.source_b == "fives"])
    # removed GRAPE images are not in the feature table (unused rows); report counts and the clean-model score
    y, sp = t.y.to_numpy(), t.split.to_numpy()
    keys = cells(t)
    sc_tr = np.flatnonzero(sp == "train")
    _, clf, sc = fit_eval(X, y, sc_tr, sc_tr[:10], keys)
    fives_g = (t.source == "fives").to_numpy() & (t.label == "G").to_numpy()
    copied = fives_g & t.image_id.isin(fives_copied).to_numpy()
    rec = lambda m: float((clf.predict(sc.transform(X[m])) == y[m]).mean()) if m.any() else None
    return {"grape_images_removed_from_dev": len(removed & set(s.image_id[s.source == "grape"])),
            "fives_G_images_with_grape_copy": int(copied.sum()),
            "G_recall_on_copied_fives_G (clean model)": rec(copied),
            "G_recall_on_other_fives_G (clean model)": rec(fives_g & ~copied)}


def e3_source_identity(X, t):
    dev = t.split.isin(["train", "validation", "calibration", "test"]).to_numpy()
    Xs = StandardScaler().fit_transform(X[dev])
    pred = cross_val_predict(LogisticRegression(C=0.05, max_iter=2000), Xs, t.source.to_numpy()[dev],
                             groups=t.group_id.to_numpy()[dev], cv=GroupKFold(5))
    return {"source_bacc": balanced_accuracy_score(t.source.to_numpy()[dev], pred), "n_sources": int(t.source[dev].nunique()),
            "chance": 1 / t.source[dev].nunique()}


def e4_leave_region_out(X, t):
    core = (t.y >= 0).to_numpy()
    y, reg = t.y.to_numpy(), t.region.to_numpy()
    keys = cells(t)
    rows = []
    for r in sorted(set(reg[core])):
        te = np.flatnonzero(core & (reg == r))
        tr = np.flatnonzero(core & (reg != r) & ~t.split.isin(["test_ood"]).to_numpy())
        if len(set(y[te])) < 2:
            continue
        rows.append({"held_out_region": r, "n_test": len(te), "classes": "".join(sorted(t.label[te].unique())),
                     **scored(X, y, tr, te, keys, t.group_id.to_numpy())})
    return pd.DataFrame(rows)


def e5_label_noise(X, t):
    y, sp = t.y.to_numpy(), t.split.to_numpy()
    keys = cells(t)
    te = np.flatnonzero(sp == "test")
    full = np.flatnonzero(sp == "train")
    no_bd = np.flatnonzero((sp == "train") & (t.source != "bangladesh_eye").to_numpy())
    g = t.group_id.to_numpy()
    return pd.DataFrame([{"train": "all development collections", "n": len(full), **scored(X, y, full, te, keys, g)},
                         {"train": "without Bangladesh (noisy labels)", "n": len(no_bd), **scored(X, y, no_bd, te, keys, g)}])


def e6_domain_distance(X, t):
    sp, y = t.split.to_numpy(), t.y.to_numpy()
    keys = cells(t)
    tr = np.flatnonzero(sp == "train")
    _, clf, sc = fit_eval(X, y, tr, tr[:10], keys)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    mu_dev = Xn[tr].mean(0)
    rows = []
    for s in sorted(set(t.source[np.isin(sp, ["test", "test_ood"])])):
        m = (t.source == s).to_numpy() & np.isin(sp, ["test", "test_ood"])
        mu = Xn[m].mean(0)
        rows.append({"source": s, "split": sp[m][0], "cosine_distance_to_dev": float(1 - mu @ mu_dev / np.linalg.norm(mu) / np.linalg.norm(mu_dev)),
                     "linear_mmd": float(np.linalg.norm(mu - mu_dev)),
                     "classes": "".join(sorted(t.label[m].unique())),   # one class: balanced accuracy is that class's recall
                     "bacc": balanced_accuracy_score(y[m], clf.predict(sc.transform(X[m])))})
    df = pd.DataFrame(rows)
    rho, pv = spearmanr(df.cosine_distance_to_dev, df.bacc)
    df.attrs["spearman"], df.attrs["spearman_p"] = float(rho), float(pv)
    return df


def e7_scaling(X, t, fractions=(0.05, 0.1, 0.25, 0.5, 1.0), seeds=(0, 1, 2)):
    sp, y = t.split.to_numpy(), t.y.to_numpy()
    keys = cells(t)
    tr = np.flatnonzero(sp == "train")
    groups = t.group_id.to_numpy()[tr]
    ug = np.unique(groups)
    rows = []
    for f in fractions:
        for seed in seeds:
            keep = np.random.default_rng(seed).choice(ug, max(1, int(f * len(ug))), replace=False)
            sub = tr[np.isin(groups, keep)]
            for split in ["validation", "test", "test_ood"]:
                rows.append({"fraction": f, "seed": seed, "split": split, "n_train": len(sub),
                             "bacc": fit_eval(X, y, sub, np.flatnonzero(sp == split), keys, seed=seed)[0]})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["dinov2_b14", "resnet50"])
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, _ = load_table(a.bench)
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for name in a.models:
        X = np.load(Path(a.features) / f"{name}.npy").astype(np.float32)
        d = out / name
        d.mkdir(exist_ok=True)
        e1 = e1_split_protocols(X, t)
        e1.to_csv(d / "e1_split_protocols.csv", index=False)
        e4, e5, e6, e7 = e4_leave_region_out(X, t), e5_label_noise(X, t), e6_domain_distance(X, t), e7_scaling(X, t)
        for k, v in {"e4_leave_region_out": e4, "e5_label_noise": e5, "e6_domain_distance": e6, "e7_scaling": e7}.items():
            v.to_csv(d / f"{k}.csv", index=False)
        summary[name] = {"e1": e1.groupby("protocol").bacc.agg(["mean", "std"]).round(4).to_dict("index"),
                         "e2": e2_known_reuse(X, t, a.bench), "e3": e3_source_identity(X, t),
                         "e5": e5.to_dict("records"), "e6_spearman_distance_vs_bacc": e6.attrs["spearman"], "e6_spearman_p": e6.attrs["spearman_p"], "e6_n_sources": len(e6)}
        print(json.dumps({name: summary[name]}, indent=1, default=float), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
