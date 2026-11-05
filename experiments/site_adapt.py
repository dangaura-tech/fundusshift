"""Label-free site standardization for LOTUS: token features of each site are standardized with that site's own mean and std
(no labels), the frozen-feature analogue of AdaBN. Rules: docs/protocol_site_adapt.md (written before the run).
  lotus      the published LOTUS: every row standardized with training mean and std
  test_site  same heads; each test site (and Chaksu, and each external cohort) standardized with its own statistics
  all_site   every source standardized with its own statistics, development sources included; heads refit
rotation: the 50 random 7/7 partitions, seed 0, workers; external: published split, seeds 0-2, Messidor-2 and G1020 (exploratory).
python -m experiments.site_adapt rotation --root ~/fundusshift --worker 0 --workers 12
python -m experiments.site_adapt external --root ~/fundusshift
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .calib import Calibrator, ece
from .data import load_table
from .revision_runs import assign, evaluate
from .scope import BANKS, K, fit_head, label_sets, scaled
from .supplement import draw_splits

VARIANTS = ("lotus", "test_site", "all_site")


def standardize(M, groups, ref):
    """Rows of each group in `groups` (dict name -> mask) use that group's own statistics; all other rows use the rows `ref`."""
    X = (M - M[ref].mean(0)) / (M[ref].std(0) + 1e-6)
    for m in groups.values():
        if m.sum() > 1:
            X[m] = (M[m] - M[m].mean(0)) / (M[m].std(0) + 1e-6)
    return X


def lotus(Xfit, Xapp, y, scope, src, tr, va, T, seed):
    """scope.pipeline with proj=False (LOTUS), heads fitted on Xfit and applied to Xapp."""
    A = label_sets(y, scope, "exact")
    Tm = np.mean([scaled(b, va) for b in T], 0)
    zs = []
    for o in [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]:
        m, vo = tr & (src != o), va & (src != o)
        w = 1 / np.bincount(y[m], minlength=K)[y[m]]
        h = fit_head(Xfit[m], A[m], y[m], w, Xfit[vo], y[vo], seed)
        with torch.no_grad():
            zf = h(torch.tensor(Xfit)).numpy()
            za = h(torch.tensor(Xapp)).numpy()
        Zf = zf - zf.mean(1, keepdims=True)
        sd = Zf[va].std()  # the scale of scaled(), always from validation rows of the fitted input
        zs.append((za - za.mean(1, keepdims=True)) / sd)
    return (.5 * np.mean(zs, 0) + .5 * Tm).astype(np.float32)


def variants(M, y, scope, src, sp, tr, va, T, seed, new_sites):
    """Logits of the three variants. new_sites: names of sites that are never fitted on (test, Chaksu, external)."""
    X0 = standardize(M, {}, tr)
    out = {"lotus": lotus(X0, X0, y, scope, src, tr, va, T, seed)}
    newm = {s: src == s for s in new_sites}
    Xt = standardize(M, newm, tr)
    out["test_site"] = lotus(X0, Xt, y, scope, src, tr, va, T, seed)
    dev = np.isin(sp, ["train", "validation", "calibration"])
    devm = {s: dev & (src == s) for s in sorted(set(src[dev])) if s not in new_sites}
    Xa = standardize(M, {**devm, **newm}, tr)
    out["all_site"] = lotus(Xa, Xa, y, scope, src, tr, va, T, seed)
    return out


def load(root):
    r = Path(root).expanduser()
    t, _ = load_table(r / "data/bench")
    M = np.load(r / "data/strata/mean3_medsiglip.npy")[:, 2].astype(np.float32)
    T = [np.load(r / f"runs/medsiglip/logits_medsiglip_{b}.npy").astype(np.float32) for b in BANKS]
    return r, t, M, T


def rotation(a):
    r, t, M, T = load(a.root)
    y, src, scope = t.y.to_numpy(), t.source.to_numpy(), t.normal_scope.to_numpy()
    ch = t.split.to_numpy() == "test_ood"
    out = r / "results/site_adapt"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"rotation_w{a.worker}.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {r_["split"] for r_ in rows}
    splits = draw_splits(t, 50)
    for k, test in enumerate(splits):
        if k % a.workers != a.worker or k in done:
            continue
        sp, sub, te = assign(t, k, test, k)
        tr, va = sp == "train", sp == "validation"
        Z = variants(M, y, scope, src, sp, tr, va, T, k, list(test) + ["chaksu"])
        for v, z in Z.items():
            rows.append(dict(split=k, seed=0, variant=v, test_sources="|".join(test), **evaluate(z, y, src, sp, sub, te, ch, test)))
        pd.DataFrame(rows).to_csv(f, index=False)
        print(k, {v: round(rows[-3 + i]["test_bacc"], 3) for i, v in enumerate(Z)}, flush=True)


def external(a):
    r, t, M0, T0 = load(a.root)
    e = pd.read_csv(r / "data/external/external.csv")
    fe = r / "data/external/features"
    n0 = len(t)
    M = np.concatenate([M0, np.load(fe / "mean3_last.npy").astype(np.float32)])
    T = [np.concatenate([T0[i], np.load(fe / f"logits_medsiglip_{b}.npy").astype(np.float32)]) for i, b in enumerate(BANKS)]
    y = np.concatenate([t.y.to_numpy(), e.y.to_numpy()])
    src = np.concatenate([t.source.to_numpy(), e.source.to_numpy()]).astype(object)
    sp = np.concatenate([t.split.to_numpy(), np.full(len(e), "external")]).astype(object)
    sub = np.concatenate([t.cal_subset.to_numpy().astype(object), np.full(len(e), "", dtype=object)])
    scope = np.concatenate([t.normal_scope.to_numpy(), np.where(e.source == "messidor2", "dr_only", "full")]).astype(object)
    tr, va = sp == "train", sp == "validation"
    test_src = sorted(set(src[sp == "test"]))
    ch = sp == "test_ood"
    te = sp == "test"
    rows = []
    for seed in range(3):
        Z = variants(M, y, scope, src, sp, tr, va, T, 1000 * seed, test_src + ["chaksu", "messidor2", "g1020"])
        ca = (sp == "calibration") & (sub == "cal_A")
        for v, z in Z.items():
            r_ = dict(seed=seed, variant=v, **{k_: v_ for k_, v_ in evaluate(z, y, src, sp, sub, te, ch, test_src).items() if not k_.startswith("src_")})
            P = Calibrator("ts").fit(z[ca], y[ca])(z)
            for s, pos in [("messidor2", 1), ("g1020", 2)]:
                m = src == s
                ym, Pm = y[m], P[m]
                r_[f"{s}_bacc"] = balanced_accuracy_score(ym, Pm.argmax(1))
                r_[f"{s}_recall_pos"] = float((Pm.argmax(1)[ym == pos] == pos).mean())
                r_[f"{s}_recall_N"] = float((Pm.argmax(1)[ym == 0] == 0).mean())
                r_[f"{s}_auroc"] = float(roc_auc_score(ym == pos, 1 - Pm[:, 0]))
                r_[f"{s}_auroc_pos"] = float(roc_auc_score(ym == pos, Pm[:, pos]))
                r_[f"{s}_ece"] = ece(Pm, ym)[0]
            rows.append(r_)
            print(seed, v, round(r_["test_bacc"], 3), round(r_["messidor2_bacc"], 3), round(r_["g1020_auroc"], 3), flush=True)
    out = r / "results/site_adapt"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "external.csv", index=False)


def summary(a):
    r = Path(a.root).expanduser()
    d = pd.concat([pd.read_csv(f) for f in sorted((r / "results/site_adapt").glob("rotation_w*.csv"))])
    cols = ["test_bacc", "ece_ts", "test_bacc_source_equal", "worst_source", "recall_N", "recall_D", "recall_G", "false_G", "chaksu_bacc", "chaksu_auroc", "cov_test", "risk_test", "lac_test"]
    print(len(d) // 3, "rotations")
    print(d.groupby("variant")[cols].mean().round(3).to_string())
    p = d.pivot_table(index="split", columns="variant", values=["test_bacc", "ece_ts"])
    for v in ("test_site", "all_site"):
        print(v, "bAcc higher on", int((p["test_bacc"][v] > p["test_bacc"]["lotus"]).sum()), "ECE lower on", int((p["ece_ts"][v] < p["ece_ts"]["lotus"]).sum()), "of", len(p))
    f = r / "results/site_adapt/external.csv"
    if f.exists():
        x = pd.read_csv(f)
        print(x.groupby("variant")[[c for c in x if c.startswith(("messidor2", "g1020")) or c in ("test_bacc", "ece_ts", "chaksu_bacc")]].mean().round(3).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rotation", "external", "summary"])
    ap.add_argument("--root", default="~/fundusshift")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    dict(rotation=rotation, external=external, summary=summary)[a.mode](a)
