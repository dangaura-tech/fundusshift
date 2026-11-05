"""Baselines asked for in review, on the same 50 rotations (seed 0) and the external cohorts.
  lp_text     the rotation linear probe (saved logits of experiments.extra_runs recipe, lp_pooled) fused 1/2 + 1/2 with the
              same zero-shot text scores as LOTUS: does LOTUS beat a plain probe + zero-shot ensemble?
  seed_bag    single LOTUS head on all development sources, six seeds averaged, + text: is the leave-one-source-out bag
              more than ordinary ensembling? (LOTUS has six or seven heads per rotation)
external: the linear probe refit on the published split (seeds 0-2, rotation recipe) and lp_text, on Messidor-2 and G1020.
python -m experiments.review_baselines rotation --root ~/fundusshift --worker 0 --workers 7
python -m experiments.review_baselines external --root ~/fundusshift
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .calib import Calibrator, ece
from .data import balanced_weights, load_table
from .probe_calib import train_head
from .revision_runs import assign, evaluate
from .scope import BANKS, K, fit_head, label_sets, scaled
from .supplement import draw_splits


def std(X, tr):
    return (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)


def seed_bag(X, y, scope, tr, va, T, seed, n=6):
    A = label_sets(y, scope, "exact")
    w = 1 / np.bincount(y[tr], minlength=K)[y[tr]]
    zs = []
    for j in range(n):
        h = fit_head(X[tr], A[tr], y[tr], w, X[va], y[va], seed + 100 * j)
        with torch.no_grad():
            zs.append(scaled(h(torch.tensor(X)).numpy(), va))
    return (.5 * np.mean(zs, 0) + .5 * np.mean([scaled(b, va) for b in T], 0)).astype(np.float32)


def rotation(a):
    r = Path(a.root).expanduser()
    t, _ = load_table(r / "data/bench")
    M = np.load(r / "data/strata/mean3_medsiglip.npy")[:, 2].astype(np.float32)
    T = [np.load(r / f"runs/medsiglip/logits_medsiglip_{b}.npy").astype(np.float32) for b in BANKS]
    y, src, scope = t.y.to_numpy(), t.source.to_numpy(), t.normal_scope.to_numpy()
    ch = t.split.to_numpy() == "test_ood"
    out = r / "results/review_baselines"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"rotation_w{a.worker}.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {x["split"] for x in rows}
    for k, test in enumerate(draw_splits(t, 50)):
        if k % a.workers != a.worker or k in done:
            continue
        sp, sub, te = assign(t, k, test, k)
        tr, va = sp == "train", sp == "validation"
        LP = np.load(r / f"results/extra/lp_logits/k{k}.npy").astype(np.float32)
        Z = {"lp": LP, "lp_text": (.5 * scaled(LP, va) + .5 * np.mean([scaled(b, va) for b in T], 0)).astype(np.float32),
             "seed_bag": seed_bag(std(M, tr), y, scope, tr, va, T, k)}
        for v, z in Z.items():
            rows.append(dict(split=k, seed=0, variant=v, test_sources="|".join(test), **evaluate(z, y, src, sp, sub, te, ch, test)))
        pd.DataFrame(rows).to_csv(f, index=False)
        print(k, {v: round(x["test_bacc"], 3) for v, x in zip(Z, rows[-3:])}, flush=True)


def external(a):
    r = Path(a.root).expanduser()
    t, _ = load_table(r / "data/bench")
    e = pd.read_csv(r / "data/external/external.csv")
    fe = r / "data/external/features"
    F = np.concatenate([np.load(r / "runs/features/medsiglip.npy"), np.load(fe / "medsiglip_pooled.npy")]).astype(np.float32)
    T = [np.concatenate([np.load(r / f"runs/medsiglip/logits_medsiglip_{b}.npy"), np.load(fe / f"logits_medsiglip_{b}.npy")]).astype(np.float32) for b in BANKS]
    y = np.concatenate([t.y.to_numpy(), e.y.to_numpy()])
    src = np.concatenate([t.source.to_numpy(), e.source.to_numpy()]).astype(object)
    sp = np.concatenate([t.split.to_numpy(), np.full(len(e), "external")]).astype(object)
    sub = np.concatenate([t.cal_subset.to_numpy().astype(object), np.full(len(e), "", dtype=object)])
    tr, va = sp == "train", sp == "validation"
    ca = (sp == "calibration") & (sub == "cal_A")
    X = std(F, tr)
    rows = []
    for sd in range(3):
        head = train_head(X[tr], y[tr], balanced_weights(y[tr].tolist()), X[va], y[va], "ce", 1000 * sd)
        with torch.no_grad():
            LP = head(torch.tensor(X)).numpy().astype(np.float32)
        for v, z in {"lp": LP, "lp_text": (.5 * scaled(LP, va) + .5 * np.mean([scaled(b, va) for b in T], 0)).astype(np.float32)}.items():
            P = Calibrator("ts").fit(z[ca], y[ca])(z)
            te = sp == "test"
            row = dict(seed=sd, variant=v, test_bacc=balanced_accuracy_score(y[te], P[te].argmax(1)), test_ece=ece(P[te], y[te])[0])
            for s, pos in [("messidor2", 1), ("g1020", 2)]:
                m = src == s
                ym, Pm = y[m], P[m]
                row.update({f"{s}_bacc": balanced_accuracy_score(ym, Pm.argmax(1)), f"{s}_recall_pos": float((Pm.argmax(1)[ym == pos] == pos).mean()),
                            f"{s}_pred_G": float((Pm.argmax(1) == 2).mean()), f"{s}_auroc": float(roc_auc_score(ym == pos, 1 - Pm[:, 0])), f"{s}_ece": ece(Pm, ym)[0]})
                if s == "messidor2":
                    row["messidor2_auroc_referable"] = float(roc_auc_score(e.grade.to_numpy()[m[len(t):]] >= 2, 1 - Pm[:, 0]))
            rows.append(row)
            print(sd, v, {k_: round(v_, 3) for k_, v_ in row.items() if isinstance(v_, float)}, flush=True)
    out = r / "results/review_baselines"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "external.csv", index=False)


def summary(a):
    r = Path(a.root).expanduser()
    d = pd.concat([pd.read_csv(f) for f in sorted((r / "results/review_baselines").glob("rotation_w*.csv"))])
    L = pd.concat([pd.read_csv(f) for f in sorted((r / "results/site_adapt").glob("rotation_w*.csv"))])
    d = pd.concat([d, L[L.variant == "lotus"]])
    cols = ["test_bacc", "ece_ts", "nll_ts", "test_bacc_source_equal", "recall_G", "chaksu_bacc"]
    print(d.groupby("variant")[cols].mean().round(3).to_string())
    p = d.pivot_table(index="split", columns="variant", values=["test_bacc", "ece_ts"])
    for v in ("lp", "lp_text", "seed_bag"):
        dd = p["test_bacc"]["lotus"] - p["test_bacc"][v]
        print(v, "LOTUS minus", round(dd.mean(), 3), "+-", round(dd.std(), 3), "higher on", int((dd > 0).sum()), "ECE lower on", int((p["ece_ts"]["lotus"] < p["ece_ts"][v]).sum()), "of", len(p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rotation", "external", "summary"])
    ap.add_argument("--root", default="~/fundusshift")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    dict(rotation=rotation, external=external, summary=summary)[a.mode](a)
