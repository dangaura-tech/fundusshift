"""SCOPE: label-scope-aware training for LANTERN. In DR-only collections a normal label means "no DR"; glaucoma was never assessed. Such rows get the label set {N, G} and the likelihood p(N) + p(G),
so sources that never graded glaucoma stop teaching that their appearance means "not glaucoma". Everything else is LANTERN.
dev: leave-one-development-source-out, reads no test row. rotation: the 50 splits of experiments.supplement. final: published split, all rows.
python -m experiments.scope dev --bench data/bench --strata data/strata --zeroshot runs/medsiglip --out results/scope
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .calib import Calibrator, ece
from .data import load_table
from .shoal import K, metrics
from .strata_gate4 import BANKS, nap
from .strata_gate7 import scaled

N, D, G = 0, 1, 2


def label_sets(y, scope, arm):
    """(n, K) mask of allowed classes."""
    A = np.eye(K, dtype=bool)[y]
    if arm != "exact":
        A[(scope == "dr_only") & (y == N), G] = True
    return A


def fit_head(X, A, y, w, Xv, yv, seed, eps=.1, epochs=12, bs=64, lr=3e-4, wd=1e-2):
    """Linear head; loss -(1 - eps) log sum_{c in A} p_c - eps mean_c log p_c. Best epoch by validation bAcc on exact labels."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    h = torch.nn.Linear(X.shape[1], K)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=wd)
    Xt, At, Xvt = torch.tensor(X), torch.tensor(A), torch.tensor(Xv)
    best, state = -1., None
    for _ in range(epochs):
        idx = rng.choice(len(X), len(X), replace=True, p=w / w.sum())
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            lp = torch.log_softmax(h(Xt[b]), 1)
            loss = -(1 - eps) * torch.logsumexp(lp.masked_fill(~At[b], -1e9), 1).mean() - eps * lp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = balanced_accuracy_score(yv, h(Xvt).argmax(1).numpy())
        if v > best:
            best, state = v, {k: x.clone() for k, x in h.state_dict().items()}
    h.load_state_dict(state)
    return h.eval()


def pipeline(X, y, scope, src, tr, va, T, arm, seed, bag=True, proj=True, text=True, head=True):
    """Logits for every row: leave-one-source-out bag of projected heads, fused with text. Flags switch components off for ablation."""
    A = label_sets(y, scope, arm)
    Tm = np.mean([scaled(b, va) for b in T], 0)
    if not head:
        return Tm
    zs = []
    for o in ([o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K] if bag else [None]):
        m, vo = (tr & (src != o), va & (src != o)) if bag else (tr, va)
        Q = X
        if proj:
            U = nap(X[m], y[m], src[m], 32)
            Q = X - (X @ U.T) @ U
        w = 1 / np.bincount(y[m], minlength=K)[y[m]]
        h = fit_head(Q[m], A[m], y[m], w, Q[vo], y[vo], seed)
        with torch.no_grad():
            zs.append(scaled(h(torch.tensor(Q)).numpy(), va))
    return .5 * np.mean(zs, 0) + .5 * Tm if text else np.mean(zs, 0)


def scope_bacc(pred, y, scope):
    """bAcc where a DR-only normal predicted G is not counted wrong, since glaucoma was never assessed there."""
    ok = (pred == y) | ((scope == "dr_only") & (y == N) & (pred == G))
    return float(np.mean([ok[y == c].mean() for c in np.unique(y)]))


def score(z, y, scope):
    pred = z.argmax(1)
    nN = (y == N)
    return dict(**metrics(z.astype(np.float32), y), bacc_scope=scope_bacc(pred, y, scope), recall_G=(pred[y == G] == G).mean() if (y == G).any() else np.nan,
                false_G=(pred[y != G] == G).mean() if (y != G).any() else np.nan,
                NtoG_dronly=(pred[nN & (scope == "dr_only")] == G).mean() if (nN & (scope == "dr_only")).any() else np.nan,
                NtoG_full=(pred[nN & (scope == "full")] == G).mean() if (nN & (scope == "full")).any() else np.nan)


def load(a):
    t, classes = load_table(a.bench)
    X = np.load(Path(a.strata) / "mean3_medsiglip.npy")[:, 2].astype(np.float32)
    T = [np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32) for b in BANKS]
    return t, X, T


def dev(a):
    t, M, T = load(a)
    sp, y, src, scope = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.normal_scope.to_numpy()
    tr0 = sp == "train"
    X = (M - M[tr0].mean(0)) / (M[tr0].std(0) + 1e-6)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in sorted(set(src[tr0])):
        tr, va, held = tr0 & (src != s), (sp == "validation") & (src != s), np.isin(sp, ["train", "validation"]) & (src == s)
        for seed in a.seeds:
            for arm in a.arms:
                z = pipeline(X, y, scope, src, tr, va, T, arm, seed)[held]
                rows.append(dict(seed=seed, source=s, arm=arm, **score(z, y[held], scope[held])))
            print(s, seed, flush=True)
            pd.DataFrame(rows).to_csv(out / "dev.csv", index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("arm")[["bacc", "bacc_scope", "ece", "nll", "recall_G", "false_G", "NtoG_dronly", "NtoG_full"]].mean().round(3).to_string())
    print(d.pivot_table(index="arm", columns="source", values="bacc").round(3).to_string())
    print(d.pivot_table(index="arm", columns="seed", values="bacc").round(3).to_string())


def rotation(a):
    from .full_eval import coverage_threshold, quantile
    from .supplement import draw_splits
    t, M, T = load(a)
    keep = t.split.isin(["train", "validation", "calibration", "test"]).to_numpy()
    base_sp, y, src, groups, scope = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.group_id.to_numpy(), t.normal_scope.to_numpy()
    ch = base_sp == "test_ood"
    f = Path(a.out) / f"rotation_{a.arms[0]}.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {r["split"] for r in rows}
    for k, test in enumerate(draw_splits(t, 50)):
        if k in done:
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
        Z = pipeline(X, y, scope, src, tr, va, T, a.arms[0], k)
        P, Pr = Calibrator("ts").fit(Z[ca], y[ca])(Z), Calibrator("none").fit(Z[ca], y[ca])(Z)
        pred, conf = P.argmax(1), P.max(1)
        cbi = np.flatnonzero(cb)
        side = np.random.default_rng(k).integers(0, 2, len(cbi)).astype(bool)
        b1, b2 = cbi[side], cbi[~side]
        thr, q = coverage_threshold(conf[b1], .8), quantile(1 - P[b1, y[b1]], .10)

        def read(m):
            ok = conf[m] >= thr
            return float(ok.mean()), float((pred[m][ok] != y[m][ok]).mean()) if ok.any() else np.nan, float(((1 - P[np.flatnonzero(m), y[m]]) <= q).mean())
        mb2 = np.zeros(len(y), bool)
        mb2[b2] = True
        ci, ri, li = read(mb2)
        ce_, re_, le_ = read(te)
        per = [balanced_accuracy_score(y[te & (src == s)], pred[te & (src == s)]) for s in test]
        full = te & (scope == "full")
        r = dict(split=k, test_sources="|".join(test), val_bacc=balanced_accuracy_score(y[va], pred[va]), test_bacc=balanced_accuracy_score(y[te], pred[te]), test_bacc_source_equal=float(np.mean(per)), test_bacc_scope=scope_bacc(pred[te], y[te], scope[te]), test_bacc_fullscope=balanced_accuracy_score(y[full], pred[full]) if len(np.unique(y[full])) > 1 else np.nan, NtoG_full=float((pred[full & (y == N)] == G).mean()) if (full & (y == N)).any() else np.nan, worst_source=float(np.min(per)), ece_raw=ece(Pr[te], y[te])[0], ece_ts=ece(P[te], y[te])[0], cov_internal=ci, risk_internal=ri, lac_internal=li, cov_test=ce_, risk_test=re_, lac_test=le_, chaksu_bacc=balanced_accuracy_score(y[ch], pred[ch]))
        for c, cn in enumerate(["N", "D", "G"]):
            r[f"recall_{cn}"] = float((pred[te & (y == c)] == c).mean()) if (te & (y == c)).any() else np.nan
        rows.append(r)
        pd.DataFrame(rows).to_csv(f, index=False)
        print(k, round(r["test_bacc"], 3), round(r["ece_ts"], 3), flush=True)


def final(a):
    t, M, T = load(a)
    sp, y, src, scope = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.normal_scope.to_numpy()
    tr, va = sp == "train", sp == "validation"
    X = (M - M[tr].mean(0)) / (M[tr].std(0) + 1e-6)
    out = Path(a.out) / f"final_{a.arms[0]}"
    out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        np.save(out / f"logits_s{seed}.npy", pipeline(X, y, scope, src, tr, va, T, a.arms[0], seed).astype(np.float32))
        print("[ok]", seed, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "rotation", "final"])
    for k in ["bench", "strata", "zeroshot", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--arms", nargs="*", default=["exact", "scope"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)
    {"dev": dev, "rotation": rotation, "final": final}[a.cmd](a)


if __name__ == "__main__":
    main()
