"""Which part of the LOTUS head recipe matters? One head on all development sources + text (0.786 over rotations, within 0.002 of
LOTUS), with one recipe component changed at a time, on the 50 rotations (seed 0). Reference recipe (scope.fit_head): smoothing
eps 0.1, class-balanced resampling, 12 epochs, AdamW lr 3e-4, weight decay 1e-2, best epoch by validation bAcc.
python -m experiments.recipe_ablation --root ~/fundusshift --worker 0 --workers 7
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .data import load_table
from .revision_runs import assign, evaluate
from .scope import BANKS, K, scaled
from .supplement import draw_splits

REF = dict(eps=.1, balanced=True, epochs=12, lr=3e-4, wd=1e-2, select="best")
VARIANTS = {"reference": {}, "no_smoothing": dict(eps=0.), "no_balancing": dict(balanced=False), "epochs_4": dict(epochs=4),
            "epochs_24": dict(epochs=24), "wd_0": dict(wd=0.), "wd_1e-1": dict(wd=1e-1), "lr_1e-3": dict(lr=1e-3), "last_epoch": dict(select="last")}


def head(X, y, Xv, yv, seed, eps, balanced, epochs, lr, wd, select, bs=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    h = torch.nn.Linear(X.shape[1], K)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=wd)
    w = 1 / np.bincount(y, minlength=K)[y] if balanced else np.ones(len(y))
    Xt, yt, Xvt = torch.tensor(X), torch.tensor(y), torch.tensor(Xv)
    best, state = -1., None
    for _ in range(epochs):
        idx = rng.choice(len(X), len(X), replace=True, p=w / w.sum())
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            lp = torch.log_softmax(h(Xt[b]), 1)
            loss = -(1 - eps) * lp[torch.arange(len(b)), yt[b]].mean() - eps * lp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = balanced_accuracy_score(yv, h(Xvt).argmax(1).numpy())
        if select == "last" or v > best:
            best, state = v, {k: x.clone() for k, x in h.state_dict().items()}
    h.load_state_dict(state)
    return h.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/fundusshift")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    r = Path(a.root).expanduser()
    out = r / "results/recipe_ablation"
    if a.summary:
        d = pd.concat([pd.read_csv(f) for f in sorted(out.glob("w*.csv"))])
        d = d[d.split != 0]
        p = d.pivot_table(index="split", columns="variant", values="test_bacc")
        s = d.groupby("variant")[["test_bacc", "ece_ts", "nll_ts", "recall_G", "chaksu_bacc"]].mean()
        s["delta_vs_ref"] = s.test_bacc - s.loc["reference", "test_bacc"]
        s["ref_higher_on"] = [int((p["reference"] > p[v]).sum()) for v in s.index]
        print(len(p), "rotations\n", s.round(3).to_string())
        s.round(4).to_csv(out / "summary.csv")
        return
    t, _ = load_table(r / "data/bench")
    M = np.load(r / "data/strata/mean3_medsiglip.npy")[:, 2].astype(np.float32)
    T = [np.load(r / f"runs/medsiglip/logits_medsiglip_{b}.npy").astype(np.float32) for b in BANKS]
    y, src = t.y.to_numpy(), t.source.to_numpy()
    ch = t.split.to_numpy() == "test_ood"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"w{a.worker}.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(x["split"], x["variant"]) for x in rows}
    for k, test in enumerate(draw_splits(t, 50)):
        if k % a.workers != a.worker:
            continue
        sp, sub, te = assign(t, k, test, k)
        tr, va = sp == "train", sp == "validation"
        X = (M - M[tr].mean(0)) / (M[tr].std(0) + 1e-6)
        Tm = np.mean([scaled(b, va) for b in T], 0)
        for v, ch_ in VARIANTS.items():
            if (k, v) in done:
                continue
            h = head(X[tr], y[tr], X[va], y[va], k, **{**REF, **ch_})
            with torch.no_grad():
                z = (.5 * scaled(h(torch.tensor(X)).numpy(), va) + .5 * Tm).astype(np.float32)
            rows.append(dict(split=k, variant=v, **evaluate(z, y, src, sp, sub, te, ch, test)))
            pd.DataFrame(rows).to_csv(f, index=False)
        print(k, flush=True)


if __name__ == "__main__":
    main()
