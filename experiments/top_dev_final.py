"""Post hoc: every configuration with leave-one-source-out development bAcc of .791 or more, on test and Chaksu, after the frozen method's test had been read. Never replaces the frozen method's numbers.
Paired bootstrap against the frozen method over leakage groups, 2,000 draws, Holm across these configurations.
python -m experiments.top_dev_final fit --bench data/bench --features runs/features --zeroshot runs/medsiglip --strata data/strata --out runs/top_dev
python -m experiments.top_dev_final eval --bench data/bench --runs runs --out results/top_dev
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .calib import POSTHOC
from .data import load_table
from .full_eval import difficulty_weights, evaluate_states, posthoc_states
from .paired_tests import bacc, softmax
from .shoal import K
from .strata_gate4 import BANKS
from .strata_gate7 import one, scaled
from .strata_gate12 import RECIPES as R12, fit as fit12
from .strata_gate15 import RECIPES as R15, fit as fit15

DEV = {"mirror_mid": .800, "mirror_reg": .799, "mirror_base": .795, "cross_bag_nap": .794, "strata_bag_nap": .792, "last_reg": .792, "pooled_plus_last": .791, "canonical": .791, "text_mirror": .791}


def fit(a):
    t, _ = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr, va, n = sp == "train", sp == "validation", len(t)
    S = Path(a.strata)
    R, M = np.load(Path(a.features) / "medsiglip.npy").astype(np.float32), np.load(S / "mirror_medsiglip.npy").astype(np.float32)
    mu, sd = R[tr].mean(0), R[tr].std(0) + 1e-6
    X, F = (R - mu) / sd, (M - mu) / sd
    X2, y2, src2 = np.concatenate([X, F]), np.concatenate([y, y]), np.concatenate([src, src])
    left = np.load(S / "disc.npy")[:, 0] < .5
    C = np.where(left[:, None], F, X)
    v = np.load(S / "mean3_medsiglip.npy")[:, 2]
    L = (v - v[tr].mean(0)) / (v[tr].std(0) + 1e-6)
    T = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32), va) for b in BANKS], 0)
    Tf = np.mean([scaled(np.load(S / f"mirror_logits_{b}.npy").astype(np.float32), va) for b in BANKS], 0)
    Tavg, Tc = .5 * (T + Tf), np.where(left[:, None], Tf, T)
    sources = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]

    def bag(head):
        return np.mean([head(tr & (src != o), va & (src != o)) for o in sources], 0)

    def mirror(kw, seed):
        def head(m, vo):
            z = fit15(X2, y2, src2, np.concatenate([m, m]), np.concatenate([vo, vo]), seed, kw)
            return .5 * (scaled(z[:n], va) + scaled(z[n:], va))
        return bag(head)

    for seed in a.seeds:
        base = bag(lambda m, vo: scaled(one(X, y, src, m, vo, "ce", 32, seed), va))
        arms = {
            "mirror_mid": lambda: .5 * mirror(R15["mid"], seed) + .5 * Tavg,
            "mirror_reg": lambda: .5 * mirror(R15["reg"], seed) + .5 * Tavg,
            "mirror_base": lambda: .5 * mirror(R15["base"], seed) + .5 * Tavg,
            "cross_bag_nap": lambda: .5 * bag(lambda m, vo: scaled(one(X, y, src, m, vo, "cross", 32, seed), va)) + .5 * T,
            "strata_bag_nap": lambda: .5 * bag(lambda m, vo: scaled(one(X, y, src, m, vo, "strata", 32, seed), va)) + .5 * T,
            "last_reg": lambda: .5 * bag(lambda m, vo: scaled(fit12(L, y, src, m, vo, 32, seed, R12["reg"]), va)) + .5 * T,
            "pooled_plus_last": lambda: .5 * np.mean([base, bag(lambda m, vo: scaled(one(L, y, src, m, vo, "ce", 32, seed), va))], 0) + .5 * T,
            "canonical": lambda: .5 * bag(lambda m, vo: scaled(one(C, y, src, m, vo, "ce", 32, seed), va)) + .5 * Tc,
            "text_mirror": lambda: .5 * base + .5 * Tavg,
        }
        for name, f in arms.items():
            d = Path(a.out) / name
            d.mkdir(parents=True, exist_ok=True)
            if not (d / f"logits_s{seed}.npy").exists():
                np.save(d / f"logits_s{seed}.npy", f().astype(np.float32))
            print("[ok]", name, seed, flush=True)


def evaluate(a):
    t, classes = load_table(a.bench)
    sp, y, sub, group = t.split.to_numpy(), t.y.to_numpy(), t.cal_subset.to_numpy(), t.group_id.to_numpy()
    ca, te = (sp == "calibration") & (sub == "cal_A"), sp == "test"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = difficulty_weights(t, a.runs, out)
    rows = defaultdict(list)
    runs = Path(a.runs)
    for name in DEV:
        ps = []
        for f in sorted((runs / "top_dev" / name).glob("logits_s?.npy")):
            states = posthoc_states(np.load(f).astype(np.float32), y, ca)
            evaluate_states(states, t, weights, classes, dict(setting="posthoc", model="medsiglip", config=name, seed=int(f.stem[-1])), rows)
            ps.append(states)
        evaluate_states({ph: (np.mean([q[ph][0] for q in ps], 0), np.nan, np.nan) for ph in POSTHOC}, t, weights, classes, dict(setting="posthoc+ens", model="medsiglip", config=name, seed=-1), rows)
        print("[eval]", name, flush=True)
    for k, v in rows.items():
        pd.DataFrame(v).to_csv(out / f"{k}.csv", index=False)
    yt = y[te]
    ug, inv = np.unique(group[te], return_inverse=True)
    rng = np.random.default_rng(0)
    W = [np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv].astype(float) for _ in range(a.boot)]
    load = lambda d, pat: [softmax(np.load(f).astype(np.float64)[te]) for f in sorted(d.glob(pat))]
    ref = load(runs / "proposed", "logits_*_s?.npy")
    stat = lambda Ps, w: float(np.mean([bacc(P.argmax(1), yt, w) for P in Ps]))
    r = []
    for name in DEV:
        A = load(runs / "top_dev" / name, "logits_s?.npy")
        bs = np.array([stat(A, w) - stat(ref, w) for w in W])
        r.append(dict(config=name, dev=DEV[name], test=stat(A, np.ones(len(yt))), frozen=stat(ref, np.ones(len(yt))), diff=stat(A, np.ones(len(yt))) - stat(ref, np.ones(len(yt))), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), p=min(1., 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / a.boot)))
    d = pd.DataFrame(r)
    o = d.sort_values("p")
    d.loc[o.index, "p_holm"] = np.maximum.accumulate(o.p.to_numpy() * (len(o) - np.arange(len(o)))).clip(max=1)
    d.to_csv(out / "paired_vs_frozen.csv", index=False)
    print(d.round(4).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    for k in ["bench", "features", "zeroshot", "strata", "out"]:
        f.add_argument("--" + k, required=True)
    f.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    e = sub.add_parser("eval")
    for k in ["bench", "runs", "out"]:
        e.add_argument("--" + k, required=True)
    e.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    (fit if a.cmd == "fit" else evaluate)(a)


if __name__ == "__main__":
    main()
