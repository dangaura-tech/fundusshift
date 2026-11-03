"""Proposed method against each setting's representative: paired bootstrap on test over leakage groups, 2,000 draws, Holm within each metric, as in experiments.paired_tests. Differences are proposed minus baseline.
python -m experiments.proposed_paired --bench data/bench --runs runs --views results/full_eval/views.csv --reps results/paired_tests/representatives.csv --proposed results/proposed/views.csv --out results/proposed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_table
from .full_eval import discover
from .paired_tests import bacc, ece, nll, softmax


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "runs", "views", "reps", "proposed", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--method-dir", default="proposed", help="folder under --runs with this method's logits")
    ap.add_argument("--method-setting", default="proposed", help="setting of the rows of --proposed")
    ap.add_argument("--method-config", default=None, help="config of the rows of --proposed, if it holds several")
    a = ap.parse_args(argv)
    t, _ = load_table(a.bench)
    te = np.flatnonzero((t.split == "test").to_numpy())
    y, groups = t.y.to_numpy()[te], t.group_id.to_numpy()[te]
    v = pd.read_csv(a.views, low_memory=False)
    v["config"] = v.config.fillna("")
    pv = pd.read_csv(a.proposed, low_memory=False)
    reps = pd.read_csv(a.reps)
    reps["config"] = reps.config.fillna("")
    files = {(s, m, str(c), seed): f for s, m, c, seed, f in discover(a.runs)}
    T = v[(v["view"] == "test") & (v.posthoc == "ts")].groupby(["setting", "model", "config", "seed"])["T"].first()
    pv = pv[pv.setting == a.method_setting]
    pv = pv if a.method_config is None else pv[pv.config == a.method_config]
    Tp = pv[(pv["view"] == "test") & (pv.posthoc == "ts")].set_index("seed")["T"]
    state = {}
    for r in reps.itertuples():
        Z = [(seed, np.load(f).astype(np.float64)) for (s, m, c, seed), f in files.items() if (s, m, c) == (r.setting, r.model, r.config)]
        if r.setting == "mcq" or not Z or Z[0][1].ndim != 2 or len(Z[0][1]) != len(t):
            print("skip", r.setting, flush=True)
            continue
        state[r.setting] = ([softmax(z[te]) for _, z in Z], [softmax(z[te] / float(T.loc[(r.setting, r.model, r.config, s)])) for s, z in Z], r.model, r.config)
    Zp = [(int(f.stem[-1]), np.load(f).astype(np.float64)) for f in sorted((Path(a.runs) / a.method_dir).glob("logits_*s?.npy"))]
    raw = [softmax(z[te]) for _, z in Zp]
    ts = [softmax(z[te] / float(Tp.loc[s])) for s, z in Zp]
    ug, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(0)
    W = [np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv].astype(float) for _ in range(a.boot)]
    one = np.ones(len(y))
    fb = lambda P, yy, w: bacc(P.argmax(1), yy, w)
    stat = lambda Ps, fn, w: float(np.mean([fn(P, y, w) for P in Ps]))
    rows = []
    for name, fn, i, mine in [("bacc", fb, 0, raw), ("ece_ts", ece, 1, ts), ("nll_ts", nll, 1, ts)]:
        for s, st in state.items():
            d = stat(mine, fn, one) - stat(st[i], fn, one)
            bs = np.array([stat(mine, fn, w) - stat(st[i], fn, w) for w in W])
            rows.append(dict(family=name, baseline=f"{s}:{st[2]}:{st[3]}", a=stat(mine, fn, one), b=stat(st[i], fn, one), diff=d, lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), p=min(1., 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / a.boot)))
        print("done", name, flush=True)
    d = pd.DataFrame(rows)
    d["p_holm"] = np.nan
    for _, g in d.groupby("family"):
        o = g.sort_values("p")
        d.loc[o.index, "p_holm"] = np.maximum.accumulate(o.p.to_numpy() * (len(o) - np.arange(len(o)))).clip(max=1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    d.to_csv(Path(a.out) / "paired.csv", index=False)
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
