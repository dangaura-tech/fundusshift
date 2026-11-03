"""Representatives chosen on validation, then paired bootstrap on test over leakage groups, with Holm correction per metric family.

A representative is the (model, config) of a setting with the highest validation balanced accuracy, seed mean, raw scores. FLAIR is left out because it saw held-out sources.
Pairs are fixed here, not picked from results. Differences are A - B over the seeds each side has.
python -m experiments.paired_tests --bench data/bench --runs runs --views results/full_eval/views.csv --out results/paired_tests
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_table
from .full_eval import discover

PAIRS = [("finetune", "linear-probe"), ("linear-probe", "16-shot-probe"), ("linear-probe", "mlp-probe"), ("16-shot-probe", "coop"), ("coop-ctxinit", "coop"), ("coop-lp", "coop"), ("tip-adapter", "coop"), ("coop", "zero-shot"), ("finetune", "zero-shot"),
         ("groupdro", "erm"), ("coral", "erm"), ("swad", "erm"), ("groupdro", "finetune"), ("wise-ft", "finetune"), ("finetune-nll", "finetune"),
         ("16-shot-probe", "finetune"), ("16-shot-probe", "zero-shot"), ("linear-probe", "groupdro")]
EXCLUDE = {"flair"}


def softmax(z):
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def bacc(pred, y, w):
    r = [(w[y == c] * (pred[y == c] == c)).sum() / max(w[y == c].sum(), 1e-9) for c in np.unique(y) if w[y == c].sum() > 0]
    return float(np.mean(r))


def ece(P, y, w, bins=15):
    conf, ok = P.max(1), P.argmax(1) == y
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs((w[b == i] * ok[b == i]).sum() - (w[b == i] * conf[b == i]).sum()) for i in range(bins)) / w.sum())


def nll(P, y, w):
    return float((w * -np.log(P[np.arange(len(y)), y] + 1e-12)).sum() / w.sum())


def representatives(v):
    va = v[(v["view"] == "validation") & (v.posthoc == "none") & ~v.model.isin(EXCLUDE) & ~v.setting.str.endswith("+ens")]
    m = va.groupby(["setting", "model", "config"], dropna=False).bacc.mean().reset_index()
    return m.loc[m.groupby("setting").bacc.idxmax()].rename(columns={"bacc": "val_bacc"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = pd.read_csv(a.views, low_memory=False)
    v["config"] = v.config.fillna("")
    reps = representatives(v)
    reps.to_csv(out / "representatives.csv", index=False)
    print(reps.to_string(index=False), flush=True)
    t, _ = load_table(a.bench)
    te = np.flatnonzero((t.split == "test").to_numpy())
    y, groups = t.y.to_numpy()[te], t.group_id.to_numpy()[te]
    files = {}
    for setting, model, cfg, seed, f in discover(a.runs):
        files[(setting, model, str(cfg), seed)] = f
    T = v[(v["view"] == "test") & (v.posthoc == "ts")].groupby(["setting", "model", "config", "seed"])["T"].first()  # CoOp CE appears in both coop and coop_ablation
    state = {}
    for r in reps.itertuples():
        raw, ts = [], []
        for (s, m, c, seed), f in files.items():
            if (s, m, c) == (r.setting, r.model, r.config):
                z = np.load(f).astype(np.float64)[te]
                raw.append(softmax(z))
                ts.append(softmax(z / float(T.loc[(s, m, c, seed)])))
        state[r.setting] = (raw, ts, r.model, r.config)
    ug, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(0)
    weights = [np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv].astype(float) for _ in range(a.boot)]
    one = np.ones(len(y))

    def stat(Ps, fn, w):
        return float(np.mean([fn(P, y, w) for P in Ps]))

    def f_bacc(P, yy, w):
        return bacc(P.argmax(1), yy, w)

    rows = []
    for A, B in PAIRS:
        if A not in state or B not in state:
            continue
        dobs = stat(state[A][0], f_bacc, one) - stat(state[B][0], f_bacc, one)
        bs = np.array([stat(state[A][0], f_bacc, w) - stat(state[B][0], f_bacc, w) for w in weights])
        rows.append({"family": "bacc_between_settings", "a": f"{A}:{state[A][2]}:{state[A][3]}", "b": f"{B}:{state[B][2]}:{state[B][3]}", "diff": dobs, "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975), "p": min(1., 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / a.boot)})
    for s, (raw, ts, m, c) in state.items():
        for name, fn in [("ece", ece), ("nll", nll)]:
            dobs = stat(ts, fn, one) - stat(raw, fn, one)
            bs = np.array([stat(ts, fn, w) - stat(raw, fn, w) for w in weights])
            rows.append({"family": f"{name}_ts_minus_raw", "a": f"{s}:{m}:{c}:ts", "b": f"{s}:{m}:{c}:raw", "diff": dobs, "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975), "p": min(1., 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / a.boot)})
        print("done", s, flush=True)
    d = pd.DataFrame(rows)
    d["p_holm"] = np.nan
    for _, g in d.groupby("family"):
        o = g.sort_values("p")
        adj = np.maximum.accumulate(o.p.to_numpy() * (len(o) - np.arange(len(o)))).clip(max=1)
        d.loc[o.index, "p_holm"] = adj
    d.to_csv(out / "paired_tests.csv", index=False)
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
