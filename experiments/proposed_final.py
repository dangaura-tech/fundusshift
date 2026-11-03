"""Applies docs/proposed_frozen_config.json: fit on the development sources, logits for every row, then the evaluator of experiments.full_eval on this method alone.
python -m experiments.proposed_final fit --bench data/bench --features runs/features --zeroshot runs/medsiglip --out runs/proposed
python -m experiments.proposed_final eval --bench data/bench --runs runs --out results/proposed
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .calib import POSTHOC
from .data import load_table
from .full_eval import difficulty_weights, evaluate_states, posthoc_states
from .shoal import K, load_x
from .strata_gate4 import BANKS
from .strata_gate7 import one, scaled

SETTING, MODEL, CONFIG = "proposed", "medsiglip", "bag_nap_allbanks"


def fit(a):
    t, classes = load_table(a.bench)
    assert classes == ["N", "D", "G"]
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    X = load_x(a.features, MODEL, t)
    tr, va = sp == "train", sp == "validation"
    T = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_{MODEL}_{b}.npy").astype(np.float32), va) for b in BANKS], 0)
    sources = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        Z = np.mean([scaled(one(X, y, src, tr & (src != o), va & (src != o), "ce", 32, seed), va) for o in sources], 0)
        np.save(out / f"logits_{MODEL}_s{seed}.npy", (.5 * Z + .5 * T).astype(np.float32))
        print("[ok] seed", seed, flush=True)


def evaluate(a):
    t, classes = load_table(a.bench)
    y, sp, sub = t.y.to_numpy(), t.split.to_numpy(), t.cal_subset.to_numpy()
    ca = (sp == "calibration") & (sub == "cal_A")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = difficulty_weights(t, a.runs, out)
    rows, ps = defaultdict(list), []
    for f in sorted((Path(a.runs) / "proposed").glob("logits_*_s?.npy")):
        Z = np.load(f).astype(np.float32)
        states = posthoc_states(Z, y, ca)
        evaluate_states(states, t, weights, classes, dict(setting=SETTING, model=MODEL, config=CONFIG, seed=int(f.stem[-1])), rows)
        ps.append(states)
    ens = {ph: (np.mean([q[ph][0] for q in ps], 0), np.nan, np.nan) for ph in POSTHOC}
    evaluate_states(ens, t, weights, classes, dict(setting=SETTING + "+ens", model=MODEL, config=CONFIG, seed=-1), rows)
    for k, v in rows.items():
        pd.DataFrame(v).to_csv(out / f"{k}.csv", index=False)
    print("[ok]", {k: len(v) for k, v in rows.items()}, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    for k in ["bench", "features", "zeroshot", "out"]:
        f.add_argument("--" + k, required=True)
    f.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    e = sub.add_parser("eval")
    for k in ["bench", "runs", "out"]:
        e.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    (fit if a.cmd == "fit" else evaluate)(a)


if __name__ == "__main__":
    main()
