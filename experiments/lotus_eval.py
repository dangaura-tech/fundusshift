"""Published-split evaluation of LOTUS (and, for reference, the initial LANTERN design) from saved logits, with the benchmark's own code:
views, frozen operating points and conformal sets through full_eval.evaluate_states, and the paired bootstrap against every
representative through proposed_paired. Logits: <runs>/lotus/logits_s{0,1,2}.npy and <runs>/lantern/logits_s{0,1,2}.npy.
python -m experiments.lotus_eval --bench data/bench --runs runs --out results/lotus
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import proposed_paired
from .calib import POSTHOC
from .data import load_table
from .full_eval import evaluate_states, posthoc_states


def weights_from_table(t, path):
    """Difficulty weights of results/full_eval/difficulty_weights.csv, mapped back onto test rows."""
    cell = pd.read_csv(path)
    key = {(int(a), b): w / n for a, b, w, n in zip(cell.y, cell.source, cell.weight, cell.n)}
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    w = np.zeros(len(t))
    te = sp == "test"
    w[te] = [key[(a, b)] for a, b in zip(y[te], src[te])]
    return w


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "runs", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--weights", default="results/full_eval/difficulty_weights.csv")
    ap.add_argument("--views", default="results/full_eval/views.csv.gz")
    ap.add_argument("--reps", default="results/paired_tests/representatives.csv")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, sub = t.split.to_numpy(), t.y.to_numpy(), t.cal_subset.to_numpy()
    ca = (sp == "calibration") & (sub == "cal_A")
    w = weights_from_table(t, a.weights)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = defaultdict(list)
    for name in ["lotus", "lantern"]:
        ps = []
        for f in sorted((Path(a.runs) / name).glob("logits_s?.npy")):
            st = posthoc_states(np.load(f).astype(np.float32), y, ca)
            evaluate_states(st, t, w, classes, dict(setting="posthoc", model="medsiglip", config=name, seed=int(f.stem[-1])), rows)
            ps.append(st)
        evaluate_states({ph: (np.mean([q[ph][0] for q in ps], 0), np.nan, np.nan) for ph in POSTHOC}, t, w, classes, dict(setting="posthoc+ens", model="medsiglip", config=name, seed=-1), rows)
        print("[eval]", name, flush=True)
    for k, v in rows.items():
        pd.DataFrame(v).to_csv(out / f"{k}.csv", index=False)
    proposed_paired.main(["--bench", a.bench, "--runs", a.runs, "--views", a.views, "--reps", a.reps, "--proposed", str(out / "views.csv"), "--out", str(out),
                          "--boot", str(a.boot), "--method-dir", "lotus", "--method-setting", "posthoc", "--method-config", "lotus"])


if __name__ == "__main__":
    main()
