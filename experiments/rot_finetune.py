"""Fine-tuned baselines over source rotations (asked in review): ConvNeXt-T with CE (experiments.finetune) and with Group DRO
(experiments.robust_train), unchanged recipes, on the split of rotation k (revision_runs.assign), seed 0; then the benchmark's
rotation metrics (revision_runs.evaluate), so rows compare directly with LOTUS and the probes on the same partitions.
python -m experiments.rot_finetune --root ~/fundusshift --k 1 --method ce
python -m experiments.rot_finetune --root ~/fundusshift --summary
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .revision_runs import assign, evaluate
from .supplement import draw_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/fundusshift")
    ap.add_argument("--k", type=int)
    ap.add_argument("--method", choices=["ce", "groupdro"])
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    r = Path(a.root).expanduser()
    out = r / "results/rot_finetune"
    out.mkdir(parents=True, exist_ok=True)
    t0, _ = data.load_table(r / "data/bench")
    if a.summary:
        d = pd.concat([pd.read_csv(f) for f in sorted(out.glob("k*_*.csv"))])
        ks = sorted(d.split.unique())
        L = pd.concat([pd.read_csv(f) for f in (r / "results/site_adapt").glob("rotation_w*.csv")])
        B = pd.concat([pd.read_csv(f) for f in (r / "results/review_baselines").glob("rotation_w*.csv")])
        ref = pd.concat([L[L.variant == "lotus"], B[B.variant.isin(["lp", "lp_text"])]]).rename(columns={"variant": "method"})
        d = pd.concat([d, ref[ref.split.isin(ks)]])
        cols = ["test_bacc", "test_bacc_source_equal", "recall_G", "ece_ts", "risk_test", "lac_test", "chaksu_bacc"]
        s = d.groupby("method")[cols].mean()
        p = d.pivot_table(index="split", columns="method", values="test_bacc")
        s["lotus_higher_on"] = [int((p["lotus"] > p[m]).sum()) for m in s.index]
        print(len(ks), "rotations:", ks, "\n", s.round(3).to_string())
        s.round(4).to_csv(out / "summary.csv")
        return
    test = draw_splits(t0, 50)[a.k]
    sp, sub, te = assign(t0, a.k, test, a.k)
    orig = data.load_table

    def patched(bench):
        t, classes = orig(bench)
        t = t.copy()
        t["split"], t["cal_subset"] = sp, sub
        return t, classes
    runs = r / "runs/rot_finetune" / f"k{a.k}"
    args = ["--bench", str(r / "data/bench"), "--images", str(r / "data/processed"), "--out", str(runs), "--seed", "0"]
    if a.method == "ce":
        from . import finetune as m
        m.load_table = patched
        m.main(args + ["--model", "convnext_t", "--loss", "ce"])
        run = runs / "convnext_t_ce_s0"
    else:
        from . import robust_train as m
        m.load_table = patched
        m.main(args + ["--backbone", "convnext_t", "--objective", "groupdro"])
        run = next(p for p in runs.iterdir() if p.is_dir() and "groupdro" in p.name)
    Z = np.load(run / "logits.npy").astype(np.float32)
    assert Z.shape[0] == len(t0), Z.shape
    y, src = t0.y.to_numpy(), t0.source.to_numpy()
    ch = t0.split.to_numpy() == "test_ood"
    row = dict(split=a.k, method=f"convnext_{a.method}", test_sources="|".join(test), **evaluate(Z, y, src, sp, sub, te, ch, test))
    pd.DataFrame([row]).to_csv(out / f"k{a.k}_{a.method}.csv", index=False)
    print(a.k, a.method, round(row["test_bacc"], 3), round(row["ece_ts"], 3), flush=True)


if __name__ == "__main__":
    main()
