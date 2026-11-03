"""Gate 11: the leading pipeline per encoder, then late fusion across encoders. FLAIR is left out: its pretraining set holds test sources.
Per encoder and fold: bag of heads that each leave out one more site, between-site projection, logits at unit scale. Saved for offline fusion. Leave-one-development-source-out; reads no test row.
python -m experiments.strata_gate11 --bench data/bench --features runs/features --zeroshot runs/medsiglip --tokens data/strata --out results/strata/gate11
python -m experiments.strata_gate11 --bench data/bench --zeroshot runs/medsiglip --out results/strata/gate11 --fuse
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_table
from .shoal import K, load_x, metrics
from .strata_gate4 import BANKS
from .strata_gate7 import one, scaled
from .strata_gate10 import mean_tokens

ENCODERS = ["medsiglip", "medsiglip_last", "retfound_dinov2", "retfound_mae", "dinov2_b14", "siglip2_l16", "biomedclip", "bioclip2_l14", "convnext_t"]
SETS = {
    "medsiglip": ["medsiglip"],
    "medsiglip_both": ["medsiglip", "medsiglip_last"],
    "plus_retfound": ["medsiglip", "medsiglip_last", "retfound_dinov2", "retfound_mae"],
    "plus_dinov2": ["medsiglip", "medsiglip_last", "dinov2_b14"],
    "all": ENCODERS,
}


def inputs(a, t, name):
    if name != "medsiglip_last":
        return load_x(a.features, name, t)
    v, tr = mean_tokens(a.tokens)[:, 2], t.split.to_numpy() == "train"
    return (v - v[tr].mean(0)) / (v[tr].std(0) + 1e-6)


def score(z, y):
    pred = z.argmax(1)
    return dict(**metrics(z.astype(np.float32), y), recall_G=(pred[y == 2] == 2).mean() if (y == 2).any() else np.nan, false_G=(pred[y != 2] == 2).mean() if (y != 2).any() else np.nan)


def fuse(a, t, y, sp, src):
    """Fixed encoder sets, and a greedy set chosen for each held-out site from the other sites alone."""
    out, rows = Path(a.out), []
    sites = sorted(set(src[sp == "train"]))
    held = {s: np.isin(sp, ["train", "validation"]) & (src == s) for s in sites}
    text = {}
    for s in sites:
        va = (sp == "validation") & (src != s)
        text[s] = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32), va) for b in BANKS], 0)[held[s]]
    have = [e for e in ENCODERS if all((out / f"{e}_{s}_s{seed}.npy").exists() for s in sites for seed in a.seeds)]
    print("encoders", have, flush=True)
    for seed in a.seeds:
        Z = {e: {s: np.load(out / f"{e}_{s}_s{seed}.npy") for s in sites} for e in have}

        def blend(es, s):
            return .5 * np.mean([Z[e][s] for e in es], 0) + .5 * text[s]

        for s in sites:
            yy = y[held[s]]
            for name, es in SETS.items():
                if set(es) <= set(have):
                    rows.append(dict(seed=seed, source=s, arm=name, **score(blend(es, s), yy)))
            chosen = ["medsiglip"]
            val = lambda es: np.mean([score(blend(es, o), y[held[o]])["bacc"] for o in sites if o != s])
            best = val(chosen)
            while True:
                c = max((e for e in have if e not in chosen), key=lambda e: val(chosen + [e]), default=None)
                if c is None or val(chosen + [c]) <= best + .002:
                    break
                chosen.append(c)
                best = val(chosen)
            rows.append(dict(seed=seed, source=s, arm="greedy_nested", chosen="+".join(chosen), **score(blend(chosen, s), yy)))
        pd.DataFrame(rows).to_csv(out / "fusion.csv", index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G", "false_G"]].mean().round(3).sort_values("bacc").to_string())
    print(d.pivot_table(index="source", columns="arm", values="bacc").round(3).to_string())
    print(d[d.arm == "greedy_nested"].groupby(["source", "chosen"]).size().to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features")
    ap.add_argument("--zeroshot", required=True)
    ap.add_argument("--tokens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--fuse", action="store_true")
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    if a.fuse:
        return fuse(a, t, y, sp, src)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ENCODERS:
        X = inputs(a, t, name)
        for s in sorted(set(src[sp == "train"])):
            tr, va, held = (sp == "train") & (src != s), (sp == "validation") & (src != s), np.isin(sp, ["train", "validation"]) & (src == s)
            others = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
            for seed in a.seeds:
                f = out / f"{name}_{s}_s{seed}.npy"
                if not f.exists():
                    np.save(f, np.mean([scaled(one(X, y, src, tr & (src != o), va & (src != o), "ce", 32, seed), va)[held] for o in others], 0))
            print(name, s, flush=True)


if __name__ == "__main__":
    main()
