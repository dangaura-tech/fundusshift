"""Gate 9: capacity follows site support. A class seen in many sites (N, D) can afford the RPAttn token head; glaucoma, seen in two, is scored by the low-capacity branch only.
log p(G | x) = log sigma(g(x)), log p(c | x) = log(1 - sigma(g(x))) + log softmax_{N,D} r_c(x), with g = z_G - logsumexp(z_N, z_D) of the G branch and r the N/D logits of the N/D branch.
Leave-one-development-source-out, reads logits of experiments.rptoken; reads no test row.
python -m experiments.strata_gate9 --bench data/bench --features runs/features --zeroshot runs/medsiglip --rptoken results/strata/rptoken --out results/strata --seeds 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_table
from .shoal import K, load_x, metrics
from .strata_gate4 import BANKS
from .strata_gate7 import one, scaled

G = 2


def compose(z_g, z_nd):
    """Three-class log probabilities: the G decision from z_g, the N-against-D decision from z_nd."""
    zg, zn = torch.tensor(z_g), torch.tensor(z_nd)
    g = zg[:, G] - torch.logsumexp(zg[:, :G], 1)
    return torch.cat([torch.nn.functional.logsigmoid(-g)[:, None] + torch.log_softmax(zn[:, :G], 1), torch.nn.functional.logsigmoid(g)[:, None]], 1).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--zeroshot", required=True)
    ap.add_argument("--rptoken", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="medsiglip")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    X = load_x(a.features, a.model, t)
    rows = []
    for s in sorted(set(src[sp == "train"])):
        tr, va, held = (sp == "train") & (src != s), (sp == "validation") & (src != s), np.isin(sp, ["train", "validation"]) & (src == s)
        Tm = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_{a.model}_{b}.npy").astype(np.float32), va) for b in BANKS], 0)
        others = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
        for seed in a.seeds:
            bag = np.mean([scaled(one(X, y, src, tr & (src != o), va & (src != o), "ce", 32, seed), va)[held] for o in others], 0)
            text = Tm[held]
            leader = .5 * bag + .5 * text
            res = {"leader": leader, "g_text_nd_leader": compose(text, leader)}
            for arm in ["mean", "rp", "rp_mix"]:
                d = Path(a.rptoken) / f"{arm}_s{seed}"
                if not (d / f"{s}.npy").exists():
                    continue
                zh, zv = np.load(d / f"{s}.npy"), np.load(d / f"{s}_val.npy")
                r = (zh - zh.mean(1, keepdims=True)) / (zv - zv.mean(1, keepdims=True)).std()
                res[arm] = r
                res[f"{arm}+text"] = .5 * r + .5 * text
                res[f"g_leader_nd_{arm}"] = compose(leader, r)
                res[f"g_leader_nd_{arm}+text"] = compose(leader, .5 * r + .5 * text)
                res[f"g_text_nd_{arm}+text"] = compose(text, .5 * r + .5 * text)
            yy = y[held]
            for arm, z in res.items():
                pred = z.argmax(1)
                rows.append(dict(seed=seed, source=s, arm=arm, **metrics(z.astype(np.float32), yy), recall_G=(pred[yy == G] == G).mean() if (yy == G).any() else np.nan, false_G=(pred[yy != G] == G).mean() if (yy != G).any() else np.nan))
            print(s, seed, flush=True)
            pd.DataFrame(rows).to_csv(Path(a.out) / f"gate9{a.tag}.csv", index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G", "false_G"]].mean().round(3).sort_values("bacc").to_string())
    print(d.pivot_table(index="source", columns="arm", values="bacc").round(3).T.to_string())


if __name__ == "__main__":
    main()
