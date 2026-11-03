"""Can the answer layer of a multimodal LLM be repaired? Its multiple-choice letter scores (mean over six option orders) are recalibrated on Cal-A with
temperature only (plain NLL), or with class offsets (bias) or a full 3x3 affine map (class-balanced NLL); test and Chaksu are only read.
python -m experiments.mcq_recal --bench data/bench --runs runs --out results/review/mcq_recal.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .calib import ece, nll, softmax
from .data import load_table
from .full_eval import load_logits


def fit(Z, y, kind, steps=500):
    Zt, yt = torch.tensor(Z, dtype=torch.float32), torch.tensor(y)
    w = torch.tensor(len(y) / (3 * np.bincount(y, minlength=3)), dtype=torch.float32)  # classes weighted equally, as bAcc does
    W = torch.eye(3, requires_grad=kind == "affine")
    s = torch.zeros(1, requires_grad=True)
    b = torch.zeros(3, requires_grad=kind in ("bias", "affine"))
    ps = [p for p in (W, s, b) if p.requires_grad]
    opt = torch.optim.LBFGS(ps, max_iter=steps, line_search_fn="strong_wolfe")

    def step():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Zt @ W.T * s.exp() + b, yt, weight=w if kind != "ts" else None)
        loss.backward()
        return loss
    opt.step(step)
    return lambda Z_: (torch.tensor(Z_, dtype=torch.float32) @ W.T * s.exp() + b).detach().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "runs", "out"]:
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, sub = t.split.to_numpy(), t.y.to_numpy(), t.cal_subset.to_numpy()
    ca, te, ch = (sp == "calibration") & (sub == "cal_A"), sp == "test", sp == "test_ood"
    rows = []
    for f in sorted((Path(a.runs) / "vlm").glob("logits_*.npy")):
        Z = load_logits(f, "mcq", t)
        for kind in ["none", "ts", "bias", "affine"]:
            g = (lambda Z_: Z_) if kind == "none" else fit(Z[ca], y[ca], kind)
            for view, m in [("test", te), ("chaksu", ch)]:
                Zm = g(Z[m])
                P, pred = softmax(Zm), Zm.argmax(1)
                r = dict(model=f.stem[7:], recal=kind, view=view, bacc=balanced_accuracy_score(y[m], pred), ece=ece(P, y[m])[0], nll=nll(P, y[m]))
                for c, cn in enumerate(classes):
                    r[f"recall_{cn}"] = float((pred[y[m] == c] == c).mean()) if (y[m] == c).any() else np.nan
                rows.append(r)
    d = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
