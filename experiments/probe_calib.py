"""Frozen-backbone benchmark: linear heads x training losses x post-hoc calibrators x seeds.

Protocol (identical for every backbone):
  features: frozen embeddings from experiments.extract_features, standardised on train
  head: linear layer, AdamW (lr 1e-3, wd 1e-4), 40 epochs, batch 256; --weightings: none (uniform), class, class_source
  model selection: epoch with best validation balanced accuracy (validation is never used for calibration)
  post-hoc: none / TS / FTS fitted on calibration cal_A
  referral threshold: max-probability threshold reaching <= 5% error on calibration cal_B
  reporting: test (7 held-out collections), test_ood (Chákṣu), test_openset referral; 3 seeds;
             95% bootstrap CIs over leakage groups (seed 0)
python -m experiments.probe_calib --bench data/bench --features runs/features --out runs/probe_calib [--models ...]
Variants, same protocol: --head mlp (one hidden layer, 512 units); --shots 16 (16 images per class, one per leakage group, redrawn per seed).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from .calib import loss_fn
from .coop import sample_shots
from .data import balanced_weights, load_table
from .evaluator import evaluate

EVAL = ["validation", "test", "test_ood"]


def make_head(kind, d, k):
    if kind == "mlp":
        return torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(512, k))
    return torch.nn.Linear(d, k)


def train_head(X, y, w, Xv, yv, loss, seed, epochs=40, bs=256, lr=1e-3, kind="linear", wd=1e-4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = make_head(kind, X.shape[1], int(y.max()) + 1)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    f = loss_fn(loss)
    Xt, yt, Xvt = torch.tensor(X), torch.tensor(y), torch.tensor(Xv)
    best, state = -1.0, None
    for _ in range(epochs):
        idx = rng.choice(len(y), size=len(y), replace=True, p=w / w.sum())
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            opt.zero_grad()
            f(head(Xt[b]), yt[b]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            b = balanced_accuracy_score(yv, head(Xvt).argmax(1).numpy())
        head.train()
        if b > best:
            best, state = b, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(state)
    return head.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--losses", nargs="*", default=["ce", "focal", "ls", "mbls", "mdca"])
    ap.add_argument("--weightings", nargs="*", default=["none"], choices=["none", "class", "class_source"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--shots", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, classes = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr = sp == "train"
    W = {"none": np.ones(tr.sum()), "class": balanced_weights(y[tr].tolist()),
         "class_source": balanced_weights(zip(src[tr], y[tr]))}
    models = a.models or sorted(p.stem for p in Path(a.features).glob("*.npy"))
    res_file = out / "results.jsonl"
    done = set()
    if res_file.exists():
        done = {(r["model"], r["loss"], r["weighting"], r["seed"]) for r in map(json.loads, res_file.read_text().splitlines())}
    va = sp == "validation"
    for name in models:
        X = np.load(Path(a.features) / f"{name}.npy").astype(np.float32)
        X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
        for wt in a.weightings:
            for loss in a.losses:
                for seed in a.seeds:
                    zf = out / f"logits_{name}_{wt}_{loss}_s{seed}.npy"
                    if (name, loss, wt, seed) in done and zf.exists():
                        continue
                    if a.shots:
                        idx = sample_shots(t, seed, a.shots)
                        head = train_head(X[idx], y[idx], np.ones(len(idx)), X[va], y[va], loss, seed, epochs=300, kind=a.head)
                    else:
                        head = train_head(X[tr], y[tr], W[wt], X[va], y[va], loss, seed, kind=a.head)
                    with torch.no_grad():
                        Z = head(torch.tensor(X)).numpy()
                    np.save(zf, Z)
                    if (name, loss, wt, seed) in done:      # result row exists from before logits were kept
                        continue
                    setting = f"{a.shots}-shot-probe" if a.shots else ("mlp-probe" if a.head == "mlp" else "linear-probe")
                    meta = {"setting": setting, "model": name, "loss": loss, "weighting": wt, "seed": seed}
                    rows = evaluate(Z, t, meta, n_boot=a.n_boot if seed == 0 else 0)
                    with open(res_file, "a") as fh:
                        fh.writelines(json.dumps(r) + "\n" for r in rows)
                    print(name, wt, loss, f"s{seed} test bAcc {rows[0]['test']['bacc']:.3f} "
                          f"ECE raw/ts {rows[0]['test']['ece']:.3f}/{rows[1]['test']['ece']:.3f}", flush=True)


if __name__ == "__main__":
    main()
