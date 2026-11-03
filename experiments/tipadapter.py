"""Tip-Adapter and Tip-Adapter-F (Zhang et al., ECCV 2022), official recipe on the same 16-shot images as CoOp, from cached features.

logits = scale f W^T + alpha exp(-beta (1 - f K^T)) V. Tip-Adapter-F fine-tunes the keys K for 20 epochs (AdamW lr 1e-3).
(alpha, beta) by the official grid (beta 0.1..7, alpha 0.1..3) and the prompt bank on validation only: balanced accuracy (benchmark rule) or --select acc (official rule).
python -m experiments.tipadapter --bench data/bench --features runs/features --out runs/tipadapter --models siglip2_l16 clip_b16 biomedclip
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .backbones import build_clip
from .coop import sample_shots
from .data import load_table
from .evaluator import evaluate
from .prompts import BANKS
from .zeroshot_clip import text_weights


def logits(x, K, V, W, scale, alpha, beta):
    return scale * x @ W.T + alpha * torch.exp(-beta * (1 - x @ K.T)) @ V


def search(xv, yv, K, V, W, scale, select="bacc"):
    best = (-1.0, 1.0, 1.0)
    for beta in np.linspace(0.1, 7, 200):
        aff = torch.exp(-float(beta) * (1 - xv @ K.T)) @ V
        base = scale * xv @ W.T
        for alpha in np.linspace(0.1, 3, 20):
            ok = ((base + float(alpha) * aff).argmax(1) == yv).float()
            acc = ok.mean().item() if select == "acc" else torch.stack([ok[yv == c].mean() for c in yv.unique()]).mean().item()
            if acc > best[0]:
                best = (acc, float(alpha), float(beta))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=["siglip2_l16", "clip_b16", "biomedclip"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--select", choices=["bacc", "acc"], default="bacc")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t, classes = load_table(a.bench)
    y = t.y.to_numpy()
    va = (t.split == "validation").to_numpy()
    with open(out / "results.jsonl", "w") as fh:
        for name in a.models:
            X = np.load(Path(a.features) / f"{name}.npy").astype(np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            model, _, tok = build_clip(name)
            scale = model.logit_scale.exp().item()
            Ws = {b: torch.tensor(text_weights(model, tok, classes, b), device=dev) for b in BANKS}
            Xt, yt = torch.tensor(X, device=dev), torch.tensor(y, device=dev)
            xv, yv = Xt[va], yt[va]
            for seed in a.seeds:
                torch.manual_seed(seed)
                idx = sample_shots(t, seed)
                K0, V = Xt[idx], F.one_hot(yt[idx], len(classes)).float()
                bank = max(BANKS, key=lambda b: search(xv, yv, K0, V, Ws[b], scale, a.select)[0])
                W = Ws[bank]
                _, al, be = search(xv, yv, K0, V, W, scale, a.select)
                K = torch.nn.Parameter(K0.clone())
                opt = torch.optim.AdamW([K], lr=1e-3, eps=1e-4)
                for _ in range(20):
                    loss = F.cross_entropy(logits(K0, K, V, W, scale, al, be), yt[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    K.data = F.normalize(K.data, dim=-1)
                KF = K.detach()
                _, alF, beF = search(xv, yv, KF, V, W, scale, a.select)
                for arm, k, aa, bb in [("tip-adapter", K0, al, be), ("tip-adapter-f", KF, alF, beF)]:
                    with torch.no_grad():
                        Z = torch.cat([logits(c, k, V, W, scale, aa, bb) for c in Xt.split(4096)]).cpu().numpy()
                    np.save(out / f"logits_{arm}_{name}_s{seed}.npy", Z)
                    for r in evaluate(Z, t, {"setting": arm, "model": name, "loss": bank, "seed": seed, "alpha": aa, "beta": bb}, n_boot=0):
                        fh.write(json.dumps(r) + "\n")
                    print(arm, name, bank, seed, f"alpha {aa:.2f} beta {bb:.2f} val bAcc {r['validation']['bacc']:.3f}", flush=True)
                fh.flush()


if __name__ == "__main__":
    main()
