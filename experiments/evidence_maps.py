"""Patch-level evidence maps for LANTERN. The head input is a mean of patch tokens and every later step is linear, so a head's logits split exactly into
per-patch contributions: z_s(x) = (1/P) sum_p W_s (I - U_s U_s^T) x_p + b_s. Text scores are global and are not mapped.
python -m experiments.evidence_maps --bench data/bench --images data/processed --strata data/strata --zeroshot runs/medsiglip --runs runs --out paper/figures/evidence.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .data import load_table
from .rptoken import DIM, GRID
from .scope import fit_head, label_sets
from .shoal import K
from .strata_gate4 import nap


def tokens(paths, dev="cpu"):
    """Patch tokens exactly as experiments.rptoken.extract stores them: last layer, 14 x 14 grid, fixed 384-d map."""
    from .followup import Medsiglip
    sg = Medsiglip().to(dev).eval()
    proc = sg.proc.image_processor
    out, R = [], None
    with torch.no_grad():
        for p in paths:
            x = proc(images=Image.open(p).convert("RGB"), return_tensors="pt")["pixel_values"].to(dev)
            h = sg.m.vision_model(pixel_values=x, output_hidden_states=True).hidden_states[-1]
            G = int(h.shape[1] ** .5)
            v = torch.nn.functional.adaptive_avg_pool2d(h.reshape(1, G, G, -1).permute(0, 3, 1, 2), GRID).permute(0, 2, 3, 1)
            if R is None:
                R = torch.linalg.qr(torch.randn(v.shape[-1], DIM, generator=torch.Generator().manual_seed(0)))[0].to(v)
            out.append((v @ R)[0].float().numpy().astype(np.float16).astype(np.float32))
    return np.stack(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "images", "strata", "zeroshot", "runs", "out"]:
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, src, scope = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.normal_scope.to_numpy()
    M = np.load(Path(a.strata) / "mean3_medsiglip.npy")[:, 2].astype(np.float32)
    tr, va = sp == "train", sp == "validation"
    mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-6
    X = (M - mu) / sd
    A = label_sets(y, scope, "exact")
    heads = []
    for o in [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]:
        m, vo = tr & (src != o), va & (src != o)
        U = nap(X[m], y[m], src[m], 32)
        Q = X - (X @ U.T) @ U
        h = fit_head(Q[m], A[m], y[m], 1 / np.bincount(y[m], minlength=K)[y[m]], Q[vo], y[vo], 0)
        W = h.weight.detach().numpy()
        z = Q[va] @ W.T
        heads.append((W @ (np.eye(len(U.T)) - U.T @ U), (z - z.mean(1, keepdims=True)).std()))
    # images: for each glaucoma test source, the most and least confident glaucoma image under LANTERN seed 0
    Z = np.load(Path(a.runs) / "top_dev" / "last_reg" / "logits_s0.npy")
    pg = np.exp(Z - Z.max(1, keepdims=True))
    pg = pg[:, 2] / pg.sum(1)
    picks = []
    for s in ["hygd", "papila", "fives", "chaksu"]:
        m = np.flatnonzero(np.isin(sp, ["test", "test_ood"]) & (src == s) & (y == 2))
        picks += [m[np.argmax(pg[m])], m[np.argmin(pg[m])]]
    paths = [Path(a.images) / src[i] / f"{t.image_id.iloc[i]}.jpg" for i in picks]
    T = tokens(paths)
    print("token mean against stored input, max abs diff:", float(np.abs(T.mean((1, 2)) - M[picks]).max()))
    fig, axs = plt.subplots(2, len(picks), figsize=(6.6, 1.9))
    for j, (i, tok) in enumerate(zip(picks, T)):
        x = (tok - mu) / sd
        ev = np.zeros((GRID, GRID))
        for W, scale in heads:
            c = x @ W.T
            c = (c - c.mean(-1, keepdims=True)) / scale
            ev += (c[..., 2] - np.maximum(c[..., 0], c[..., 1])) / len(heads)
        img = np.asarray(Image.open(paths[j]).convert("RGB").resize((448, 448)))
        axs[0, j].imshow(img)
        axs[1, j].imshow(img)
        lim = np.abs(ev).max()
        axs[1, j].imshow(np.kron(ev, np.ones((32, 32))), cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=.55)
        name = {"hygd": "HYGD", "papila": "PAPILA", "fives": "FIVES", "chaksu": "Chákṣu"}[src[i]]
        axs[0, j].set_title(f"{name} {pg[i]:.2f}"[:-3] + f"{pg[i]:.2f}"[1:] if False else f"{name}\np(G)={pg[i]:.2f}", fontsize=5.2, pad=1.2)
        for ax in axs[:, j]:
            ax.axis("off")
    fig.subplots_adjust(wspace=.03, hspace=.04)
    fig.savefig(a.out, bbox_inches="tight", dpi=200)
    print("[ok]", a.out)


if __name__ == "__main__":
    main()
