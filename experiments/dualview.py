"""Gate 13: whole image plus a magnified optic-disc view, since glaucoma is read at the disc. The disc is located with no labels (experiments.strata_tokens.locate_disc).
The CoREx idea here gets what gate 5 lacked: the fusion gate reads the image, w(x) = softmax(w0 + A phi(x)), with phi a low PCA of the feature, fitted on rows of sites its heads never saw.
An oracle arm picks per-site class offsets with the labels of the held-out site. It is a ceiling for label-free correction of class balance and threshold, never a method. Leave-one-development-source-out; reads no test row.
python -m experiments.dualview extract --bench data/bench --images data/processed --out data/strata
python -m experiments.dualview dev --bench data/bench --features runs/features --zeroshot runs/medsiglip --disc data/strata --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score

from .data import load_table
from .shoal import K, load_x, metrics
from .strata_gate4 import BANKS
from .strata_gate7 import one, scaled
from .strata_gate9 import compose
from .strata_tokens import DISC, locate_disc

HALF = 2.5  # half the crop side, in disc radii
PCS = 8


class DiscCrop:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, im):
        x, y = locate_disc(np.asarray(im.resize((224, 224), Image.BILINEAR)))
        W, H = im.size
        r = HALF * DISC * W
        cx, cy = x / 224 * W, y / 224 * H
        return self.processor(images=im.crop((int(cx - r), int(cy - r), int(cx + r), int(cy + r))), return_tensors="pt")["pixel_values"][0]


def extract(a):
    from .data import ImageTable
    from .followup import Medsiglip
    from .medsiglip_zeroshot import tensor
    from .prompts import BANKS as ALL, bank
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t, classes = load_table(a.bench)
    sg = Medsiglip().to(dev).eval()
    dl = torch.utils.data.DataLoader(ImageTable(t, a.images, DiscCrop(sg.proc.image_processor)), 32, num_workers=7)
    out, fs, n = Path(a.out), [], 0
    with torch.no_grad():
        for b, (x, _) in enumerate(dl):
            fs.append(tensor(sg.m.get_image_features(pixel_values=x.to(dev))).float().cpu().numpy())
            n += len(x)
            if b % 50 == 0:
                print(n, len(t), flush=True)
        X = np.concatenate(fs).astype(np.float32)
        np.save(out / "disc_medsiglip.npy", X)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        scale = sg.m.logit_scale.exp().item()
        for name in ALL:
            W = []
            for c in classes:
                tok = sg.proc.tokenizer(bank(name, c), padding="max_length", truncation=True, return_tensors="pt").to(dev)
                e = tensor(sg.m.get_text_features(**tok)).float()
                e = (e / e.norm(dim=1, keepdim=True)).mean(0)
                W.append((e / e.norm()).cpu().numpy())
            np.save(out / f"disc_logits_{name}.npy", scale * Xn @ np.stack(W).T)
    print("[ok]", n, flush=True)


def fit_gate(pools, b, shrink=.05, steps=400, seed=0):
    """pools: (Z (n, b, K), phi (n, PCS), y). Starts at the plain average, shrunk towards it."""
    torch.manual_seed(seed)
    w0, A = torch.zeros(b, requires_grad=True), torch.zeros(b, PCS, requires_grad=True)
    opt = torch.optim.Adam([w0, A], lr=.02)
    for _ in range(steps):
        loss = shrink * ((w0 ** 2).sum() + (A ** 2).sum())
        for Z, phi, y in pools:
            w = torch.softmax(w0 + torch.tensor(phi) @ A.T, 1)
            z = (torch.tensor(Z) * w[:, :, None]).sum(1)
            yt = torch.tensor(y)
            loss = loss + torch.stack([torch.nn.functional.cross_entropy(z[yt == c], yt[yt == c]) for c in np.unique(y)]).mean() / len(pools)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w0.detach(), A.detach()


def oracle(z, y):
    """Best class offsets for this site, found with its labels: the ceiling of any class-balance correction."""
    best, grid = (balanced_accuracy_score(y, z.argmax(1)), 0, 0), np.linspace(-3, 3, 25)
    for a in grid:
        for b in grid:
            v = balanced_accuracy_score(y, (z + np.array([a, b, 0.])).argmax(1))
            if v > best[0]:
                best = (v, a, b)
    return best[0]


def score(z, y):
    pred = z.argmax(1)
    return dict(**metrics(z.astype(np.float32), y), recall_G=(pred[y == 2] == 2).mean() if (y == 2).any() else np.nan, false_G=(pred[y != 2] == 2).mean() if (y != 2).any() else np.nan)


def dev_run(a):
    t, _ = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr0, dv = sp == "train", np.isin(sp, ["train", "validation"])
    X = load_x(a.features, "medsiglip", t)
    D = np.load(Path(a.disc) / "disc_medsiglip.npy").astype(np.float32)
    D = (D - D[tr0].mean(0)) / (D[tr0].std(0) + 1e-6)
    rows, cm = [], []
    for s in sorted(set(src[tr0])):
        tr, va, held = tr0 & (src != s), (sp == "validation") & (src != s), dv & (src == s)
        T = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32), va) for b in BANKS], 0)
        Td = np.mean([scaled(np.load(Path(a.disc) / f"disc_logits_{b}.npy").astype(np.float32), va) for b in BANKS], 0)
        U, sv, Vt = np.linalg.svd(X[tr][::5] - X[tr][::5].mean(0), full_matrices=False)
        phi = (X - X[tr].mean(0)) @ Vt[:PCS].T
        phi = phi / phi[tr].std(0)
        others = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
        yy = y[held]
        for seed in a.seeds:
            zg, zd, pools = [], [], []
            for o in others:
                m, vo = tr & (src != o), va & (src != o)
                g, d = scaled(one(X, y, src, m, vo, "ce", 32, seed), va), scaled(one(D, y, src, m, vo, "ce", 32, seed), va)
                zg.append(g[held])
                zd.append(d[held])
                r = dv & (src == o)
                pools.append((np.stack([g[r], T[r], d[r], Td[r]], 1), phi[r].astype(np.float32), y[r]))
            g, d = np.mean(zg, 0), np.mean(zd, 0)
            leader, disc = .5 * g + .5 * T[held], .5 * d + .5 * Td[held]
            two = .5 * leader + .5 * disc
            w0, A = fit_gate(pools, 4, seed=seed)
            w = torch.softmax(w0 + torch.tensor(phi[held].astype(np.float32)) @ A.T, 1)
            gated = (torch.tensor(np.stack([g, T[held], d, Td[held]], 1).astype(np.float32)) * w[:, :, None]).sum(1).numpy()
            res = {"leader": leader, "disc_head": d, "disc_text": Td[held], "disc": disc, "two_views": two, "g_two_nd_leader": compose(two, leader), "g_disc_nd_leader": compose(disc, leader), "image_gate": gated}
            for arm, z in res.items():
                rows.append(dict(seed=seed, source=s, arm=arm, **score(z, yy)))
            for arm in ["leader", "two_views"]:
                rows.append(dict(seed=seed, source=s, arm=arm + "_oracle", bacc=oracle(res[arm], yy)))
            for i in range(K):
                for j in range(K):
                    cm.append(dict(seed=seed, source=s, true=i, pred=j, n=int(((yy == i) & (leader.argmax(1) == j)).sum())))
            print(s, seed, flush=True)
            pd.DataFrame(rows).to_csv(Path(a.out) / "gate13.csv", index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G", "false_G"]].mean().round(3).sort_values("bacc").to_string())
    print(d.pivot_table(index="source", columns="arm", values="bacc").round(3).T.to_string())
    c = pd.DataFrame(cm)
    print(c[c.seed == a.seeds[0]].pivot_table(index=["source", "true"], columns="pred", values="n").to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    for k in ["bench", "images", "out"]:
        e.add_argument("--" + k, required=True)
    c = sub.add_parser("dev")
    for k in ["bench", "features", "zeroshot", "disc", "out"]:
        c.add_argument("--" + k, required=True)
    c.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)
    (extract if a.cmd == "extract" else dev_run)(a)


if __name__ == "__main__":
    main()
