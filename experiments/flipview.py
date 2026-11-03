"""Gate 14: the leading pipeline reads N as G more often when the disc lies on the left of the image, on every development site, though flipping an image cannot change its disease.
Arms: text averaged over an image and its mirror; heads trained on both and averaged over both; every image mirrored so its disc lies on the right. The disc side comes from experiments.strata_tokens.locate_disc, with no labels. Leave-one-development-source-out; reads no test row.
python -m experiments.flipview extract --bench data/bench --images data/processed --out data/strata
python -m experiments.flipview dev --bench data/bench --features runs/features --zeroshot runs/medsiglip --strata data/strata --out results/strata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import ImageOps

from .data import load_table
from .shoal import K, metrics
from .strata_gate4 import BANKS
from .strata_gate7 import one, scaled


class Mirror:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, im):
        return self.processor(images=ImageOps.mirror(im), return_tensors="pt")["pixel_values"][0]


def extract(a):
    from .data import ImageTable
    from .followup import Medsiglip
    from .medsiglip_zeroshot import tensor
    from .prompts import BANKS as ALL, bank
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t, classes = load_table(a.bench)
    sg = Medsiglip().to(dev).eval()
    dl = torch.utils.data.DataLoader(ImageTable(t, a.images, Mirror(sg.proc.image_processor)), 32, num_workers=7)
    out, fs, n = Path(a.out), [], 0
    with torch.no_grad():
        for b, (x, _) in enumerate(dl):
            fs.append(tensor(sg.m.get_image_features(pixel_values=x.to(dev))).float().cpu().numpy())
            n += len(x)
            if b % 50 == 0:
                print(n, len(t), flush=True)
        X = np.concatenate(fs).astype(np.float32)
        np.save(out / "mirror_medsiglip.npy", X)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        scale = sg.m.logit_scale.exp().item()
        for name in ALL:
            W = []
            for c in classes:
                tok = sg.proc.tokenizer(bank(name, c), padding="max_length", truncation=True, return_tensors="pt").to(dev)
                e = tensor(sg.m.get_text_features(**tok)).float()
                e = (e / e.norm(dim=1, keepdim=True)).mean(0)
                W.append((e / e.norm()).cpu().numpy())
            np.save(out / f"mirror_logits_{name}.npy", scale * Xn @ np.stack(W).T)
    print("[ok]", n, flush=True)


def dev_run(a):
    t, _ = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr0, dv = sp == "train", np.isin(sp, ["train", "validation"])
    n = len(t)
    R = np.load(Path(a.features) / "medsiglip.npy").astype(np.float32)
    M = np.load(Path(a.strata) / "mirror_medsiglip.npy").astype(np.float32)
    mu, sd = R[tr0].mean(0), R[tr0].std(0) + 1e-6
    X, F = (R - mu) / sd, (M - mu) / sd
    left = np.load(Path(a.strata) / "disc.npy")[:, 0] < .5
    C = np.where(left[:, None], F, X)  # disc on the right everywhere
    X2, y2, src2 = np.concatenate([X, F]), np.concatenate([y, y]), np.concatenate([src, src])
    rows = []
    for s in sorted(set(src[tr0])):
        tr, va, held = tr0 & (src != s), (sp == "validation") & (src != s), dv & (src == s)
        T = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32), va) for b in BANKS], 0)
        Tf = np.mean([scaled(np.load(Path(a.strata) / f"mirror_logits_{b}.npy").astype(np.float32), va) for b in BANKS], 0)
        Tavg, Tc = .5 * (T + Tf), np.where(left[:, None], Tf, T)
        others = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
        yy, ll = y[held], left[held]
        for seed in a.seeds:
            base, both, canon = [], [], []
            for o in others:
                m, vo = tr & (src != o), va & (src != o)
                base.append(scaled(one(X, y, src, m, vo, "ce", 32, seed), va)[held])
                z = one(X2, y2, src2, np.concatenate([m, m]), np.concatenate([vo, vo]), "ce", 32, seed)
                both.append(.5 * (scaled(z[:n], va)[held] + scaled(z[n:], va)[held]))
                canon.append(scaled(one(C, y, src, m, vo, "ce", 32, seed), va)[held])
            base, both, canon = np.mean(base, 0), np.mean(both, 0), np.mean(canon, 0)
            res = {"leader": .5 * base + .5 * T[held], "text_mirror": .5 * base + .5 * Tavg[held], "both_views": .5 * both + .5 * Tavg[held], "both_head_only": both, "canonical": .5 * canon + .5 * Tc[held], "canonical_head_only": canon}
            for arm, z in res.items():
                pred = z.argmax(1)
                N = yy == 0
                rows.append(dict(seed=seed, source=s, arm=arm, **metrics(z.astype(np.float32), yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan, false_G=(pred[yy != 2] == 2).mean() if (yy != 2).any() else np.nan, N_G_left=(pred[N & ll] == 2).mean() if (N & ll).any() else np.nan, N_G_right=(pred[N & ~ll] == 2).mean() if (N & ~ll).any() else np.nan))
            print(s, seed, flush=True)
            pd.DataFrame(rows).to_csv(Path(a.out) / "gate14.csv", index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G", "false_G", "N_G_left", "N_G_right"]].mean().round(3).sort_values("bacc").to_string())
    print(d.pivot_table(index="source", columns="arm", values="bacc").round(3).T.to_string())
    print(d.pivot_table(index="source", columns="arm", values="false_G").round(3).T.to_string())
    print(d.pivot_table(index="seed", columns="arm", values="bacc").round(3).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    for k in ["bench", "images", "out"]:
        e.add_argument("--" + k, required=True)
    c = sub.add_parser("dev")
    for k in ["bench", "features", "zeroshot", "strata", "out"]:
        c.add_argument("--" + k, required=True)
    c.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)
    (extract if a.cmd == "extract" else dev_run)(a)


if __name__ == "__main__":
    main()
