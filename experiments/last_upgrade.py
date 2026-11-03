"""Upgrades of last_reg on development sources: mirror views, all 1152 token dimensions, NAP k, weight decay. Leave-one-development-source-out; no test row is scored.
python -m experiments.last_upgrade extract --bench data/bench --images data/processed --out data/strata
python -m experiments.last_upgrade dev --bench data/bench --zeroshot runs/medsiglip --strata data/strata --out results/strata/last_upgrade
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import balanced_weights, load_table
from .flipview import Mirror
from .probe_calib import train_head
from .shoal import K, metrics
from .strata_gate4 import BANKS, nap
from .strata_gate7 import scaled


def arms():
    """(dim, mirror, k, wd). The first is last_reg itself."""
    a = [(384, False, 32, 1e-2), (384, True, 32, 1e-2), (1152, False, 32, 1e-2), (1152, True, 32, 1e-2)]
    return a + [(384, True, k, wd) for k in (8, 16, 32) for wd in (1e-3, 1e-2, 1e-1) if (k, wd) != (32, 1e-2)]


def extract(a):
    from .data import ImageTable
    from .followup import Medsiglip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t, _ = load_table(a.bench)
    sg = Medsiglip().to(dev).eval()
    proc = sg.proc.image_processor
    plain = lambda im: proc(images=im, return_tensors="pt")["pixel_values"][0]
    for name, tf in [("last1152_plain", plain), ("last1152_mirror", Mirror(proc))]:
        out, n, fs = Path(a.out) / f"{name}.npy", 0, []
        if out.exists():
            continue
        dl = torch.utils.data.DataLoader(ImageTable(t, a.images, tf), 32, num_workers=7)
        with torch.no_grad():
            for b, (x, _) in enumerate(dl):
                h = sg.m.vision_model(pixel_values=x.to(dev), output_hidden_states=True).hidden_states[-1]
                fs.append(h.mean(1).float().cpu().numpy())
                n += len(x)
                if b % 100 == 0:
                    print(name, n, len(t), flush=True)
        np.save(out, np.concatenate(fs))
        print("[ok]", name, n, flush=True)


def dev_run(a):
    t, _ = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr0, dv, n = sp == "train", np.isin(sp, ["train", "validation"]), len(t)
    S, out = Path(a.strata), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P, M = np.load(S / "last1152_plain.npy").astype(np.float32), np.load(S / "last1152_mirror.npy").astype(np.float32)
    R = torch.linalg.qr(torch.randn(1152, 384, generator=torch.Generator().manual_seed(0)))[0].numpy()
    # the same map as experiments.rptoken: this row must be near zero, or the 384 arm is not last_reg
    print("gap to stored last_reg input", float(np.abs(P[:200] @ R - np.load(S / "mean3_medsiglip.npy")[:200, 2]).mean()), flush=True)
    left = np.load(S / "disc.npy")[:, 0] < .5
    inputs = {}
    for d in (384, 1152):
        p, m = (P @ R, M @ R) if d == 384 else (P, M)
        mu, sd = p[tr0].mean(0), p[tr0].std(0) + 1e-6
        inputs[d] = ((p - mu) / sd).astype(np.float32), ((m - mu) / sd).astype(np.float32)
    rows = []
    for s in sorted(set(src[tr0])):
        tr, va, held = tr0 & (src != s), (sp == "validation") & (src != s), dv & (src == s)
        T = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32), va) for b in BANKS], 0)
        Tf = np.mean([scaled(np.load(S / f"mirror_logits_{b}.npy").astype(np.float32), va) for b in BANKS], 0)
        others = [o for o in sorted(set(src[tr])) if len(np.unique(y[tr & (src != o)])) == K]
        yy, ll, N = y[held], left[held], y[held] == 0
        for seed in a.seeds:
            for d, mirror, k, wd in arms():
                name = f"d{d}_{'mirror' if mirror else 'plain'}_k{k}_wd{wd:g}"
                f = out / f"{name}_{s}_s{seed}.npy"
                X, F = inputs[d]
                A, ya, sa = (np.concatenate([X, F]), np.concatenate([y, y]), np.concatenate([src, src])) if mirror else (X, y, src)
                zs = []
                for o in others:
                    m, vo = tr & (src != o), va & (src != o)
                    m2, v2 = (np.concatenate([m, m]), np.concatenate([vo, vo])) if mirror else (m, vo)
                    U = nap(A[m2], ya[m2], sa[m2], k)
                    Q = A - (A @ U.T) @ U
                    h = train_head(Q[m2], ya[m2], balanced_weights(ya[m2].tolist()), Q[v2], ya[v2], "ls", seed, epochs=12, bs=64, lr=3e-4, wd=wd)
                    with torch.no_grad():
                        z = h(torch.tensor(Q)).numpy()
                    zs.append(.5 * (scaled(z[:n], va) + scaled(z[n:], va)) if mirror else scaled(z, va))
                z = .5 * np.mean(zs, 0) + .5 * (.5 * (T + Tf) if mirror else T)
                np.save(f, z.astype(np.float32))  # every row, for calibration rows of other sites in the post-processing check
                zh = z[held]
                pred = zh.argmax(1)
                rows.append(dict(seed=seed, source=s, arm=name, **metrics(zh.astype(np.float32), yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan, false_G=(pred[yy != 2] == 2).mean() if (yy != 2).any() else np.nan, N_G_left=(pred[N & ll] == 2).mean() if (N & ll).any() else np.nan, N_G_right=(pred[N & ~ll] == 2).mean() if (N & ~ll).any() else np.nan))
            print(s, seed, flush=True)
            pd.DataFrame(rows).to_csv(out / "dev.csv", index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G", "false_G", "N_G_left", "N_G_right"]].mean().round(3).sort_values("bacc").to_string())
    print(d.pivot_table(index="arm", columns="source", values="bacc").round(3).to_string())
    print(d.pivot_table(index="arm", columns="seed", values="bacc").round(3).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    for k in ["bench", "images", "out"]:
        e.add_argument("--" + k, required=True)
    c = sub.add_parser("dev")
    for k in ["bench", "zeroshot", "strata", "out"]:
        c.add_argument("--" + k, required=True)
    c.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)
    (extract if a.cmd == "extract" else dev_run)(a)


if __name__ == "__main__":
    main()
