"""RPAttn head C rebuilt for the frozen patch tokens of a ViT: three depths of MedSigLIP take the place of the three CNN stages, and dilation grows with stage depth since all grids share one resolution. Conv path, pooled-key attention path, spatial gate and GeM tail as in the paper.
Tokens are pooled to 14 x 14 and sent through one fixed random orthonormal map to 384 dimensions, so nothing is fitted on any source. Training is leave-one-development-source-out and reads no test row.
python -m experiments.rptoken extract --bench data/bench --images data/processed --out data/strata
python -m experiments.rptoken train --bench data/bench --tokens data/strata --zeroshot runs/medsiglip --out results/strata/rptoken --arm rp
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .rpattn import SE, block
from .shoal import K, metrics
from .strata_gate4 import BANKS
from .strata_gate7 import scaled

GRID, DIM, LAYERS = 14, 384, 3


def mixstyle(x, alpha=.1):
    """Swaps channel statistics between images of a batch (Zhou et al., ICLR 2021)."""
    n = len(x)
    lam = torch.distributions.Beta(alpha, alpha).sample((n, 1, 1, 1)).to(x)
    mu, sd = x.mean((2, 3), keepdim=True), x.std((2, 3), keepdim=True) + 1e-6
    i = torch.randperm(n, device=x.device)
    return (x - mu) / sd * (lam * sd + (1 - lam) * sd[i]) + lam * mu + (1 - lam) * mu[i]


class TokenRPAttn(nn.Module):
    def __init__(self, d=DIM, D=256, heads=8, p=3.0, mix=False):
        super().__init__()
        self.paths = nn.ModuleList(nn.Sequential(block(d, D, k=1), block(D, D, d=r), SE(D)) for r in (3, 2, 1))
        self.fuse = nn.Sequential(block(3 * D, D, k=1), block(D, D))
        self.conv = nn.Sequential(block(D, D, k=1), block(D, D))
        self.pool = nn.AvgPool2d(2)
        self.attn = nn.MultiheadAttention(D, heads, batch_first=True)
        self.norm = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.SiLU(), nn.Linear(4 * D, D))
        self.gate = nn.Sequential(nn.Conv2d(2 * D, D, 1), nn.BatchNorm2d(D), nn.Sigmoid())
        self.gem = nn.Parameter(torch.tensor(p))
        self.tail = nn.Sequential(nn.BatchNorm1d(D), nn.Dropout(.3), nn.Linear(D, K))
        self.mix = mix

    def forward(self, x):
        """x: (B, LAYERS, GRID, GRID, d)."""
        c = [x[:, i].permute(0, 3, 1, 2) for i in range(LAYERS)]
        if self.mix and self.training:
            c = [mixstyle(v) for v in c]
        z = self.fuse(torch.cat([f(v) for f, v in zip(self.paths, c)], 1))
        b, d, h, w = z.shape
        q, kv = z.flatten(2).transpose(1, 2), self.pool(z).flatten(2).transpose(1, 2)
        a = q + self.attn(q, kv, kv, need_weights=False)[0]
        a = self.norm(a + self.mlp(a)).transpose(1, 2).reshape(b, d, h, w)
        c = self.conv(z)
        g = self.gate(torch.cat([c, a], 1))
        f = g * a + (1 - g) * c
        p = self.gem.clamp_min(1.0)
        return self.tail(f.clamp_min(1e-6).pow(p).mean((2, 3)).pow(1 / p))


class MeanLinear(nn.Module):
    """Control on the same inputs: mean of the last layer's tokens, one linear map."""

    def __init__(self, d=DIM):
        super().__init__()
        self.f = nn.Linear(d, K)

    def forward(self, x):
        return self.f(x[:, -1].mean((1, 2)))


def extract(a):
    from .data import ImageTable, load_table
    from .followup import Medsiglip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t, _ = load_table(a.bench)
    sg = Medsiglip().to(dev).eval()
    proc = sg.proc.image_processor
    dl = torch.utils.data.DataLoader(ImageTable(t, a.images, lambda im: proc(images=im, return_tensors="pt")["pixel_values"][0]), 32, num_workers=8)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = np.lib.format.open_memmap(out / "tokens_medsiglip.npy", mode="w+", dtype=np.float16, shape=(len(t), LAYERS, GRID, GRID, DIM))
    R, n = None, 0
    with torch.no_grad():
        for b, (x, _) in enumerate(dl):
            hs = sg.m.vision_model(pixel_values=x.to(dev), output_hidden_states=True).hidden_states
            L = len(hs) - 1
            for j, i in enumerate((L // 3, 2 * L // 3, L)):
                h = hs[i]
                G = int(h.shape[1] ** .5)
                v = nn.functional.adaptive_avg_pool2d(h.reshape(len(x), G, G, -1).permute(0, 3, 1, 2), GRID).permute(0, 2, 3, 1)
                if R is None:
                    R = torch.linalg.qr(torch.randn(v.shape[-1], DIM, generator=torch.Generator().manual_seed(0)))[0].to(v)
                S[n:n + len(x), j] = (v @ R).float().cpu().numpy().astype(np.float16)
            n += len(x)
            if b % 50 == 0:
                print(n, len(t), flush=True)
    S.flush()
    print("[ok]", n, flush=True)


def train(a):
    from .data import load_table
    from sklearn.metrics import balanced_accuracy_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t, _ = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    use = np.flatnonzero(np.isin(sp, ["train", "validation"]))  # test rows stay on disk
    S = np.load(Path(a.tokens) / "tokens_medsiglip.npy", mmap_mode="r")
    X = np.asarray(S[use])
    pos = np.full(len(t), -1)
    pos[use] = np.arange(len(use))
    out = Path(a.out) / f"{a.arm}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def batch(idx, stats):
        return (torch.tensor(X[pos[idx]].astype(np.float32)).to(dev) - stats[0]) / stats[1]

    def run(idx, net, stats):
        net.eval()
        z = []
        with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
            for i in range(0, len(idx), 128):
                z.append(net(batch(idx[i:i + 128], stats)).float().cpu())
        return torch.cat(z).numpy()

    for s in a.sources or sorted(set(src[sp == "train"])):
        f = out / f"{s}.csv"
        if f.exists():
            rows.append(pd.read_csv(f))
            continue
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)
        tr, va, held = np.flatnonzero((sp == "train") & (src != s)), np.flatnonzero((sp == "validation") & (src != s)), np.flatnonzero(np.isin(sp, ["train", "validation"]) & (src == s))
        # one scale per channel from this fold's training rows, so the held-out site plays no part
        sample = X[pos[rng.choice(tr, min(2000, len(tr)), replace=False)]].astype(np.float32)
        stats = torch.tensor(sample.mean((0, 1, 2, 3))).to(dev), torch.tensor(sample.std((0, 1, 2, 3)) + 1e-6).to(dev)
        net = (MeanLinear() if a.arm == "mean" else TokenRPAttn(mix=a.arm == "rp_mix")).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
        steps = len(tr) // a.bs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps, pct_start=.1)
        scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")
        cw = 1 / np.bincount(y[tr], minlength=K)[y[tr]]
        best, keep = -1, None
        for ep in range(a.epochs):
            net.train()
            for _ in range(steps):
                i = tr[rng.choice(len(tr), a.bs, p=cw / cw.sum())]
                x = batch(i, stats)
                if rng.random() < .5:  # the left-right flip of a fundus image is another valid eye
                    x = x.flip(3)
                with torch.autocast(dev, enabled=dev == "cuda"):
                    loss = nn.functional.cross_entropy(net(x), torch.tensor(y[i]).to(dev), label_smoothing=.1)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            v = balanced_accuracy_score(y[va], run(va, net, stats).argmax(1))
            print(s, a.arm, ep + 1, a.epochs, f"val {v:.4f}", flush=True)
            if v > best:
                best, keep = v, {k: w.clone() for k, w in net.state_dict().items()}
        net.load_state_dict(keep)
        zh, zv = run(held, net, stats), run(va, net, stats)
        np.save(out / f"{s}.npy", zh)
        np.save(out / f"{s}_val.npy", zv)
        # text banks and the head share one scale, taken from the rows of the validation split
        m = np.zeros(len(t), bool)
        m[va] = True
        Tm = np.mean([scaled(np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32), m) for b in BANKS], 0)[held]
        zc = zh - zh.mean(1, keepdims=True)
        zs = zc / (zv - zv.mean(1, keepdims=True)).std()
        yy = y[held]
        r = []
        for arm, z in [(a.arm, zh), (a.arm + "+allbanks", .5 * zs + .5 * Tm)]:
            pred = z.argmax(1)
            r.append(dict(seed=a.seed, source=s, arm=arm, val=best, **metrics(z, yy), recall_G=(pred[yy == 2] == 2).mean() if (yy == 2).any() else np.nan))
        r = pd.DataFrame(r)
        r.to_csv(f, index=False)
        rows.append(r)
        print(r.to_string(), flush=True)
    d = pd.concat(rows)
    print(d.groupby("arm")[["bacc", "ece", "nll", "recall_G"]].mean().round(3).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--bench", required=True)
    e.add_argument("--images", required=True)
    e.add_argument("--out", required=True)
    c = sub.add_parser("train")
    c.add_argument("--bench", required=True)
    c.add_argument("--tokens", required=True)
    c.add_argument("--zeroshot", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--arm", required=True, choices=["mean", "rp", "rp_mix"])
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--epochs", type=int, default=12)
    c.add_argument("--bs", type=int, default=64)
    c.add_argument("--lr", type=float, default=3e-4)
    c.add_argument("--sources", nargs="*")
    a = ap.parse_args(argv)
    (extract if a.cmd == "extract" else train)(a)


if __name__ == "__main__":
    main()
