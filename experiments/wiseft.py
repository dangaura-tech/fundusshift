"""WiSE-FT (Wortsman et al., CVPR 2022) on a VLM image tower: head initialised from the zero-shot text classifier, end-to-end fine-tuning,
then weight-space interpolation theta(alpha) = (1 - alpha) theta_zero-shot + alpha theta_fine-tuned, alpha = 0, 0.1, ..., 1.

Bank = highest zero-shot validation balanced accuracy. AdamW lr 1e-5, wd 0.1, cosine, 15 epochs, batch 32, (source, class)-balanced sampling;
checkpoint and alpha by validation balanced accuracy (benchmark rule); the same choice by validation NLL and alpha = 0.5 (the paper's default) are also saved.
python -m experiments.wiseft --bench data/bench --images data/processed --out runs/wiseft --model siglip2_l16 --seed 0
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torchvision.transforms import v2 as T

from .backbones import build_clip
from .data import ImageTable, balanced_weights, load_table
from .evaluator import evaluate
from .finetune import LabelCollate
from .prompts import BANKS
from .zeroshot_clip import text_weights


class Net(torch.nn.Module):
    def __init__(self, model, W, scale):
        super().__init__()
        self.visual = model.visual
        self.head = torch.nn.Parameter(scale * W.clone())

    def forward(self, x):
        return F.normalize(self.visual(x), dim=-1) @ self.head.T


@torch.no_grad()
def predict(net, dl, dev):
    net.eval()
    out = []
    for x, _ in dl:
        with torch.autocast(dev, enabled=dev == "cuda"):
            out.append(net(x.to(dev)).float().cpu())
    return torch.cat(out).numpy()


def interp(net, a0, a1, alpha):
    with torch.no_grad():
        for k, v in net.state_dict().items():
            if v.is_floating_point():
                v.copy_((1 - alpha) * a0[k].to(v.device) + alpha * a1[k].to(v.device))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="siglip2_l16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    run = Path(a.out) / f"{a.model}_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "results.jsonl").exists() and not a.limit:
        print(f"[skip] {run.name} already finished")
        return
    t, classes = load_table(a.bench)
    if a.limit:
        t = pd.concat([g.sample(min(a.limit, len(g)), random_state=0) for _, g in t.groupby("split")]).reset_index(drop=True)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    model, pre, tok = build_clip(a.model)
    scale = model.logit_scale.exp().item()
    va = sp == "validation"
    tr = np.flatnonzero(sp == "train")
    va_dl = torch.utils.data.DataLoader(ImageTable(t[va], a.images, pre), 64, num_workers=a.workers)
    all_dl = torch.utils.data.DataLoader(ImageTable(t, a.images, pre), 64, num_workers=a.workers)
    Ws = {b: torch.tensor(text_weights(model, tok, classes, b)) for b in BANKS}
    model = model.to(dev)
    with torch.no_grad():
        vf = []
        model.eval()
        for x, _ in va_dl:
            vf.append(F.normalize(model.visual(x.to(dev)), dim=-1).float().cpu())
    vf = torch.cat(vf)
    bank = max(BANKS, key=lambda b: balanced_accuracy_score(y[va], (vf @ Ws[b].T).argmax(1).numpy()))
    print("bank", bank, flush=True)
    net = Net(model, Ws[bank].to(dev), scale).to(dev).float()
    for p in net.parameters():
        p.requires_grad_(True)
    theta0 = copy.deepcopy({k: v.cpu() for k, v in net.state_dict().items()})
    size = pre.transforms[0].size if hasattr(pre.transforms[0], "size") else 224
    size = size if isinstance(size, int) else size[0]
    norm = [m for m in pre.transforms if isinstance(m, type(pre.transforms[-1]))][-1]
    cfg = {"input_size": (3, size, size), "mean": norm.mean, "std": norm.std}
    base = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(cfg["mean"], cfg["std"])]
    aug = T.Compose([T.Resize((size, size)), T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(),
                     T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.1, 0.02), *base])
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(balanced_weights(zip(src[tr], y[tr]))), len(tr), replacement=True)
    tr_dl = torch.utils.data.DataLoader(ImageTable(t.iloc[tr], a.images, aug), a.bs, sampler=sampler, num_workers=a.workers, drop_last=True,
                                        collate_fn=LabelCollate(y[tr]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=0.1)
    steps = a.epochs * len(tr_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    yv = torch.tensor(y[va])
    best_nll, theta_nll, best_b, theta_b = np.inf, None, -1.0, None
    for ep in range(a.epochs):
        net.train()
        for x, yb in tr_dl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = F.cross_entropy(net(x.to(dev)).float(), yb.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        Zv = predict(net, va_dl, dev)
        v, b = F.cross_entropy(torch.tensor(Zv), yv).item(), balanced_accuracy_score(y[va], Zv.argmax(1))
        print(f"epoch {ep + 1}/{a.epochs} val bAcc {b:.4f} NLL {v:.4f}", flush=True)
        if b > best_b:
            best_b, theta_b = b, copy.deepcopy({k: x.cpu() for k, x in net.state_dict().items()})
        if v < best_nll:
            best_nll, theta_nll = v, copy.deepcopy({k: x.cpu() for k, x in net.state_dict().items()})
    baccs, nlls = {}, {}
    for al in np.round(np.arange(0, 1.01, 0.1), 1):
        interp(net, theta0, theta_b, float(al))
        baccs[float(al)] = balanced_accuracy_score(y[va], predict(net, va_dl, dev).argmax(1))
        interp(net, theta0, theta_nll, float(al))
        nlls[float(al)] = F.cross_entropy(torch.tensor(predict(net, va_dl, dev)), yv).item()
    al_b, al_n = max(baccs, key=baccs.get), min(nlls, key=nlls.get)
    print("alpha val bAcc", baccs, "chosen", al_b, "alpha val NLL", nlls, "chosen", al_n, flush=True)
    with open(run / "results.jsonl", "w") as fh:
        for label, theta, al in [("val-bacc", theta_b, al_b), ("val-nll", theta_nll, al_n), ("alpha0.5", theta_b, 0.5), ("fully-tuned", theta_b, 1.0)]:
            interp(net, theta0, theta, al)
            Z = predict(net, all_dl, dev)
            np.save(run / f"logits_{label}.npy", Z)
            for r in evaluate(Z, t, {"setting": "wise-ft", "model": a.model, "loss": label, "alpha": al, "seed": a.seed}):
                fh.write(json.dumps(r) + "\n")
                print(label, al, r["posthoc"], f"test bAcc {r['test']['bacc']:.3f} ECE {r['test']['ece']:.3f}", flush=True)


if __name__ == "__main__":
    main()
