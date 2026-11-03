"""STRATA end to end: fine-tuning with the conditional-logit loss, leave-one-development-source-out. The table is cut to train and validation, so no test row is loaded.
Same recipe as experiments.finetune. --head rpattn swaps in RPAttn head C. Arms: ce (usual batch of 32), strata (16 usual + 8 same-site pairs), cross (16 usual + 8 pairs across sites).
python -m experiments.strata_ft --bench data/bench --images data/processed --out results/strata/ft --model convnext_t --arm strata --lam 0.3 [--head rpattn]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .backbones import VM, build_vm
from .calib import loss_fn
from .data import ImageTable, balanced_weights, load_table
from .finetune import LabelCollate, predict, transforms
from .rpattn import RPAttnNet
from .shoal import metrics
from .strata import pair_loss, pair_table


def epoch_batches(n, w, table, bs, rng):
    """Row positions per step. Pair arms: the last bs / 2 positions are a0, b0, a1, b1, ..."""
    types = sorted(table) if table else []
    out = []
    for _ in range(n // bs):
        k = bs // 2 if table else bs
        b = list(rng.choice(n, k, p=w / w.sum()))
        for _ in range((bs - k) // 2):
            c = types[rng.integers(len(types))]
            ia, ib = table[c][rng.integers(len(table[c]))]
            b += [ia[rng.integers(len(ia))], ib[rng.integers(len(ib))]]
        out.append(b)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="convnext_t")
    ap.add_argument("--arm", required=True, choices=["ce", "strata", "cross"])
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--head", default="linear", choices=["linear", "rpattn"])
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t, classes = load_table(a.bench)
    t = t[t.split.isin(["train", "validation"])].reset_index(drop=True)
    out = Path(a.out) / f"{a.model}{'_rpattn' if a.head == 'rpattn' else ''}_{a.arm}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    for s in a.sources or sorted(set(t.source)):
        f = out / f"{s}.csv"
        if f.exists():
            continue
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)
        fold = t[t.source != s].reset_index(drop=True)
        held = t[t.source == s].reset_index(drop=True)
        tr_t = fold[fold.split == "train"].reset_index(drop=True)
        va_t = fold[fold.split == "validation"].reset_index(drop=True)
        y, src = tr_t.y.to_numpy(), tr_t.source.to_numpy()
        w = balanced_weights(zip(src, y))
        table = None if a.arm == "ce" else pair_table(y, src if a.arm == "strata" else np.full(len(y), "all"))
        net, cfg = build_vm(a.model, len(classes))
        if a.head == "rpattn":
            net = RPAttnNet(VM[a.model][0], len(classes))
        net = net.to(dev)
        ds = ImageTable(tr_t, a.images, transforms(cfg, True))
        col = LabelCollate(y)
        ev = lambda d: torch.utils.data.DataLoader(ImageTable(d, a.images, transforms(cfg, False)), 64, num_workers=a.workers)
        va_dl, held_dl = ev(va_t), ev(held)
        head = list(net.get_classifier().parameters())
        hid = {id(p) for p in head}
        opt = torch.optim.AdamW([{"params": [p for p in net.parameters() if id(p) not in hid], "lr": 1e-4}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
        per = len(y) // a.bs
        steps = a.epochs * per
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / per) * 0.5 * (1 + math.cos(math.pi * i / steps)))
        scaler = torch.amp.GradScaler(enabled=dev == "cuda")
        ce = loss_fn("ce")
        best, state = -1.0, None
        for ep in range(a.epochs):
            net.train()
            dl = torch.utils.data.DataLoader(ds, batch_sampler=epoch_batches(len(y), w, table, a.bs, rng), num_workers=a.workers, collate_fn=col)
            for x, yb in dl:
                opt.zero_grad(set_to_none=True)
                x, yb = x.to(dev), yb.to(dev)
                with torch.autocast(dev, enabled=dev == "cuda"):
                    z = net(x).float()
                    if table is None:
                        loss = ce(z, yb)
                    else:
                        k = a.bs // 2
                        zp, yp = z[k:], yb[k:]
                        loss = pair_loss(zp[::2], zp[1::2], yp[::2], yp[1::2]) + a.lam * ce(z[:k], yb[:k])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            b = balanced_accuracy_score(va_t.y, predict(net, va_dl, dev).argmax(1))
            print(s, f"epoch {ep + 1}/{a.epochs} val bAcc {b:.4f}", flush=True)
            if b > best:
                best, state = b, {k: v.cpu().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(state)
        Z, yy = predict(net, held_dl, dev), held.y.to_numpy()
        np.save(out / f"{s}.npy", Z)  # the fold's held-out logits, row order of held, for late fusion
        pred = Z.argmax(1)
        rec = {f"recall_{c}": (pred[yy == k] == k).mean() if (yy == k).any() else np.nan for k, c in enumerate(classes)}
        pd.DataFrame([dict(model=a.model, arm=a.arm, seed=a.seed, source=s, n=len(yy), val=best, **metrics(Z, yy), **rec)]).to_csv(f, index=False)
        print("[ok]", s, f.read_text().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
