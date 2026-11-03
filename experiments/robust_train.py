"""Robust and regularised end-to-end baselines, on the fine-tuning recipe of experiments.finetune with only the named change.

--backbone convnext_t (full fine-tuning) or dinov2_lora (DINOv2-B/14 at 224 px, LoRA rank 8 on query and value of every block, linear head).
--objective erm | groupdro (Sagawa et al. 2020; eta 0.01; group = development source) | coral (Sun & Saenko 2016, DomainBed form; lambda 1).
--swad: SWAD (Cha et al., NeurIPS 2021) dense weight average over segments of 150 updates, official loss-valley rule
        (n_converge 3, n_tolerance 6, ratio 0.3); replaces the validation checkpoint.
Checkpoint otherwise = best validation balanced accuracy, as in fine-tuning.
python -m experiments.robust_train --bench data/bench --images data/processed --out runs/robust --backbone dinov2_lora --objective groupdro --seed 0
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from .backbones import build_vm
from .data import ImageTable, balanced_weights, load_table
from .evaluator import evaluate
from .finetune import LabelCollate, predict, transforms


class LoraQKV(torch.nn.Module):
    """LoRA on timm's joint qkv projection: the update reaches the query and value thirds only, never the key."""

    def __init__(self, base, r=8):
        super().__init__()
        self.base = base
        d, o = base.in_features, base.out_features
        self.A = torch.nn.Linear(d, r, bias=False)
        self.B = torch.nn.Linear(r, o, bias=False)
        torch.nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.B.weight)
        m = torch.ones(o)
        m[o // 3:2 * o // 3] = 0
        self.register_buffer("mask", m)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.mask


def build(name, k):
    if name == "convnext_t":
        return build_vm("convnext_t", k)
    import timm
    net = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=k, img_size=224)
    for p in net.parameters():
        p.requires_grad_(False)
    for blk in net.blocks:
        blk.attn.qkv = LoraQKV(blk.attn.qkv)
    for p in net.get_classifier().parameters():
        p.requires_grad_(True)
    return net, dict(timm.data.resolve_model_data_config(net), input_size=(3, 224, 224))


def coral_penalty(f, g):
    idx = [torch.nonzero(g == s).squeeze(1) for s in g.unique()]
    idx = [i for i in idx if len(i) >= 4]
    pen, n = f.sum() * 0.0, 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = f[idx[i]], f[idx[j]]
            pen = pen + (a.mean(0) - b.mean(0)).pow(2).mean() + (torch.cov(a.T) - torch.cov(b.T)).pow(2).mean()
            n += 1
    return pen / max(n, 1)


def valley(losses, n_converge=3, n_tol=6, ratio=0.3):
    """Official SWAD window over segment validation losses: (first, last) segment index."""
    s = next((i for i in range(len(losses) - n_converge + 1) if losses[i] == min(losses[i:i + n_converge])), 0)
    thr = float(np.mean(losses[s:s + n_converge])) * (1 + ratio)
    for e in range(s + n_converge, len(losses) - n_tol + 1):
        if min(losses[e:e + n_tol]) > thr:
            return s, e - 1
    return s, len(losses) - 1


@torch.no_grad()
def val_nll(net, dl, y, dev):
    return F.cross_entropy(torch.tensor(predict(net, dl, dev)), torch.tensor(y)).item()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", required=True, choices=["convnext_t", "dinov2_lora"])
    ap.add_argument("--objective", default="erm", choices=["erm", "groupdro", "coral"])
    ap.add_argument("--swad", action="store_true")
    ap.add_argument("--swad-segment", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tag = f"{a.backbone}_{a.objective}{'_swad' if a.swad else ''}_s{a.seed}"
    run = Path(a.out) / tag
    run.mkdir(parents=True, exist_ok=True)
    if (run / "results.jsonl").exists() and not a.limit:
        print(f"[skip] {tag} already finished")
        return
    t, classes = load_table(a.bench)
    if a.limit:
        t = pd.concat([g.sample(min(a.limit, len(g)), random_state=0) for _, g in t.groupby("split")]).reset_index(drop=True)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr = np.flatnonzero(sp == "train")
    groups = sorted(set(src[tr]))
    gid = np.array([groups.index(s) for s in src[tr]])
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(balanced_weights(zip(src[tr], y[tr]))), len(tr), replacement=True)
    net, cfg = build(a.backbone, len(classes))
    net = net.to(dev)
    tr_dl = torch.utils.data.DataLoader(ImageTable(t.iloc[tr], a.images, transforms(cfg, True)), a.bs, sampler=sampler, num_workers=a.workers,
                                        drop_last=True, collate_fn=LabelCollate(np.stack([y[tr], gid], 1)))
    eval_dl = torch.utils.data.DataLoader(ImageTable(t, a.images, transforms(cfg, False)), 64, num_workers=a.workers)
    va = sp == "validation"
    va_dl = torch.utils.data.DataLoader(ImageTable(t[va], a.images, transforms(cfg, False)), 64, num_workers=a.workers)
    head = list(net.get_classifier().parameters())
    hid = {id(p) for p in head}
    body = [p for p in net.parameters() if p.requires_grad and id(p) not in hid]
    opt = torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
    steps = a.epochs * len(tr_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / len(tr_dl)) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    q = torch.ones(len(groups), device=dev) / len(groups)
    names = [n for n, p in net.named_parameters() if p.requires_grad]
    params = dict(net.named_parameters())
    seg_sum, seg_n, segments, seg_loss, u = None, 0, [], [], 0
    best = -1.0
    for ep in range(a.epochs):
        net.train()
        for x, lab in tr_dl:
            yb, g = lab[:, 0].to(dev), lab[:, 1].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                feats = net.forward_features(x.to(dev))
                pre = net.forward_head(feats, pre_logits=True)
                z = net.get_classifier()(pre) if a.backbone == "dinov2_lora" else net.forward_head(feats)
            z = z.float()
            if a.objective == "groupdro":
                per = F.cross_entropy(z, yb, reduction="none")
                ls = torch.zeros(len(groups), device=dev)
                for i in g.unique():
                    ls[i] = per[g == i].mean()
                    q[i] = q[i] * torch.exp(0.01 * ls[i].detach())
                q = q / q.sum()
                loss = (q * ls).sum()
            else:
                loss = F.cross_entropy(z, yb)
                if a.objective == "coral":
                    loss = loss + coral_penalty(pre.float().flatten(1), g)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            u += 1
            if a.swad:
                with torch.no_grad():
                    cur = {n: params[n].detach().to("cpu", dtype=torch.float32) for n in names}
                    seg_sum = cur if seg_sum is None else {n: seg_sum[n] + cur[n] for n in names}
                    seg_n += 1
                if seg_n == a.swad_segment:
                    segments.append({n: v / seg_n for n, v in seg_sum.items()})
                    seg_sum, seg_n = None, 0
                    keep = {n: params[n].detach().clone() for n in names}
                    with torch.no_grad():
                        for n in names:
                            params[n].copy_(segments[-1][n].to(dev))
                    seg_loss.append(val_nll(net, va_dl, y[va], dev))
                    with torch.no_grad():
                        for n in names:
                            params[n].copy_(keep[n])
                    net.train()
        b = balanced_accuracy_score(y[va], predict(net, va_dl, dev).argmax(1))
        print(f"epoch {ep + 1}/{a.epochs} val bAcc {b:.4f}", flush=True)
        if b > best:
            best = b
            torch.save({n: params[n].detach().cpu() for n in names}, run / "best.pt")
    if a.swad and segments:
        s, e = valley(seg_loss)
        print(f"SWAD valley: segments {s}-{e} of {len(segments)}", flush=True)
        with torch.no_grad():
            for n in names:
                params[n].copy_(torch.stack([segments[i][n] for i in range(s, e + 1)]).mean(0).to(dev))
        best = float(balanced_accuracy_score(y[va], predict(net, va_dl, dev).argmax(1)))
    else:
        for n, v in torch.load(run / "best.pt", map_location=dev).items():
            params[n].data.copy_(v)
    Z = predict(net, eval_dl, dev)
    np.save(run / "logits.npy", Z)
    with open(run / "results.jsonl", "w") as fh:
        setting = "swad" if a.swad else {"erm": "finetune-reference", "groupdro": "group-dro", "coral": "coral"}[a.objective]
        for r in evaluate(Z, t, {"setting": setting, "model": a.backbone, "loss": "ce", "seed": a.seed, "best_val_bacc": best}):
            fh.write(json.dumps(r) + "\n")
            print(r["posthoc"], f"test bAcc {r['test']['bacc']:.3f} ECE {r['test']['ece']:.3f}", flush=True)


if __name__ == "__main__":
    main()
