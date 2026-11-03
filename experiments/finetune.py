"""Full fine-tuning: backbone x training loss x seed, then the same post-hoc calibration / metrics as the probes.

Protocol (fixed for all runs; only --model / --loss / --seed change):
  model's native input size and normalisation, AdamW lr 1e-4 (head 1e-3), wd 0.05, cosine schedule with 1 warm-up epoch,
  15 epochs (set in hpc/tasks_B.txt), batch 32, (source, class)-balanced sampling, light fundus augmentation (crop, flip, rotation, colour),
  checkpoint = best validation balanced accuracy (--select nll: lowest validation NLL, sensitivity arm only); mixed precision on CUDA.
python -m experiments.finetune --bench data/bench --images data/processed --out runs/finetune \
    --model resnet50_in1k --loss focal --seed 0
"""
import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from sklearn.metrics import balanced_accuracy_score

from .backbones import VM, build_vm
from .calib import loss_fn
from .data import ImageTable, Preprocess, balanced_weights, load_table
from .evaluator import evaluate

def transforms(cfg, train):
    size = cfg["input_size"][-1]
    base = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(cfg["mean"], cfg["std"])]
    if train:
        # square first, as at evaluation: otherwise non-square sources get a different geometry in training
        return T.Compose([T.Resize((size, size)), T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(),
                          T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.1, 0.02), *base])
    return T.Compose([T.Resize((size, size)), *base])


class LabelCollate:
    """Picklable collate: ImageTable yields (image, row index); labels come from the split's label array."""

    def __init__(self, labels):
        self.labels = labels

    def __call__(self, batch):
        return torch.stack([x for x, _ in batch]), torch.tensor(self.labels[[i for _, i in batch]])


def claim(run):
    """Two SLURM jobs may walk the same runs: only one trains a run. A claim whose job has ended can be taken over."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return True
    lock = run / ".claim"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner = lock.read_text().strip()
            alive = owner and owner != job and subprocess.run(["squeue", "-h", "-j", owner, "-o", "%T"], capture_output=True, text=True).stdout.strip()
            if alive:
                return False
            lock.unlink(missing_ok=True)
            continue
        os.write(fd, job.encode())
        os.close(fd)
        return True
    return False


@torch.no_grad()
def predict(net, dl, dev):
    net.eval()
    out = []
    for x, _ in dl:
        with torch.autocast(dev, enabled=dev == "cuda"):
            out.append(net(x.to(dev)).float().cpu())
    return torch.cat(out).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, choices=[k for k, v in VM.items() if v[2]])
    ap.add_argument("--loss", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: images per split")
    ap.add_argument("--select", choices=["bacc", "nll"], default="bacc")
    ap.add_argument("--preprocess", choices=["none", "clahe", "graham"], default="none")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    run = Path(a.out) / f"{a.model}_{a.loss}_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "results.jsonl").exists() and not a.limit:
        print(f"[skip] {run.name} already finished")
        return
    if not a.limit and not claim(run):
        print(f"[skip] {run.name} running in another job")
        return
    t, classes = load_table(a.bench)
    if a.limit:
        t = pd.concat([g.sample(min(a.limit, len(g)), random_state=0) for _, g in t.groupby("split")]).reset_index(drop=True)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr = np.flatnonzero(sp == "train")
    w = balanced_weights(zip(src[tr], y[tr]))
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w), len(tr), replacement=True)

    net, cfg = build_vm(a.model, len(classes))
    net = net.to(dev)
    wrap = (lambda tf: tf) if a.preprocess == "none" else (lambda tf: Preprocess(a.preprocess, tf))
    tr_ds = ImageTable(t.iloc[tr], a.images, wrap(transforms(cfg, True)))
    tr_dl = torch.utils.data.DataLoader(tr_ds, a.bs, sampler=sampler, num_workers=a.workers, drop_last=True,
                                        collate_fn=LabelCollate(y[tr]))
    eval_dl = torch.utils.data.DataLoader(ImageTable(t, a.images, wrap(transforms(cfg, False))), 64, num_workers=a.workers)
    va = sp == "validation"
    va_dl = torch.utils.data.DataLoader(ImageTable(t[va], a.images, wrap(transforms(cfg, False))), 64, num_workers=a.workers)

    head = list(net.get_classifier().parameters())
    hid = {id(p) for p in head}
    opt = torch.optim.AdamW([{"params": [p for p in net.parameters() if id(p) not in hid], "lr": 1e-4},
                             {"params": head, "lr": 1e-3}], weight_decay=0.05)
    steps = a.epochs * len(tr_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / len(tr_dl)) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    f = loss_fn(a.loss)
    best, best_bacc = -np.inf, -1.0
    for ep in range(a.epochs):
        net.train()
        for x, yb in tr_dl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = f(net(x.to(dev)).float(), yb.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        Zv = predict(net, va_dl, dev)
        b = balanced_accuracy_score(y[va], Zv.argmax(1))
        print(f"epoch {ep + 1}/{a.epochs} val bAcc {b:.4f}", flush=True)
        if a.select == "nll":  # keep the bAcc choice from this same training, so the two rules compare without a hardware difference
            if b > best_bacc:
                best_bacc = b
                torch.save(net.state_dict(), run / "best_bacc.pt")
            b = -torch.nn.functional.cross_entropy(torch.tensor(Zv), torch.tensor(y[va])).item()
        if b > best:
            best = b
            torch.save(net.state_dict(), run / "best.pt")
    if a.select == "nll":
        net.load_state_dict(torch.load(run / "best_bacc.pt", map_location=dev))
        np.save(run / "logits_bacc_rule.npy", predict(net, eval_dl, dev))
    net.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    Z = predict(net, eval_dl, dev)
    np.save(run / "logits.npy", Z)
    t[["image_id", "split", "source"]].to_csv(run / "index.csv", index=False)

    with open(run / "results.jsonl", "w") as fh:
        for r in evaluate(Z, t, {"setting": "finetune", "model": a.model, "loss": a.loss, "seed": a.seed, "best_val_bacc": best}):
            fh.write(json.dumps(r) + "\n")
            print(r["posthoc"], f"test bAcc {r['test']['bacc']:.3f} ECE {r['test']['ece']:.3f}", flush=True)


if __name__ == "__main__":
    main()
