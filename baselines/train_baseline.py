"""Minimal ERM baseline for FundusShift: ImageNet ResNet-50 fine-tuned on train, checkpoint chosen on validation
(balanced accuracy), logits written for every split -> baselines/evaluate.py.

python baselines/train_baseline.py --out runs/resnet50 [--epochs 10 --size 384 --bs 32 --limit 0]
python baselines/train_baseline.py --out runs/resnet50 --splits-csv metadata/splits_final.csv --image-root images/
Class imbalance and source imbalance are handled with a sampler weighting each (source, class) cell equally,
as in the benchmark protocol (pooled training otherwise lets the largest source dominate).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights, convnext_tiny, resnet50

CANON = ["N", "D", "G", "A", "M"]   # canonical class order; the release uses those present in train
CLASSES = list(CANON)


def transforms(size, train):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(),
                          T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.1, 0.02), T.ToImage(),
                          T.ToDtype(torch.float32, scale=True), norm])
    return T.Compose([T.Resize((size, size)), T.ToImage(), T.ToDtype(torch.float32, scale=True), norm])


class LocalSplit(torch.utils.data.Dataset):
    """Rows of splits_final.csv + processed images: <image_root>/<source>/<image_id>.jpg"""

    def __init__(self, df, root, tf):
        self.df, self.root, self.tf = df.reset_index(drop=True), Path(root), tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(self.root / r.source / f"{r.image_id}.jpg").convert("RGB")
        return {"x": self.tf(im), "y": int(r.y), "image_id": r.image_id}


class HubSplit(torch.utils.data.Dataset):
    def __init__(self, ds, tf):
        self.ds, self.tf = ds, tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        return {"x": self.tf(r["image"].convert("RGB")), "y": r["label"], "image_id": r["image_id"]}


def load_splits(a):
    """{split: (meta DataFrame with image_id/source/y, torch Dataset factory)} from the Hub or local files."""
    if a.splits_csv:
        d = pd.read_csv(a.splits_csv, dtype=str, keep_default_na=False)
        d["split"] = d.split.replace({"val": "validation", "cal": "calibration"})
        CLASSES[:] = [c for c in CANON if c in set(d.label[d.split == "train"])]
        d["y"] = [CLASSES.index(l) if l in CLASSES else -1 for l in d.label]
        out = {}
        for s in ["train", "validation", "calibration", "test", "test_ood", "test_openset"]:
            sub = d[d.split == s]
            if a.limit:
                sub = sub.sample(min(a.limit, len(sub)), random_state=0)
            out[s] = (sub[["image_id", "source", "y"]].reset_index(drop=True),
                      lambda tf, sub=sub: LocalSplit(sub, a.image_root, tf))
        return out
    from datasets import load_dataset
    ds = load_dataset(a.repo)
    CLASSES[:] = ds["train"].features["label"].names
    out = {}
    for s, v in ds.items():
        if a.limit:
            v = v.shuffle(seed=0).select(range(min(a.limit, len(v))))
        meta = pd.DataFrame({"image_id": v["image_id"], "source": v["source"], "y": v["label"]})
        out[s] = (meta, lambda tf, v=v: HubSplit(v, tf))
    return out


def collate(b):
    return torch.stack([r["x"] for r in b]), torch.tensor([r["y"] for r in b]), [r["image_id"] for r in b]


def loader(split, size, train, bs, workers):
    meta, make = split
    ds = make(transforms(size, train))
    if train:
        cell = pd.Series(list(zip(meta.source, meta.y)))
        w = 1.0 / cell.map(cell.value_counts()).to_numpy()
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        return torch.utils.data.DataLoader(ds, bs, sampler=sampler, num_workers=workers, collate_fn=collate)
    return torch.utils.data.DataLoader(ds, bs, shuffle=False, num_workers=workers, collate_fn=collate)


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    ids, logits, ys = [], [], []
    for x, y, i in dl:
        logits.append(model(x.to(device)).float().cpu())
        ys.append(y)
        ids += i
    return ids, torch.cat(logits).numpy(), torch.cat(ys).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="le-dang1haha/FundusShift")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images per split")
    ap.add_argument("--splits-csv", help="local mode: metadata/splits_final.csv instead of the Hub")
    ap.add_argument("--image-root", help="local mode: folder with <source>/<image_id>.jpg")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "convnext_tiny"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    ds = load_splits(a)
    if a.arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    else:
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, len(CLASSES))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    tr = loader(ds["train"], a.size, True, a.bs, a.workers)
    va = loader(ds["validation"], a.size, False, a.bs, a.workers)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tr))
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        for x, y, _ in tr:
            loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device), label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        _, lg, y = predict(model, va, device)
        bacc = balanced_accuracy_score(y, lg.argmax(1))
        print(f"epoch {ep + 1}: val balanced acc {bacc:.4f}")
        if bacc > best:
            best = bacc
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    for split in ds:
        ids, lg, _ = predict(model, loader(ds[split], a.size, False, a.bs, a.workers), device)
        df = pd.DataFrame(lg, columns=[f"logit_{c}" for c in CLASSES])
        df.insert(0, "image_id", ids)
        df.to_csv(out / f"{split}.csv", index=False)
    print(f"best val balanced acc {best:.4f}; logits in {out}")


if __name__ == "__main__":
    main()
