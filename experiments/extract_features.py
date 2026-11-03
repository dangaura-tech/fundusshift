"""Frozen image embeddings for every released image, per model (cached).
python -m experiments.extract_features --bench data/bench --images data/processed --out runs/features [--models ...]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .backbones import ALL, feature_extractor
from .data import ImageTable, Preprocess, load_table


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def extract(name, table, images, bs=64, workers=6, prep="none"):
    model, pre, emb = feature_extractor(name)
    pre = pre if prep == "none" else Preprocess(prep, pre)
    model = model.to(device()).eval()
    dl = torch.utils.data.DataLoader(ImageTable(table, images, pre), batch_size=bs, num_workers=workers)
    return np.concatenate([emb(model, x.to(device())).float().cpu().numpy() for x, _ in dl]).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=ALL)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--preprocess", choices=["none", "clahe", "graham"], default="none")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, _ = load_table(a.bench)
    for name in a.models:
        f = out / f"{name}.npy"
        if f.exists() and np.load(f, mmap_mode="r").shape[0] == len(t):
            print(f"[cached] {name}", flush=True)
            continue
        X = extract(name, t, a.images, workers=a.workers, prep=a.preprocess)
        np.save(f, X)
        print(f"[ok] {name}: {X.shape}", flush=True)


if __name__ == "__main__":
    main()
