"""MedSigLIP features under CLAHE and Graham preprocessing. It is gated and loaded apart from extract_features, as in medsiglip_zeroshot.
First re-extracts 256 images without preprocessing and checks them against the clean features.
python -m experiments.preprocess_medsiglip --bench data/bench --images data/processed --features runs/features --out runs/preprocess
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .data import ImageTable, Preprocess, load_table
from .followup import Medsiglip, SiglipTransform


def embed(sg, t, images, tf, dev, workers=4):
    dl = torch.utils.data.DataLoader(ImageTable(t, images, tf), 32, num_workers=workers)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            o = sg.m.get_image_features(pixel_values=x.to(dev))
            out.append((o if torch.is_tensor(o) else o.pooler_output).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    t, _ = load_table(a.bench)
    sg = Medsiglip().to(dev).eval()
    tf = SiglipTransform(sg.proc.image_processor)
    ref = np.load(Path(a.features) / "medsiglip.npy", mmap_mode="r")
    idx = np.random.default_rng(0).choice(len(t), 256, replace=False)
    X, R = embed(sg, t.iloc[idx], a.images, tf, dev), np.asarray(ref[idx])
    cos = (X * R).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(R, axis=1))
    print("clean check cosine min/mean", round(float(cos.min()), 5), round(float(cos.mean()), 5), flush=True)
    if cos.min() < .99:
        raise SystemExit("features do not match the clean extraction")
    for kind in ["clahe", "graham"]:
        f = Path(a.out) / kind / "features" / "medsiglip.npy"
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists() and np.load(f, mmap_mode="r").shape[0] == len(t):
            print("[cached]", kind, flush=True)
            continue
        np.save(f, embed(sg, t, a.images, Preprocess(kind, tf), dev))
        print("[ok]", kind, flush=True)


if __name__ == "__main__":
    main()
