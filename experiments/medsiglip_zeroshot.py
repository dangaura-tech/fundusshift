"""MedSigLIP-448 (gated weights, local copy): frozen features and zero-shot, raw and calibrated.
python -m experiments.medsiglip_zeroshot --bench data/bench --images data/processed --features runs/features --out runs/medsiglip
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .backbones import GATED
from .data import ImageTable, load_table
from .evaluator import evaluate
from .extract_features import device
from .prompts import BANKS, bank


def tensor(out):
    return out if torch.is_tensor(out) else out.pooler_output


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args(argv)
    from transformers import AutoModel, AutoProcessor
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, classes = load_table(a.bench)
    dev = device()
    path = GATED / "medsiglip"
    model = AutoModel.from_pretrained(path).to(dev).eval()
    proc = AutoProcessor.from_pretrained(path)
    f = Path(a.features) / "medsiglip.npy"
    if f.exists() and np.load(f, mmap_mode="r").shape[0] == len(t):
        X = np.load(f)
    else:
        pre = lambda im: proc.image_processor(images=im, return_tensors="pt")["pixel_values"][0]
        dl = torch.utils.data.DataLoader(ImageTable(t, a.images, pre), batch_size=16, num_workers=a.workers)
        with torch.no_grad():
            X = np.concatenate([tensor(model.get_image_features(pixel_values=x.to(dev))).float().cpu().numpy() for x, _ in dl]).astype(np.float32)
        np.save(f, X)
    print(f"[ok] medsiglip: {X.shape}", flush=True)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    scale = model.logit_scale.exp().item()
    with open(out / "results.jsonl", "w") as fh:
        for b in BANKS:
            W = []
            for c in classes:
                tok = proc.tokenizer(bank(b, c), padding="max_length", truncation=True, return_tensors="pt").to(dev)
                with torch.no_grad():
                    e = tensor(model.get_text_features(**tok)).float()
                e = (e / e.norm(dim=1, keepdim=True)).mean(0)
                W.append((e / e.norm()).cpu().numpy())
            Z = scale * Xn @ np.stack(W).T
            np.save(out / f"logits_medsiglip_{b}.npy", Z)
            for r in evaluate(Z, t, {"setting": "zero-shot", "model": "medsiglip", "loss": b, "seed": 0}):
                fh.write(json.dumps(r) + "\n")
                print("medsiglip", b, r["posthoc"], f"test bAcc {r['test']['bacc']:.3f} ECE {r['test']['ece']:.3f}", flush=True)


if __name__ == "__main__":
    main()
