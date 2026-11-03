"""FLAIR (Silva-Rodriguez et al., MedIA 2025; open weights jusiro2/FLAIR): frozen features and zero-shot, raw and calibrated.

FLAIR is the only retina-specific VLM in the sweep. Its pretraining pool includes FundusShift test sources
(PAPILA, Paraguay, HRF, FIVES), so its test numbers are contaminated: report them alongside the per-source results.
FLAIR's own preprocessing is used: scale to [0, 1], longer side to 512, pad bottom/right. Needs the FLAIR source on PYTHONPATH.
python -m experiments.flair_zeroshot --bench data/bench --images data/processed --features runs/features --out runs/flair
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor, resize

from .data import ImageTable, load_table
from .evaluator import evaluate
from .extract_features import device
from .prompts import BANKS, bank

SIZE = 512


def preprocess(img):
    x = pil_to_tensor(img.convert("RGB")).float() / 255
    s = SIZE / max(x.shape[-2:])
    x = resize(x, [int(x.shape[-2] * s), int(x.shape[-1] * s)], antialias=True)
    return torch.nn.functional.pad(x, (0, SIZE - x.shape[-1], 0, SIZE - x.shape[-2]))


@torch.no_grad()
def text_weights(model, classes, name, dev):
    W = []
    for c in classes:
        tok = model.text_model.tokenizer(bank(name, c), truncation=True, padding=True, return_tensors="pt").to(dev)
        e = model.text_model(tok["input_ids"], tok["attention_mask"]).mean(0)
        W.append(e / e.norm())
    return torch.stack(W).cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args(argv)
    from flair import FLAIRModel
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, classes = load_table(a.bench)
    dev = device()
    model = FLAIRModel.from_pretrained("jusiro2/FLAIR").to(dev).eval()
    f = Path(a.features) / "flair.npy"
    if f.exists() and np.load(f, mmap_mode="r").shape[0] == len(t):
        X = np.load(f)
    else:
        dl = torch.utils.data.DataLoader(ImageTable(t, a.images, preprocess), batch_size=32, num_workers=a.workers)
        with torch.no_grad():
            X = np.concatenate([model.vision_model(x.to(dev)).float().cpu().numpy() for x, _ in dl]).astype(np.float32)
        np.save(f, X)
    print(f"[ok] flair: {X.shape}", flush=True)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    scale = model.logit_scale.exp().item()
    with open(out / "results.jsonl", "w") as fh:
        for b in BANKS:
            Z = scale * Xn @ text_weights(model, classes, b, dev).T
            np.save(out / f"logits_flair_{b}.npy", Z)
            for r in evaluate(Z, t, {"setting": "zero-shot", "model": "flair", "loss": b, "seed": 0}):
                fh.write(json.dumps(r) + "\n")
                print("flair", b, r["posthoc"], f"test bAcc {r['test']['bacc']:.3f} ECE {r['test']['ece']:.3f}", flush=True)


if __name__ == "__main__":
    main()
