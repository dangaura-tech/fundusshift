"""Zero-shot VLMs x prompt banks, raw and calibrated, from cached image features.
python -m experiments.zeroshot_clip --bench data/bench --features runs/features --out runs/zeroshot
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .backbones import VLM, build_clip
from .data import load_table
from .evaluator import evaluate
from .prompts import BANKS, bank


@torch.no_grad()
def text_weights(model, tok, classes, name):
    W = []
    for c in classes:
        e = model.encode_text(tok(bank(name, c))).float()
        e = (e / e.norm(dim=1, keepdim=True)).mean(0)
        W.append(e / e.norm())
    return torch.stack(W).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=list(VLM))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, classes = load_table(a.bench)
    with open(out / "results.jsonl", "w") as fh:
        for name in a.models:
            X = np.load(Path(a.features) / f"{name}.npy").astype(np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            model, _, tok = build_clip(name)
            scale = model.logit_scale.exp().item()
            for b in BANKS:
                Z = scale * X @ text_weights(model, tok, classes, b).T
                np.save(out / f"logits_{name}_{b}.npy", Z)
                for r in evaluate(Z, t, {"setting": "zero-shot", "model": name, "loss": b, "seed": 0}):
                    fh.write(json.dumps(r) + "\n")
                    print(name, b, r["posthoc"], f"test bAcc {r['test']['bacc']:.3f} ECE {r['test']['ece']:.3f}", flush=True)


if __name__ == "__main__":
    main()
