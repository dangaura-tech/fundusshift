"""Zero-shot generative VLMs (LLaVA-OneVision, Qwen2.5-VL) as a multiple-choice screener, uncalibrated and calibrated.

The model sees the fundus image and a 3-option question; class scores are the next-token logits of the option
letters (A/B/C). Each image is scored under all 6 option orders; we report each order (mean over orders) and the
average of the 6 per-class log-likelihoods, which removes letter-position bias.
python -m experiments.vlm_mcq --bench data/bench --images data/processed --out runs/vlm \
    --model Qwen/Qwen2.5-VL-7B-Instruct [--splits calibration test test_ood test_openset] [--limit N]
"""
import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .data import load_table
from .evaluator import evaluate

OPTION_TEXT = {"N": "normal fundus (no diabetic retinopathy, no glaucoma)",
               "D": "diabetic retinopathy", "G": "glaucoma (glaucomatous optic neuropathy)"}
QUESTION = ("You are an ophthalmologist screening a colour fundus photograph. Which finding best describes this image?\n"
            "{options}\nAnswer with the letter only.")


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load(model_id):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    dtype = torch.bfloat16 if device() == "cuda" else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(device()).eval()
    return model, AutoProcessor.from_pretrained(model_id)


@torch.no_grad()
def score(model, proc, image, classes, order):
    letters = "ABC"[:len(classes)]
    opts = "\n".join(f"{letters[k]}. {OPTION_TEXT[classes[c]]}" for k, c in enumerate(order))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION.format(options=opts)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[image], return_tensors="pt").to(device())
    inp = {k: (v.to(model.dtype) if v.is_floating_point() else v) for k, v in inp.items()}
    logits = model(**inp).logits[0, -1].float()
    ids = [proc.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
    z = logits[ids].cpu().numpy()
    out = np.empty(len(classes))
    out[list(order)] = z            # letter k answers class order[k]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    ap.add_argument("--splits", nargs="*", default=["calibration", "test", "test_ood", "test_openset"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, classes = load_table(a.bench)
    t = t[t.split.isin(a.splits)].reset_index(drop=True)
    if a.limit:
        t = pd.concat([g.sample(min(a.limit, len(g)), random_state=0) for _, g in t.groupby("split")]).reset_index(drop=True)
    tag = a.model.split("/")[-1]
    res_file = out / "results.jsonl"
    if res_file.exists() and any(json.loads(l)["model"] == tag for l in res_file.read_text().splitlines()):
        print(f"[done] {tag}", flush=True)      # a resumed sweep must not append the same rows twice
        return
    zf = out / f"logits_{tag}.npy"          # images x orders x classes
    model, proc = load(a.model)
    orders = list(permutations(range(len(classes))))
    Z = np.load(zf) if zf.exists() else np.full((len(t), len(orders), len(classes)), np.nan, np.float32)
    for i, r in enumerate(t.itertuples()):
        if not np.isnan(Z[i]).any():
            continue
        im = Image.open(Path(a.images) / r.source / f"{r.image_id}.jpg").convert("RGB")
        for k, o in enumerate(orders):
            Z[i, k] = score(model, proc, im, classes, o)
        if i % 200 == 0:
            np.save(zf, Z)
            print(f"{i}/{len(t)}", flush=True)
    np.save(zf, Z)
    with open(out / "results.jsonl", "a") as fh:
        rows = []
        for k, o in enumerate(orders):
            rows += evaluate(Z[:, k], t, {"setting": "mcq-single-order", "model": tag, "loss": "".join(classes[c] for c in o), "seed": k}, n_boot=0)
        rows += evaluate(Z.mean(1), t, {"setting": "mcq-order-average", "model": tag, "loss": "avg6", "seed": 0})
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        for r in rows[-3:]:
            print(tag, r["posthoc"], f"test bAcc {r['test']['bacc']:.3f} ECE {r['test']['ece']:.3f}", flush=True)


if __name__ == "__main__":
    main()
