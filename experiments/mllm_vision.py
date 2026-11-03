"""Image features from the vision side of multimodal LLMs: the image tokens each model passes to its language model (after its projector), mean-pooled.
A frozen probe on these features asks whether a multimodal LLM's poor multiple-choice answers come from what it sees or from how it answers.
python -m experiments.mllm_vision --bench data/bench --images data/processed --out runs/features --model Qwen/Qwen2.5-VL-7B-Instruct
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import load_table

NAMES = {"Qwen/Qwen2.5-VL-7B-Instruct": "mllm_qwen25vl", "Kimokcheon/Fundus-R1-7B": "mllm_fundusr1",
         "llava-hf/llava-onevision-qwen2-7b-ov-hf": "mllm_llava_ov", "chaoyinshe/llava-med-v1.5-mistral-7b-hf": "mllm_llava_med"}


def pooled(out):
    if isinstance(out, (list, tuple)):
        out = torch.cat([o.reshape(-1, o.shape[-1]) for o in out])
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        out = out.pooler_output
    elif hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    return out.reshape(-1, out.shape[-1]).float().mean(0)


def main(argv=None):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    ap = argparse.ArgumentParser()
    for k in ["bench", "images", "out", "model"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    t, _ = load_table(a.bench)
    keep = np.flatnonzero(t.split.isin(["train", "validation", "calibration", "test", "test_ood"]).to_numpy())
    if a.limit:
        keep = keep[:a.limit]
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    ip = proc.image_processor
    if hasattr(ip, "max_pixels"):  # Qwen2.5-VL family: fix the resolution to 448 x 448 for every image
        ip.min_pixels = ip.max_pixels = 448 * 448
    name = NAMES[a.model] + ("_smoke" if a.limit else "")
    out = Path(a.out) / f"{name}.npy"
    done_f = Path(a.out) / f"{name}.done.npy"
    F = None
    start = int(np.load(done_f)) if done_f.exists() else 0
    with torch.no_grad():
        for n, i in enumerate(keep[start:], start):
            r = t.iloc[i]
            im = Image.open(Path(a.images) / r.source / f"{r.image_id}.jpg").convert("RGB")
            px = ip(images=[im], return_tensors="pt")
            kw = {k: (v.to("cuda", torch.bfloat16) if v.is_floating_point() else v.to("cuda")) for k, v in px.items() if torch.is_tensor(v)}
            f = pooled(model.get_image_features(**kw)).cpu().numpy()
            if F is None:
                F = np.lib.format.open_memmap(out, mode="r+" if start else "w+", dtype=np.float32, shape=(len(t), f.shape[0]))
            F[i] = f
            if n % 500 == 0:
                F.flush()
                np.save(done_f, n)
                print(name, n, len(keep), flush=True)
    F.flush()
    np.save(done_f, len(keep))
    print("[ok]", name, F.shape, flush=True)


if __name__ == "__main__":
    main()
