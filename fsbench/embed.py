"""Step 1b: DINOv2 embeddings of the cropped fundus for near-duplicate detection.

64-bit perceptual hashes are too coarse for fundus photographs (every image is an orange disc with a darker
macula), so near-duplicates are found with self-supervised embeddings instead, with a threshold calibrated on
synthetic copies (see `calibrate`).
python -m fsbench.embed --raw data/raw --bench data/bench [--model facebook/dinov2-small]
Writes embeddings.npy (float16, L2-normalised, row order = manifest.csv) and embed_calibration.csv
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from .hashing import crop_fundus

Image.MAX_IMAGE_PIXELS = None


class _DS(torch.utils.data.Dataset):
    def __init__(self, paths, proc, transform=None):
        self.paths, self.proc, self.transform = paths, proc, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = crop_fundus(Image.open(self.paths[i]).convert("RGB"))
            if self.transform:
                im = self.transform(im, i)
        except Exception:
            im = Image.new("RGB", (224, 224))
        return self.proc(images=im, return_tensors="pt")["pixel_values"][0]


def embed(paths, model, proc, device, transform=None, bs=64):
    dl = torch.utils.data.DataLoader(_DS(paths, proc, transform), batch_size=bs, num_workers=6)
    out = []
    with torch.no_grad():
        for x in dl:
            h = model(pixel_values=x.to(device)).pooler_output
            out.append(torch.nn.functional.normalize(h, dim=1).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def _copy_transform(im, i):
    """Typical re-distribution edits: rescale, JPEG recompression, small crop, brightness change."""
    rng = np.random.default_rng(i)
    s = rng.uniform(0.35, 0.8)
    im = im.resize((max(32, int(im.width * s)), max(32, int(im.height * s))))
    c = rng.uniform(0, 0.04)
    im = im.crop((int(im.width * c), int(im.height * c), int(im.width * (1 - c)), int(im.height * (1 - c))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(rng.integers(60, 90)))
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return Image.eval(im, lambda v: min(255, int(v * rng.uniform(0.9, 1.1))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--n-calib", type=int, default=600)
    a = ap.parse_args(argv)
    bench = Path(a.bench)
    m = pd.read_csv(bench / "manifest.csv", dtype=str, keep_default_na=False)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(device).eval()
    paths = [str(Path(a.raw) / s / p) for s, p in zip(m.source, m.rel_path)]
    ok = (m.sha256 != "").to_numpy()
    E = np.zeros((len(m), model.config.hidden_size), np.float16)
    E[ok] = embed([p for p, k in zip(paths, ok) if k], model, proc, device)
    np.save(bench / "embeddings.npy", E)

    # calibration: cosine(original, edited copy) vs cosine(original, nearest *other-patient* image of same source)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(ok), size=min(a.n_calib, ok.sum()), replace=False)
    C = embed([paths[i] for i in idx], model, proc, device, transform=_copy_transform)
    copy_cos = (C.astype(np.float32) * E[idx].astype(np.float32)).sum(1)
    Ef = E.astype(np.float32)
    nn_other = []
    for i in idx:
        same_src = (m.source == m.source.iloc[i]).to_numpy() & ok
        other = same_src & ~((m.patient_id == m.patient_id.iloc[i]).to_numpy() & (m.patient_id.iloc[i] != ""))
        other[i] = False
        nn_other.append(float((Ef[other] @ Ef[i]).max()) if other.any() else np.nan)
    cal = pd.DataFrame({"image_id": m.image_id.to_numpy()[idx], "source": m.source.to_numpy()[idx],
                        "cos_edited_copy": copy_cos, "cos_nearest_other_image": nn_other})
    cal.to_csv(bench / "embed_calibration.csv", index=False)
    q = cal[["cos_edited_copy", "cos_nearest_other_image"]].quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    print(q.to_string())


if __name__ == "__main__":
    main()
