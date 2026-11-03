"""Step 3b: standardize released images + per-image quality statistics.

For every image in splits.csv (split != unused, plus a quality row for every readable image):
  - crop to the illuminated fundus field (black border removed), resize longest side to --size, JPEG q95
  - quality stats on the original: field-of-view fraction, mean brightness, RMS contrast, Laplacian sharpness,
    clipped-dark / clipped-bright fractions
python -m fsbench.process_images --raw data/raw --bench data/bench --out data/processed --size 1024
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .hashing import crop_fundus

Image.MAX_IMAGE_PIXELS = None


def _quality(img):
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    small = np.asarray(img.convert("L").resize((512, int(512 * img.height / img.width) or 1)), dtype=np.float32)
    fov = rgb.max(axis=2) > 15
    g = small
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    inside = rgb[fov] if fov.any() else rgb.reshape(-1, 3)
    lum = inside.mean(axis=1)
    return {"fov_fraction": float(fov.mean()), "brightness": float(lum.mean()), "contrast": float(lum.std()),
            "sharpness": float(lap.var()), "dark_clip": float((lum < 10).mean()), "bright_clip": float((lum > 245).mean()),
            "mean_r": float(inside[:, 0].mean()), "mean_g": float(inside[:, 1].mean()), "mean_b": float(inside[:, 2].mean())}


def _work(args):
    src_path, dst_path, size = args
    try:
        with Image.open(src_path) as im:
            im.load()
            im = im.convert("RGB")
            q = _quality(im)
            c = crop_fundus(im)
            if max(c.size) > size:
                s = size / max(c.size)
                c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.LANCZOS)
            if dst_path:
                Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
                c.save(dst_path, "JPEG", quality=95)
            return {**q, "proc_width": c.width, "proc_height": c.height, "proc_error": ""}
    except Exception as e:
        return {"proc_error": repr(e)[:200]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    s = pd.read_csv(Path(a.bench) / "splits.csv", dtype=str, keep_default_na=False)
    s = s[s.unused_reason != "file_missing"]
    jobs = [(str(Path(a.raw) / r.source / r.rel_path),
             str(Path(a.out) / r.source / f"{r.image_id}.jpg") if r.split != "unused" else "", a.size)
            for r in s.itertuples()]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_work, jobs, chunksize=32))
    q = pd.DataFrame(res)
    q.insert(0, "image_id", s.image_id.to_numpy())
    q.to_csv(Path(a.bench) / "quality.csv", index=False)
    print(f"processed {len(q)} images; errors: {(q.proc_error.fillna('') != '').sum()}")


if __name__ == "__main__":
    main()
