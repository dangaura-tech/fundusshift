"""Looks at N images of a held-out site read as G against those read as N: montage, and simple image statistics. Development rows only.
python -m experiments.inspect_false_g --bench data/bench --images data/processed --zeroshot runs/medsiglip --logits results/strata/gate11 --site ddr --out results/strata/ddr_false_g
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .data import load_table
from .strata_gate4 import BANKS
from .strata_gate7 import scaled
from .strata_tokens import locate_disc


def stats(path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im.resize((224, 224), Image.BILINEAR)).astype(np.float32)
    m = a.sum(2) > 30
    x, y = locate_disc(a.astype(np.uint8))
    fg = a[m] if m.any() else a.reshape(-1, 3)
    return dict(w=im.size[0], h=im.size[1], fov=m.mean(), r=fg[:, 0].mean(), g=fg[:, 1].mean(), b=fg[:, 2].mean(), contrast=fg[:, 1].std(), disc_x=x / 224, disc_y=y / 224)


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "images", "zeroshot", "logits", "site", "out"]:
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    t, _ = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    s = a.site
    va, held = (sp == "validation") & (src != s), np.isin(sp, ["train", "validation"]) & (src == s)
    T = [scaled(np.load(Path(a.zeroshot) / f"logits_medsiglip_{b}.npy").astype(np.float32), va)[held] for b in BANKS]
    head = np.mean([np.load(Path(a.logits) / f"medsiglip_{s}_s{i}.npy") for i in range(3)], 0)
    z, sub = .5 * head + .5 * np.mean(T, 0), t[held].reset_index(drop=True)
    print(list(sub.columns))
    yy, pred = y[held], z.argmax(1)
    print("N read as G: head alone", ((head.argmax(1) == 2) & (yy == 0)).sum(), "| text alone", ((np.mean(T, 0).argmax(1) == 2) & (yy == 0)).sum(), "| both", ((head.argmax(1) == 2) & (np.mean(T, 0).argmax(1) == 2) & (yy == 0)).sum(), "| of N", (yy == 0).sum())
    margin = z[:, 2] - z[:, 0]
    fg, ok = np.flatnonzero((yy == 0) & (pred == 2)), np.flatnonzero((yy == 0) & (pred == 0))
    rng = np.random.default_rng(0)
    pick = lambda v, n: rng.choice(v, min(n, len(v)), replace=False)
    file = lambda i: Path(a.images) / s / f"{sub.image_id[i]}.jpg"
    S = {n: pd.DataFrame([stats(file(i)) for i in pick(v, 400)]) for n, v in [("N_as_G", fg), ("N_as_N", ok)]}
    print(pd.DataFrame({k: v.mean() for k, v in S.items()}).round(3).to_string())
    print(pd.DataFrame({k: v.std() for k, v in S.items()}).round(3).to_string())
    print("sizes N_as_G", S["N_as_G"].groupby(["w", "h"]).size().sort_values().tail(4).to_dict())
    print("sizes N_as_N", S["N_as_N"].groupby(["w", "h"]).size().sort_values().tail(4).to_dict())
    q = ["fov_fraction", "brightness", "contrast", "sharpness", "dark_clip", "bright_clip", "mean_r", "mean_g", "mean_b", "orig_width", "orig_height"]
    print(pd.DataFrame({"N_as_G": sub.loc[fg, q].apply(pd.to_numeric, errors="coerce").mean(), "N_as_N": sub.loc[ok, q].apply(pd.to_numeric, errors="coerce").mean(), "sd_all_N": sub.loc[yy == 0, q].apply(pd.to_numeric, errors="coerce").std()}).round(3).to_string())
    print("flag rate N_as_G", sub.loc[fg, "quality_flags"].fillna("").astype(str).value_counts(normalize=True).head(5).round(3).to_dict())
    print("flag rate N_as_N", sub.loc[ok, "quality_flags"].fillna("").astype(str).value_counts(normalize=True).head(5).round(3).to_dict())
    tile, n = 180, 8
    for name, v in [("N_as_G_top", fg[np.argsort(-margin[fg])[:n * n]]), ("N_as_G_random", pick(fg, n * n)), ("N_as_N_random", pick(ok, n * n))]:
        M = Image.new("RGB", (n * tile, n * tile))
        for k, i in enumerate(v):
            im = Image.open(file(i)).convert("RGB").resize((tile, tile))
            x, y0 = locate_disc(np.asarray(im.resize((224, 224))))
            ImageDraw.Draw(im).ellipse([x / 224 * tile - 18, y0 / 224 * tile - 18, x / 224 * tile + 18, y0 / 224 * tile + 18], outline=(0, 255, 0), width=2)
            M.paste(im, (k % n * tile, k // n * tile))
        M.save(Path(a.out) / f"{name}.jpg", quality=85)


if __name__ == "__main__":
    main()
