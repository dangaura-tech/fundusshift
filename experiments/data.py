"""Benchmark access shared by all experiments: split table, class list, image loading."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

CANON = ["N", "D", "G", "A", "M"]
SPLITS = ["train", "validation", "calibration", "test", "test_ood", "test_openset"]
DEV = ["train", "validation", "calibration"]


def load_table(bench):
    d = pd.read_csv(Path(bench) / "splits_final.csv", dtype=str, keep_default_na=False)
    d = d[d.split.isin(SPLITS)].reset_index(drop=True)
    classes = [c for c in CANON if c in set(d.label[d.split == "train"])]
    d["y"] = [classes.index(l) if l in classes else -1 for l in d.label]
    d["dr"] = pd.to_numeric(d.dr_grade, errors="coerce").fillna(-1).astype(int)
    return d, classes


class ImageTable(torch.utils.data.Dataset):
    def __init__(self, table, image_root, transform):
        self.t, self.root, self.tf = table.reset_index(drop=True), Path(image_root), transform

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        r = self.t.iloc[i]
        return self.tf(Image.open(self.root / r.source / f"{r.image_id}.jpg").convert("RGB")), i


class Preprocess:
    """Runs before the usual transforms. clahe: on L of LAB, clip 2, 8x8 tiles. graham: 4 * image - 4 * Gaussian blur (sigma = width / 30) + 128."""

    def __init__(self, kind, transform):
        self.kind, self.tf = kind, transform

    def __call__(self, img):
        import cv2
        a = np.asarray(img)
        if self.kind == "clahe":
            lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
            lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
            a = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        elif self.kind == "graham":
            a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), a.shape[1] / 30), -4, 128)
        return self.tf(Image.fromarray(a))


def balanced_weights(keys):
    """Weight each sample by 1 / |its (source, class) cell| so every cell contributes equally."""
    s = pd.Series(list(keys))
    w = (1.0 / s.map(s.value_counts())).to_numpy().astype(np.float64)
    return w * len(w) / w.sum()
