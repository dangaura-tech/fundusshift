"""Image-level dataset validation: frozen backbone + head fit on the released train features, test images changed at inference only.

V9 preprocessing : JPEG quality, low resolution, gamma, no field crop (raw photograph)
V10 occlusion    : optic-disc region removed or kept alone, vessels blurred away, periphery removed. The disc position is a brightness heuristic, not a segmentation.
V11 excluded labels: how a head scores the labels the benchmark excludes (PAPILA suspect, HYGD GON-)
python -m experiments.data_validation_images --bench data/bench --images data/processed --raw data/raw --features runs/features --out results/data_validation
"""
import argparse
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score

from fsbench.process_images import _work

from .backbones import feature_extractor
from .data import load_table
from .data_exps import cells, fit_eval
from .extract_features import device

VARIANTS = ["clean", "jpeg50", "jpeg20", "res128", "gamma07", "gamma14", "raw_uncropped", "disc_removed", "disc_only", "vessels_blurred", "periphery_removed"]


def disc_center(rgb):
    """Peak of a difference of Gaussians on the green channel inside the field. Against PAPILA expert contours (488 images) the centre falls inside the disc in 96.9 %."""
    small = cv2.resize(rgb, (256, int(256 * rgb.shape[0] / rgb.shape[1])))
    g = small[..., 1].astype(np.float32)
    m = (small.max(2) > 15).astype(np.float32)
    norm = lambda sg: cv2.GaussianBlur(g * m, (0, 0), sg) / (cv2.GaussianBlur(m, (0, 0), sg) + 1e-3)
    d = norm(256 * .035) - norm(256 * .11)
    d[~cv2.erode(m.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)] = -1e9
    y, x = np.unravel_index(d.argmax(), d.shape)
    return x / small.shape[1], y / small.shape[0]


def apply_variant(img, v):
    if v in ("clean", "raw_uncropped"):
        return img
    if v.startswith("jpeg"):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(v[4:]))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if v == "res128":
        w, h = img.size
        s = 128 / max(w, h)
        return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC).resize((w, h), Image.BICUBIC)
    a = np.asarray(img)
    if v.startswith("gamma"):
        g = {"gamma07": .7, "gamma14": 1.4}[v]
        return Image.fromarray((255 * (a / 255.) ** g).astype(np.uint8))
    H, W = a.shape[:2]
    if v in ("disc_removed", "disc_only"):
        cx, cy = disc_center(a)
        mask = np.zeros((H, W), np.uint8)
        cv2.circle(mask, (int(cx * W), int(cy * H)), int(.14 * min(H, W)), 1, -1)
        fill = np.median(a[a.max(2) > 15].reshape(-1, 3), 0) if (a.max(2) > 15).any() else 0
        out = a.copy()
        out[(mask == 1) if v == "disc_removed" else (mask == 0)] = fill if v == "disc_removed" else 0
        return Image.fromarray(out.astype(np.uint8))
    if v == "vessels_blurred":
        k = int(.04 * max(H, W)) | 1
        return Image.fromarray(cv2.medianBlur(a, k))
    if v == "periphery_removed":
        mask = np.zeros((H, W), np.uint8)
        cv2.circle(mask, (W // 2, H // 2), int(.3 * min(H, W)), 1, -1)
        out = a.copy()
        out[mask == 0] = 0
        return Image.fromarray(out)
    raise ValueError(v)


class VariantTable(torch.utils.data.Dataset):
    def __init__(self, t, images, raw, tf, variant):
        self.t, self.images, self.raw, self.tf, self.v = t.reset_index(drop=True), Path(images), Path(raw), tf, variant

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        r = self.t.iloc[i]
        p = self.raw / r.source / r.rel_path if self.v == "raw_uncropped" else self.images / r.source / f"{r.image_id}.jpg"
        return self.tf(apply_variant(Image.open(p).convert("RGB"), self.v)), i


def embed(model, emb, t, a, tf, variant, dev):
    dl = torch.utils.data.DataLoader(VariantTable(t, a.images, a.raw, tf, variant), batch_size=a.batch, num_workers=a.workers, multiprocessing_context="spawn", persistent_workers=False)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(emb(model, x.to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["resnet50", "dinov2_b14"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variants", nargs="*", default=VARIANTS, help="subset; clean always runs first")
    ap.add_argument("--table", default="v9_v10_image_variants.csv")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, _ = load_table(a.bench)
    s = pd.read_csv(Path(a.bench) / "splits_final.csv", dtype=str, keep_default_na=False)
    excl = s[(s.unused_reason == "label:excluded_native_label") & s.source.isin(["papila", "hygd"])].reset_index(drop=True)
    y, sp = t.y.to_numpy(), t.split.to_numpy()
    te = np.flatnonzero(np.isin(sp, ["test", "test_ood"]))
    tt = t.iloc[te].reset_index(drop=True)
    rows, ex_rows = [], []
    dev = device()
    for name in a.models:
        X = np.load(Path(a.features) / f"{name}.npy").astype(np.float32)
        model, tf, emb = feature_extractor(name)
        model = model.to(dev).eval()
        tr = np.flatnonzero((sp == "train") & (y >= 0))
        _, clf, sc = fit_eval(X, y, tr, te, cells(t))
        ref = None
        for v in dict.fromkeys(["clean", *a.variants]):
            Z = embed(model, emb, tt, a, tf, v, dev)
            p = clf.predict(sc.transform(Z))
            if v == "clean":
                ref = p
                cos = (Z * X[te]).sum(1) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(X[te], axis=1) + 1e-9)
                print(name, "parity with HPC features: min cosine", float(cos.min()), "median", float(np.median(cos)), flush=True)
            ty, so, spp = tt.y.to_numpy(), tt.source.to_numpy(), tt.split.to_numpy()
            views = {"test": spp == "test", "chaksu": spp == "test_ood", **{f"src:{x}": (spp == "test") & (so == x) for x in np.unique(so[spp == "test"])}}
            for vw, m in views.items():
                rows.append({"model": name, "variant": v, "view": vw, "n": int(m.sum()), "bacc": balanced_accuracy_score(ty[m], p[m]), **{f"recall_{c}": float((p[m & (ty == i)] == i).mean()) if (m & (ty == i)).any() else np.nan for i, c in enumerate("NDG")}, "pred_changed_vs_clean": float((p[m] != ref[m]).mean())})
            print(name, v, rows[-len(views)]["bacc"], flush=True)
            pd.DataFrame(rows).to_csv(out / a.table, index=False)
        if a.table != "v9_v10_image_variants.csv":
            continue
        with tempfile.TemporaryDirectory() as d:
            for r in excl.itertuples():  # excluded images are not in the release, so run them through the same processing
                res = _work((str(Path(a.raw) / r.source / r.rel_path), f"{d}/{r.source}/{r.image_id}.jpg", 1024))
                if res.get("proc_error"):
                    raise RuntimeError(res["proc_error"])
            b = argparse.Namespace(images=d, raw=a.raw, batch=a.batch, workers=a.workers)
            Z = embed(model, emb, excl, b, tf, "clean", dev)
        p = clf.predict(sc.transform(Z))
        for (so, nat), g in excl.assign(pred=p).groupby(["source", "native_label"]):
            ex_rows.append({"model": name, "source": so, "native_label": nat, "n": len(g), "pred_N": float((g.pred == 0).mean()), "pred_D": float((g.pred == 1).mean()), "pred_G": float((g.pred == 2).mean())})
        pd.DataFrame(ex_rows).to_csv(out / "v11_excluded_labels.csv", index=False)
    # one contact sheet so the disc heuristic can be inspected
    rng = np.random.default_rng(0)
    sheet = []
    for r in tt.iloc[rng.choice(len(tt), 24, replace=False)].itertuples():
        a0 = np.asarray(Image.open(Path(a.images) / r.source / f"{r.image_id}.jpg").convert("RGB").resize((180, 180)))
        full = Image.open(Path(a.images) / r.source / f"{r.image_id}.jpg").convert("RGB")
        sheet.append(np.concatenate([a0, np.asarray(apply_variant(full, "disc_removed").resize((180, 180)))], 0))
    rows_img = [np.concatenate(sheet[i:i + 12], 1) for i in (0, 12)]
    Image.fromarray(np.concatenate(rows_img, 0)).save(out / "v10_disc_heuristic_check.png")


if __name__ == "__main__":
    main()
