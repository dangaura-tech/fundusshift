"""E2 leaky arm: restores the GRAPE images FundusShift removed as copies of test images, and measures the inflation.

Clean = the released train split. Leaky = train + the removed GRAPE copies (same processing as the release, labels G).
Both heads score FIVES glaucoma images with a GRAPE copy, the other FIVES glaucoma images, and the full test split.
python -m experiments.e2_leaky --raw data/raw --bench data/bench --features runs/features --out runs/data_exps --model dinov2_b14
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from fsbench.process_images import _work

from .backbones import feature_extractor
from .data import ImageTable, load_table
from .data_exps import cells, fit_eval
from .extract_features import device


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="dinov2_b14")
    a = ap.parse_args(argv)
    t, _ = load_table(a.bench)
    X = np.load(Path(a.features) / f"{a.model}.npy").astype(np.float32)
    s = pd.read_csv(Path(a.bench) / "splits_final.csv", dtype=str, keep_default_na=False)
    rem = s[(s.unused_reason == "duplicate_of_test_side_image") & (s.source == "grape")].reset_index(drop=True)
    model, pre, emb = feature_extractor(a.model)
    model = model.to(device()).eval()
    with tempfile.TemporaryDirectory() as d:
        for r in rem.itertuples():
            out = _work((str(Path(a.raw) / r.source / r.rel_path), f"{d}/{r.source}/{r.image_id}.jpg", 1024))
            if out.get("proc_error"):
                raise RuntimeError(out["proc_error"])
        dl = torch.utils.data.DataLoader(ImageTable(rem, d, pre), batch_size=16)
        with torch.no_grad():
            Xr = np.concatenate([emb(model, x.to(device())).float().cpu().numpy() for x, _ in dl])
        # parity: the same code on released images must reproduce the cached HPC features
        chk = t.sample(8, random_state=0)
        dl = torch.utils.data.DataLoader(ImageTable(chk, Path(a.bench).parent / "processed", pre), batch_size=8)
        with torch.no_grad():
            Xc = np.concatenate([emb(model, x.to(device())).float().cpu().numpy() for x, _ in dl])
    Xh = X[chk.index]
    parity = float(((Xc * Xh).sum(1) / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(Xh, axis=1))).min())

    sp, y = t.split.to_numpy(), t.y.to_numpy()
    G = int(y[(t.label == "G").to_numpy()][0])
    pairs = pd.read_csv(Path(a.bench) / "near_duplicates.csv", dtype=str)
    pairs = pairs[pairs.linked.isin(["True", "1", "true"]) | pairs.exact.isin(["True", "1", "true"])]
    fg = pairs[(pairs.source_a.isin(["fives", "grape"])) & (pairs.source_b.isin(["fives", "grape"])) & (pairs.source_a != pairs.source_b)]
    copied_ids = set(fg.image_id_a[fg.source_a == "fives"]) | set(fg.image_id_b[fg.source_b == "fives"])
    fives_g = (t.source == "fives").to_numpy() & (t.label == "G").to_numpy()
    copied = fives_g & t.image_id.isin(copied_ids).to_numpy()
    tr, te = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "test")
    keys = cells(t)

    Xa = np.vstack([X, Xr])
    ya = np.r_[y, np.full(len(rem), G)]
    ka = np.r_[keys, [f"grape|{G}"] * len(rem)]
    tr_l = np.r_[tr, np.arange(len(X), len(Xa))]
    rows = []
    for name, idx in [("clean (released train)", tr), ("leaky (+removed GRAPE copies)", tr_l)]:
        _, clf, sc = fit_eval(Xa, ya, idx, idx[:10], ka)
        pred = clf.predict(sc.transform(X))
        rows.append({"model": a.model, "train": name, "n_train": len(idx),
                     "G_recall_fives_with_copy": float((pred[copied] == G).mean()), "n_with_copy": int(copied.sum()),
                     "G_recall_fives_without_copy": float((pred[fives_g & ~copied] == G).mean()), "n_without_copy": int((fives_g & ~copied).sum()),
                     "test_bacc": float(balanced_accuracy_score(y[te], pred[te]))})
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "e2_leaky.csv", index=False)
    print(json.dumps({"removed": len(rem), "parity_min_cos": parity, "rows": rows}, indent=1), flush=True)


if __name__ == "__main__":
    main()
