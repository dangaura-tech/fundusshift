"""Frozen-feature linear probe: a fast difficulty check of the FundusShift splits.

Features: frozen backbone (default DINOv2-small, CLS + mean patch token) on the released images.
Classifier: logistic regression with (source, class)-balanced sample weights; C chosen on validation.
Reports balanced accuracy / macro-F1 per split and per source, worst test source, and open-set AUROC.

python baselines/linear_probe.py --bench data/bench --processed data/processed --out runs/probe_dinov2s
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel

CANON = ["N", "D", "G", "A", "M"]
SPLITS = ["train", "validation", "calibration", "test", "test_ood", "test_openset"]


class _DS(torch.utils.data.Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.proc(images=Image.open(self.paths[i]).convert("RGB"), return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def features(paths, model_name, device):
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    dl = torch.utils.data.DataLoader(_DS(paths, proc), batch_size=64, num_workers=6)
    out = []
    for x in dl:
        h = model(pixel_values=x.to(device)).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
    return np.concatenate(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/dinov2-small")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(Path(a.bench) / "splits_final.csv", dtype=str, keep_default_na=False)
    d = d[d.split.isin(SPLITS)].reset_index(drop=True)
    feat_file = out / "features.npy"
    if feat_file.exists():
        X = np.load(feat_file)
    else:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        X = features([str(Path(a.processed) / s / f"{i}.jpg") for s, i in zip(d.source, d.image_id)], a.model, device)
        np.save(feat_file, X)
    CLASSES = [c for c in CANON if c in set(d.label[d.split == "train"])]
    y = np.array([CLASSES.index(l) if l in CLASSES else -1 for l in d.label])
    sp = d.split.to_numpy()
    tr = sp == "train"
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    cell = pd.Series(list(zip(d.source[tr], d.label[tr])))
    w = (1.0 / cell.map(cell.value_counts())).to_numpy().copy()
    w *= len(w) / w.sum()

    best = None
    for C in [0.001, 0.003, 0.01, 0.03, 0.1]:
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xs[tr], y[tr], sample_weight=w)
        va = sp == "validation"
        b = balanced_accuracy_score(y[va], clf.predict(Xs[va]))
        print(f"C={C}: val bacc {b:.4f}")
        if best is None or b > best[0]:
            best = (b, C, clf)
    _, C, clf = best
    P = clf.predict_proba(Xs)
    res = {"model": a.model, "C": C}
    for s in ["train", "validation", "calibration", "test", "test_ood"]:
        m = sp == s
        pred = P[m].argmax(1)
        per_src = {src: round(balanced_accuracy_score(y[m][d.source[m] == src], pred[d.source[m] == src]), 4)
                   for src in sorted(set(d.source[m]))}
        recall = {c: round(float((pred[y[m] == k] == k).mean()), 4) for k, c in enumerate(CLASSES) if (y[m] == k).any()}
        res[s] = {"balanced_accuracy": round(balanced_accuracy_score(y[m], pred), 4),
                  "macro_f1": round(f1_score(y[m], pred, average="macro"), 4), "recall": recall,
                  "per_source_bacc": per_src, "worst_source_bacc": min(per_src.values())}
    t, o = sp == "test", sp == "test_openset"
    res["test_openset_auroc_msp"] = round(roc_auc_score(np.r_[np.zeros(t.sum()), np.ones(o.sum())],
                                                         np.r_[1 - P[t].max(1), 1 - P[o].max(1)]), 4)
    res["classes"] = CLASSES

    # task 3: DR grading (ICDR 0-4) on images that carry a grade
    g = pd.to_numeric(d.dr_grade, errors="coerce").fillna(-1).astype(int).to_numpy()
    gtr = tr & (g >= 0)
    if gtr.any() and ((sp == "test") & (g >= 0)).any():
        cw = pd.Series(list(zip(d.source[gtr], g[gtr])))
        wg = (1.0 / cw.map(cw.value_counts())).to_numpy().copy()
        reg = LogisticRegression(C=C, max_iter=3000).fit(Xs[gtr], g[gtr], sample_weight=wg * len(wg) / wg.sum())
        res["dr_grading"] = {}
        for s_ in ["validation", "test"]:
            m = (sp == s_) & (g >= 0)
            pred = reg.predict(Xs[m])
            res["dr_grading"][s_] = {"n": int(m.sum()), "quadratic_kappa": round(cohen_kappa_score(g[m], pred, weights="quadratic"), 4),
                                      "accuracy": round(float((pred == g[m]).mean()), 4),
                                      "sources": sorted(set(d.source[m]))}
    print(json.dumps(res, indent=2))
    (out / "probe_metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
