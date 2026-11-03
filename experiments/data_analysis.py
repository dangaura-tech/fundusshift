"""Controlled data analyses that complement E1-E7 (same heads as data_exps).

A1 same images     : held-out test images scored out-of-fold by a pooled group-aware 5-fold model vs by the dev-only model
A2 error anatomy   : confusion matrix and per-class recall on test and test_ood; D recall per DR grade
A3 Bangladesh      : removing all / only N+D / only G of the collection, to separate its noise from its glaucoma supply
A4 leave-one-source: each development source: its validation+calibration images, with vs without that source in train
A5 shortcut        : the same head on 12 trivial acquisition statistics instead of image features
python -m experiments.data_analysis --bench data/bench --features runs/features --out runs/data_exps --models dinov2_b14 resnet50
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from .calib import bootstrap_ci
from .data import load_table
from .data_exps import cells, fit_eval

TRIVIAL = ["fov_fraction", "brightness", "contrast", "sharpness", "dark_clip", "bright_clip", "mean_r", "mean_g", "mean_b", "orig_width", "orig_height", "aspect"]


def predict(X, y, tr, keys):
    _, clf, sc = fit_eval(X, y, tr, tr[:10], keys)
    return clf.predict(sc.transform(X))


def a1(X, t, y, keys):
    sp, g = t.split.to_numpy(), t.group_id.to_numpy()
    pool = np.flatnonzero(np.isin(sp, ["train", "validation", "calibration", "test"]))
    oof = np.full(len(y), -1)
    for tr, te in GroupKFold(5, shuffle=True, random_state=0).split(pool, groups=g[pool]):
        oof[pool[te]] = predict(X, y, pool[tr], keys)[pool[te]]
    te = np.flatnonzero(sp == "test")
    held = predict(X, y, np.flatnonzero(sp == "train"), keys)
    ys, a, b, gs = y[te], oof[te], held[te], g[te]
    d = bootstrap_ci(lambda i: balanced_accuracy_score(ys[i], a[i]) - balanced_accuracy_score(ys[i], b[i]), gs, 500)
    per = pd.DataFrame([{"source": s, "classes": "".join(sorted(t.label[te][t.source.to_numpy()[te] == s].unique())),
                         "pooled_oof": balanced_accuracy_score(ys[m], a[m]), "held_out": balanced_accuracy_score(ys[m], b[m])}
                        for s in sorted(set(t.source.to_numpy()[te])) for m in [t.source.to_numpy()[te] == s]])
    return {"pooled_oof": balanced_accuracy_score(ys, a), "held_out": balanced_accuracy_score(ys, b), "gap": balanced_accuracy_score(ys, a) - balanced_accuracy_score(ys, b),
            "gap_ci": d}, per, held


def a2(y, held, t, classes):
    sp = t.split.to_numpy()
    rows, out = [], {}
    for s in ["validation", "test", "test_ood"]:
        m = sp == s
        cm = confusion_matrix(y[m], held[m], labels=range(len(classes)))
        out[s] = cm.tolist()
        rows += [{"split": s, "true": classes[k], "n": int(cm[k].sum()), **{f"pred_{c}": int(cm[k, j]) for j, c in enumerate(classes)},
                  "recall": float(cm[k, k] / cm[k].sum()) if cm[k].sum() else None} for k in range(len(classes))]
    D = classes.index("D")
    te = (sp == "test") & (y == D)
    gr = pd.DataFrame([{"split": s, "dr_grade": k, "n": int(m.sum()), "D_recall": float((held[m] == D).mean())}
                       for s in ["validation", "test"] for k in sorted(set(t.dr[(sp == s) & (y == D)]))
                       for m in [(sp == s) & (y == D) & (t.dr == k).to_numpy()]])
    return pd.DataFrame(rows), gr


def a3(X, t, y, keys, classes):
    sp, te = t.split.to_numpy(), np.flatnonzero(t.split == "test")
    tr = sp == "train"
    bd, G = (t.source == "bangladesh_eye").to_numpy(), y == classes.index("G")
    arms = {"all": tr, "drop all Bangladesh": tr & ~bd, "drop Bangladesh N+D, keep G": tr & ~(bd & ~G), "drop Bangladesh G, keep N+D": tr & ~(bd & G)}
    rows = []
    for k, m in arms.items():
        p = predict(X, y, np.flatnonzero(m), keys)
        ci = bootstrap_ci(lambda i: balanced_accuracy_score(y[te][i], p[te][i]), t.group_id.to_numpy()[te], 500)
        rec = {c: float((p[te][y[te] == j] == j).mean()) for j, c in enumerate(classes)}
        rows.append({"train": k, "n": int(m.sum()), "n_G": int((m & G).sum()), "test_bacc": balanced_accuracy_score(y[te], p[te]), "lo": ci[0], "hi": ci[1], **{f"recall_{c}": v for c, v in rec.items()}})
    return pd.DataFrame(rows)


def a4(X, t, y, keys):
    sp, src = t.split.to_numpy(), t.source.to_numpy()
    tr, ev = sp == "train", np.isin(sp, ["validation", "calibration"])
    full = predict(X, y, np.flatnonzero(tr), keys)
    rows = []
    for s in sorted(set(src[tr])):
        m = ev & (src == s)
        p = predict(X, y, np.flatnonzero(tr & (src != s)), keys)
        rows.append({"source": s, "classes": "".join(sorted(t.label[m].unique())), "n": int(m.sum()),
                     "with_source": balanced_accuracy_score(y[m], full[m]), "without_source": balanced_accuracy_score(y[m], p[m])})
    return pd.DataFrame(rows)


def a5(t, y, keys):
    F = t[TRIVIAL].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    sp = t.split.to_numpy()
    tr = np.flatnonzero(sp == "train")
    F = np.where(np.isnan(F), np.nanmedian(F[tr], 0), F)
    p = predict(F, y, tr, keys)
    return pd.DataFrame([{"split": s, "bacc_trivial_stats": balanced_accuracy_score(y[sp == s], p[sp == s])} for s in ["validation", "test", "test_ood"]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["dinov2_b14", "resnet50"])
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    y, keys = t.y.to_numpy(), cells(t)
    a5(t, y, keys).to_csv(Path(a.out) / "a5_trivial_shortcut.csv", index=False)
    for name in a.models:
        X = np.load(Path(a.features) / f"{name}.npy").astype(np.float32)
        d = Path(a.out) / name
        s, per, held = a1(X, t, y, keys)
        per.to_csv(d / "a1_same_images_per_source.csv", index=False)
        (d / "a1_same_images.json").write_text(json.dumps(s, indent=1))
        rec, gr = a2(y, held, t, classes)
        rec.to_csv(d / "a2_confusion.csv", index=False)
        gr.to_csv(d / "a2_dr_grade.csv", index=False)
        a3(X, t, y, keys, classes).to_csv(d / "a3_bangladesh.csv", index=False)
        a4(X, t, y, keys).to_csv(d / "a4_leave_one_source.csv", index=False)
        print(name, json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
