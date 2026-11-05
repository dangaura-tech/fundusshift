"""Do fine-tuned features encode the acquisition site more than frozen ones? A logistic probe predicts the source from features, within one class to break the
label-source link, by 5-fold cross-validation on development (validation + calibration) rows and on test rows; a disease probe on the same rows is the reference.
Fine-tuned features are the penultimate features cached by experiments.followup (gate); frozen ones come from runs/features and data/strata. Also draws t-SNE maps.
python -m experiments.site_probe --bench data/bench --features runs/features --strata data/strata --ckpt ckpt --out results/review
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_table
from .strata_gate4 import nap


def cv_bacc(F, lab, seed=0):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=2000, class_weight="balanced"))
    pred = cross_val_predict(clf, F, lab, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return balanced_accuracy_score(lab, pred)


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "features", "strata", "ckpt", "out"]:
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    keep = np.flatnonzero(np.isin(sp, ["validation", "calibration", "test", "test_ood", "test_openset"]))
    tr = sp == "train"
    M = np.load(Path(a.strata) / "mean3_medsiglip.npy")[:, 2].astype(np.float32)
    Xs = (M - M[tr].mean(0)) / (M[tr].std(0) + 1e-6)
    U = nap(Xs[tr], y[tr], src[tr], 32)
    feats = {
        "Fine-tuned Swin-T": np.load(Path(a.ckpt) / "_gate_features" / "finetune_swin_t_mbls_s0.npy"),
        "Group DRO ConvNeXt-T": np.load(Path(a.ckpt) / "_gate_features" / "robust_convnext_t_groupdro_s0.npy"),
        "ERM DINOv2 LoRA": np.load(Path(a.ckpt) / "_gate_features" / "robust_dinov2_lora_erm_s0.npy"),
        "Frozen MedSigLIP (pooled)": np.load(Path(a.features) / "medsiglip.npy").astype(np.float32)[keep],
        "Frozen MedSigLIP tokens (LOTUS input)": Xs[keep],
        "LANTERN input after projection": (Xs - (Xs @ U.T) @ U)[keep],
    }
    spk, yk, srck = sp[keep], y[keep], src[keep]
    views = {"development": np.isin(spk, ["validation", "calibration"]), "test": spk == "test"}
    rows = []
    for name, F in feats.items():
        for vname, m in views.items():
            r = dict(features=name, view=vname, n=int(m.sum()), disease=cv_bacc(F[m], yk[m]), site_all=cv_bacc(F[m], srck[m]))
            for c, cn in [(0, "N"), (1, "D")]:
                mm = m & (yk == c)
                ok = pd.Series(srck[mm]).value_counts()
                mm &= np.isin(srck, ok[ok >= 10].index)
                r[f"site_within_{cn}"] = cv_bacc(F[mm], srck[mm]) if len(np.unique(srck[mm])) > 1 else np.nan
                r[f"sites_{cn}"] = int(len(np.unique(srck[mm])))
            rows.append(r)
            print(name, vname, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items() if k not in ("features", "view")}, flush=True)
    d = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "site_probe.csv", index=False)

    # t-SNE of test rows, coloured by source, for a fine-tuned and a frozen encoder
    m = np.flatnonzero(spk == "test")
    rng = np.random.default_rng(0)
    m = np.sort(rng.choice(m, min(2500, len(m)), replace=False))
    names = {"afio_composite": "AFIO", "fives": "FIVES", "hrf": "HRF", "hygd": "HYGD", "palm": "PALM", "papila": "PAPILA", "paraguay_dr": "Paraguay"}
    cols = dict(zip(names, ["#2F6DB5", "#2A9D8F", "#E08A1E", "#8E5BB5", "#C8455A", "#6B8E23", "#7F7F7F"]))
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.3))
    for ax, name in zip(axs, ["Fine-tuned Swin-T", "Frozen MedSigLIP (pooled)", "Frozen MedSigLIP tokens (LOTUS input)"]):
        E = TSNE(n_components=2, perplexity=40, init="pca", random_state=0).fit_transform(StandardScaler().fit_transform(feats[name][m]))
        for s_, c_ in cols.items():
            for k, mk in zip(range(3), "o^s"):
                q = (srck[m] == s_) & (yk[m] == k)
                if q.any():
                    ax.scatter(E[q, 0], E[q, 1], s=2.5, c=c_, marker=mk, linewidths=0, alpha=.7)
        ax.set_title(name.replace(" (LOTUS input)", "\n(LOTUS input)"), fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])
    h = [plt.Line2D([], [], ls="", marker="o", ms=3, color=c_, label=names[s_]) for s_, c_ in cols.items()]
    h += [plt.Line2D([], [], ls="", marker=mk, ms=3, color="#333333", label=cn) for mk, cn in zip("o^s", ["N", "D", "G"])]
    fig.legend(handles=h, loc="lower center", ncol=10, fontsize=5.5, frameon=False, bbox_to_anchor=(.5, -.06))
    fig.savefig(Path("paper/figures/tsne.pdf"), bbox_inches="tight")
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
