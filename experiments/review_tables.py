"""Further reviewer tables from saved logits on the published split: per-source accuracy, bAcc, macro-F1 and ECE; disease-level one-vs-rest metrics for D and G;
a label-scope-compatible test subset; risk-coverage curves; and a gallery of LANTERN's errors.
python -m experiments.review_tables --bench data/bench --images data/processed --runs runs --out results/review
LOTUS: add --method-dir lotus --method-name LOTUS --fig-dir paper/latex/figures (the error gallery is skipped when --images does not exist).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .calib import Calibrator, ece
from .data import load_table
from .full_eval import coverage_threshold, discover

REPS = [("LANTERN", None), ("16-shot probe, MedSigLIP", ("16-shot-probe", "medsiglip", "none/ce")), ("Linear probe, MedSigLIP", ("linear-probe", "medsiglip", "class/ls")),
        ("Zero-shot, MedSigLIP", ("zero-shot", "medsiglip", "descriptive_single")), ("Group DRO, ConvNeXt-T", ("groupdro", "convnext_t", "ce")),
        ("ERM, DINOv2 LoRA", ("erm", "dinov2_lora", "ce")), ("Fine-tune, Swin-T", ("finetune", "swin_t", "mbls"))]
NAMES = {"afio_composite": "AFIO", "fives": "FIVES", "hrf": "HRF", "hygd": "HYGD", "palm": "PALM", "papila": "PAPILA", "paraguay_dr": "Paraguay", "chaksu": "Chákṣu"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "images", "runs", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--method-dir", default="top_dev/last_reg", help="folder under --runs with the method's logits_s{0,1,2}.npy")
    ap.add_argument("--method-name", default="LANTERN")
    ap.add_argument("--fig-dir", default="paper/figures")
    a = ap.parse_args(argv)
    REPS[0] = (a.method_name, None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, classes = load_table(a.bench)
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    ca, cb, te, ch = (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B"), sp == "test", sp == "test_ood"
    files = {(s, m, str(c), seed): f for s, m, c, seed, f in discover(a.runs)}
    probs = {}
    for name, key in REPS:
        Ps = []
        for seed in range(3):
            f = Path(a.runs) / a.method_dir / f"logits_s{seed}.npy" if key is None else files.get((*key, seed))
            if f is None and seed == 0:
                f = files.get((*key, 0))
            if f is None:
                continue
            Z = np.load(f).astype(np.float64)
            Ps.append(Calibrator("ts").fit(Z[ca], y[ca])(Z))
        probs[name] = Ps

    # per source
    rows = []
    for name, Ps in probs.items():
        for s in ["paraguay_dr", "hygd", "fives", "papila", "palm", "hrf", "afio_composite", "chaksu"]:
            m = (src == s) & (te | ch)
            cl = sorted(set(y[m]))
            r = dict(method=name, source=s, n=int(m.sum()), classes="/".join(classes[c] for c in cl))
            r["acc"] = np.mean([(P[m].argmax(1) == y[m]).mean() for P in Ps])
            r["bacc"] = np.mean([balanced_accuracy_score(y[m], P[m].argmax(1)) for P in Ps])
            r["macro_f1"] = np.mean([f1_score(y[m], P[m].argmax(1), labels=cl, average="macro", zero_division=0) for P in Ps])
            r["ece_ts"] = np.mean([ece(P[m], y[m])[0] for P in Ps])
            rows.append(r)
    pd.DataFrame(rows).to_csv(out / "per_source.csv", index=False)

    # disease level: one-vs-rest on pooled test and on Chakṣu
    rows = []
    for name, Ps in probs.items():
        for view, m in [("test", te), ("chaksu", ch)]:
            for c in [1, 2]:
                if not (y[m] == c).any():
                    continue
                pos = y[m] == c
                pred = [P[m].argmax(1) == c for P in Ps]
                rows.append(dict(method=name, view=view, disease=classes[c], sensitivity=np.mean([p[pos].mean() for p in pred]), specificity=np.mean([1 - p[~pos].mean() for p in pred]),
                                 auroc=np.mean([roc_auc_score(pos, P[m][:, c]) for P in Ps]), ppv=np.mean([(pos[p]).mean() if p.any() else np.nan for p in pred])))
    pd.DataFrame(rows).to_csv(out / "disease_level.csv", index=False)

    # label-scope-compatible subset: test sources that graded every class and hold more than one
    full = te & np.isin(src, ["fives", "papila", "hrf", "afio_composite"])
    rows = []
    for name, Ps in probs.items():
        rows.append(dict(method=name, all_test=np.mean([balanced_accuracy_score(y[te], P[te].argmax(1)) for P in Ps]),
                         multiclass_full_scope=np.mean([balanced_accuracy_score(y[full], P[full].argmax(1)) for P in Ps]),
                         without_single_class=np.mean([balanced_accuracy_score(y[te & ~np.isin(src, ["hygd", "palm"])], P[te & ~np.isin(src, ["hygd", "palm"])].argmax(1)) for P in Ps]),
                         n_full=int(full.sum())))
    pd.DataFrame(rows).to_csv(out / "scope_subset.csv", index=False)

    # risk-coverage on test, seed 0, with the Cal-B 80% threshold marked
    plt.rcParams.update({"font.size": 7})
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.1))
    cols = ["#111111", "#E08A1E", "#2A9D8F", "#8E5BB5", "#C8455A", "#2F6DB5", "#7F7F7F"]
    for (name, Ps), col in zip(probs.items(), cols):
        P = Ps[0]
        for ax, m in zip(axs, [te, ch]):
            conf = P[m].max(1)
            o = np.argsort(-conf)
            err = (P[m].argmax(1) != y[m])[o]
            cov = np.arange(1, len(o) + 1) / len(o)
            ax.plot(cov, np.cumsum(err) / np.arange(1, len(o) + 1), color=col, lw=1.4 if name == a.method_name else .8, label=name)
            thr = coverage_threshold(P[cb].max(1), .8)
            acc = conf >= thr
            ax.scatter([acc.mean()], [(P[m].argmax(1) != y[m])[acc].mean()], color=col, s=10, zorder=4)
    for ax, ttl in zip(axs, ["Test", "Chákṣu"]):
        ax.axvline(.8, ls="--", lw=.5, color="#888888")
        ax.set_xlabel("Coverage")
        ax.set_title(ttl, fontsize=7)
        ax.grid(lw=.3, color="#D9D9D9")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Selective risk")
    axs[1].legend(frameon=False, fontsize=5.5, loc="upper left")
    fig.savefig(Path(a.fig_dir) / "risk_coverage.pdf", bbox_inches="tight")

    # failure gallery: the method's seed 0, the most confident errors by type
    if not Path(a.images).exists():
        print("no images: error gallery skipped")
        return
    P = probs[a.method_name][0]
    pred, conf = P.argmax(1), P.max(1)
    picks = []
    for tc, pc in [(2, 0), (2, 1), (0, 2), (1, 0), (0, 1), (1, 2)]:
        m = np.flatnonzero((te | ch) & (y == tc) & (pred == pc))
        m = m[np.argsort(-conf[m])]
        seen = set()
        for i in m:
            if src[i] not in seen:
                picks.append(i)
                seen.add(src[i])
            if len(seen) == 3:
                break
    picks = picks[:16]
    rows = []
    fig, axs = plt.subplots(2, 8, figsize=(6.6, 2.1))
    for ax, i in zip(axs.ravel(), picks):
        ax.imshow(np.asarray(Image.open(Path(a.images) / src[i] / f"{t.image_id.iloc[i]}.jpg").convert("RGB").resize((224, 224))))
        ax.set_title(f"{NAMES[src[i]]}\n{classes[y[i]]}$\\to${classes[pred[i]]} ({conf[i]:.2f})", fontsize=5, pad=1)
        ax.axis("off")
        rows.append(dict(source=src[i], true=classes[y[i]], pred=classes[pred[i]], confidence=conf[i], image_id=t.image_id.iloc[i]))
    for ax in axs.ravel()[len(picks):]:
        ax.axis("off")
    fig.subplots_adjust(wspace=.05, hspace=.35)
    fig.savefig(Path(a.fig_dir) / "failures.pdf", bbox_inches="tight", dpi=200)
    pd.DataFrame(rows).to_csv(out / "failures.csv", index=False)
    pd.set_option("display.width", 220)
    for f_ in ["per_source", "disease_level", "scope_subset"]:
        print(pd.read_csv(out / f"{f_}.csv").round(3).to_string())
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
