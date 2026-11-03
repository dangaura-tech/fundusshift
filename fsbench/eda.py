"""Step 5: exploratory analysis, quality flags and shortcut (confound) checks.

python -m fsbench.eda --bench data/bench --processed data/processed
Writes: eda/eda_report.md, eda/*.png, quality_flags in splits_final.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

CLASSES = ["N", "D", "G", "A", "M", "OPENSET"]
CLASS_COLOR = {"N": "#2a78d6", "D": "#eb6834", "G": "#1baf7a", "A": "#eda100", "M": "#e87ba4",
               "OPENSET": "#a3a29c"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SPLITS = ["train", "validation", "calibration", "test", "test_ood", "test_openset"]
TRIVIAL = ["orig_width", "orig_height", "aspect", "fov_fraction", "brightness", "contrast", "sharpness",
           "mean_r", "mean_g", "mean_b"]

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})


def _md(df, floatfmt="{:.3f}"):
    df = df.reset_index()
    fmt = lambda v: floatfmt.format(v) if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def quality_flags(d):
    flags = pd.Series("", index=d.index)
    sharp_lo = d.groupby("source").sharpness.transform(lambda s: s.quantile(0.01))
    rules = {"low_fov": d.fov_fraction < 0.15, "very_dark": d.brightness < 25, "overexposed": d.bright_clip > 0.25,
             "blurry_for_source": d.sharpness < sharp_lo, "tiny": d[["orig_width", "orig_height"]].min(axis=1) < 300}
    for name, m in rules.items():
        flags[m.fillna(False)] += name + ";"
    return flags.str.rstrip(";")


def fig_composition(u, out):
    classes = [c for c in CLASSES if c in set(u.label)]
    tab = pd.crosstab(u.source, u.label).reindex(columns=classes, fill_value=0)
    side = u.groupby("source").split.first().map(lambda s: "test" if s.startswith("test") else "dev")
    tab = tab.loc[sorted(tab.index, key=lambda s: (side[s] != "dev", -tab.loc[s].sum()))]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(tab) + 1.2))
    left = np.zeros(len(tab))
    for c in classes:
        ax.barh(range(len(tab)), tab[c], left=left, color=CLASS_COLOR[c], label=c, height=0.7,
                edgecolor=SURF, linewidth=1)
        left += tab[c].to_numpy()
    ax.set_yticks(range(len(tab)), [f"{s}  ({side[s]})" for s in tab.index])
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("images (symlog)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncols=4, loc="lower right", frameon=False, title="class")
    ax.set_title("Images per source and class", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "composition.png", dpi=150)
    plt.close(fig)


def fig_acquisition(d, out):
    srcs = d.groupby("source").size().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.3 * len(srcs) + 1.5), sharey=True)
    for ax, col, title in zip(axes, ["orig_width", "brightness", "sharpness"],
                              ["original width (px)", "mean brightness in field", "Laplacian sharpness (log)"]):
        data = [d.loc[d.source == s, col].dropna().to_numpy() for s in srcs]
        if col == "sharpness":
            data = [np.log10(x + 1) for x in data]
        ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#2a78d6", alpha=0.35, edgecolor="#2a78d6"),
                   medianprops=dict(color=INK), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(1, len(srcs) + 1), srcs)
    axes[0].invert_yaxis()
    fig.suptitle("Acquisition differences between sources", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "acquisition.png", dpi=150)
    plt.close(fig)


def fig_samples(u, processed, out, per=4):
    rng = np.random.default_rng(0)
    srcs = sorted(u.source.unique())
    fig, axes = plt.subplots(len(srcs), per, figsize=(per * 1.6, len(srcs) * 1.7))
    for i, s in enumerate(srcs):
        sub = u[u.source == s]
        pick = sub.sample(frac=1, random_state=0).groupby("label").head(per)
        pick = pick.sample(min(per, len(pick)), random_state=int(rng.integers(1e6)))
        for j in range(per):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(pick):
                r = pick.iloc[j]
                ax.imshow(Image.open(processed / s / f"{r.image_id}.jpg").resize((160, 160)))
                ax.set_title(f"{s}·{r.label}" if j == 0 else r.label, fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=110)
    plt.close(fig)


def shortcut_analysis(u):
    """How much do trivial acquisition statistics reveal source and (within source) label?"""
    X = u[TRIVIAL].fillna(0).to_numpy()
    rf = lambda: RandomForestClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=0)
    k = min(5, u.group_id.nunique())
    pred = cross_val_predict(rf(), X, u.source, groups=u.group_id, cv=GroupKFold(k))
    src_bacc = balanced_accuracy_score(u.source, pred)
    rows = []
    for s, sub in u.groupby("source"):
        for c in CLASSES[:-1]:
            y = (sub.label == c).astype(int)
            if y.sum() < 10 or (1 - y).sum() < 10:
                continue
            kk = min(5, sub.group_id.nunique())
            p = cross_val_predict(rf(), sub[TRIVIAL].fillna(0).to_numpy(), y, groups=sub.group_id,
                                  cv=GroupKFold(kk), method="predict_proba")[:, 1]
            rows.append({"source": s, "class_vs_rest": c, "n": len(sub), "auc_trivial_features": roc_auc_score(y, p)})
    return src_bacc, 1 / u.source.nunique(), pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--processed", required=True)
    a = ap.parse_args(argv)
    bench, processed = Path(a.bench), Path(a.processed)
    out = bench / "eda"
    out.mkdir(exist_ok=True)
    s = pd.read_csv(bench / "splits.csv", dtype=str, keep_default_na=False)
    s["split"] = s.split.replace({"val": "validation", "cal": "calibration"})
    man = pd.read_csv(bench / "manifest.csv", dtype=str, keep_default_na=False)[["image_id", "width", "height"]]
    q = pd.read_csv(bench / "quality.csv")
    d = s.merge(man, on="image_id", how="left").merge(q, on="image_id", how="left")
    d["orig_width"] = pd.to_numeric(d.width, errors="coerce")
    d["orig_height"] = pd.to_numeric(d.height, errors="coerce")
    d["aspect"] = d.orig_width / d.orig_height
    d["quality_flags"] = quality_flags(d)
    d.drop(columns=["width", "height"]).to_csv(bench / "splits_final.csv", index=False)

    u = d[d.split.isin(SPLITS)].copy()
    fig_composition(u, out)
    fig_acquisition(u, out)
    fig_samples(u, processed, out)
    src_bacc, chance, sc = shortcut_analysis(u)
    sc.to_csv(out / "shortcut_auc.csv", index=False)

    flagged = u[u.quality_flags != ""]
    flag_tab = flagged.quality_flags.str.split(";").explode().value_counts().rename("images")
    acq = u.groupby("source").agg(images=("image_id", "size"), median_w=("orig_width", "median"),
                                  median_h=("orig_height", "median"), brightness=("brightness", "median"),
                                  sharpness=("sharpness", "median"), fov=("fov_fraction", "median"))
    lines = ["# FundusShift — EDA & data-quality report", "",
             f"Released images: {len(u)} from {u.source.nunique()} sources.", "",
             "![composition](composition.png)", "", "## Acquisition statistics per source", "", _md(acq, "{:.1f}"),
             "", "![acquisition](acquisition.png)", "",
             "## Shortcut / confound analysis", "",
             f"A random forest on 10 trivial image statistics (resolution, aspect, field-of-view, brightness, contrast, "
             f"sharpness, mean RGB) identifies the **source** with balanced accuracy **{src_bacc:.3f}** "
             f"(chance {chance:.3f}, group 5-fold CV). Source identity is therefore almost free to learn — "
             "a model can exploit it whenever label priors differ by source.", "",
             "Within-source label predictability from the same trivial statistics (AUC; 0.5 = no shortcut). "
             "AUC > 0.8 means classes in that source were captured under different conditions, so in-source accuracy "
             "overstates disease recognition:", "", _md(sc.sort_values("auc_trivial_features", ascending=False)), "",
             "## Quality flags (kept, not removed)", "",
             _md(flag_tab.rename_axis("flag")) if len(flag_tab) else "No image flagged.", "",
             "## Sample images (after fundus crop)", "", "![samples](samples.png)"]
    (out / "eda_report.md").write_text("\n".join(lines) + "\n")
    print(f"source identifiability bacc={src_bacc:.3f} (chance {chance:.3f})")
    print(sc.sort_values("auc_trivial_features", ascending=False).to_string(index=False))
    print(f"flagged {len(flagged)} images: {flag_tab.to_dict()}")


if __name__ == "__main__":
    main()
