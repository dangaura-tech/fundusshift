"""Supplement tables for the external cohorts (S10) and the recipe ablation (S9), from saved results only.
python paper/latex/make_supp_review.py   (run from the repository root)
"""
import glob
from pathlib import Path

import pandas as pd

OUT = Path("paper/latex/generated/supp")
f3 = lambda v: "--" if pd.isna(v) else (f"{v:.3f}"[1:] if 0 <= v < 1 else (f"$-${-v:.3f}"[0:] if v < 0 else f"{v:.2f}" if v < 10 else f"{v:.0f}"))
f3s = lambda v: "--" if pd.isna(v) else ("$-$" + f"{-v:.3f}"[1:] if v < 0 else ("+" + f"{v:.3f}"[1:] if v > 0 else ".000"))


def fx(v):
    if pd.isna(v):
        return "--"
    if v >= 1:
        return f"{v:.2f}" if v < 10 else f"{v:.0f}"
    return f"{v:.3f}"[1:]


e = pd.read_csv("results/external/external_runs.csv")
b = pd.read_csv("results/review_baselines/external.csv")
NAME = {"LOTUS": "LOTUS", "initial_design": "Initial design", "zero_shot": "MedSigLIP zero-shot", "convnext_ce": "ConvNeXt-T, CE", "convnext_groupdro": "ConvNeXt-T, Group DRO"}
m = e[e.set == "messidor2"].groupby("model").mean(numeric_only=True)
g = e[e.set == "g1020"].groupby("model").mean(numeric_only=True)
txt = ""
for k, n in NAME.items():
    a, c = m.loc[k], g.loc[k]
    txt += " & ".join([n] + [fx(a[x]) for x in ["bacc", "recall_D", "pred_rate_G", "auroc_pos_vs_N", "auroc_referable", "ece_ts", "lac90"]] + [fx(c[x]) for x in ["bacc", "recall_G", "pred_rate_D", "auroc_pos_vs_N"]]) + "\\\\\n"
bb = b.groupby("variant").mean(numeric_only=True)
for k, n in [("lp", "Linear probe$^*$"), ("lp_text", "Linear probe + text$^*$")]:
    a = bb.loc[k]
    txt += " & ".join([n] + [fx(a[x]) for x in ["messidor2_bacc", "messidor2_recall_pos", "messidor2_pred_G", "messidor2_auroc", "messidor2_auroc_referable", "messidor2_ece"]] + ["--"]
                      + [fx(a[x]) for x in ["g1020_bacc", "g1020_recall_pos"]] + ["--", fx(a["g1020_auroc"])]) + "\\\\\n"
(OUT / "s10_ext.tex").write_text(txt + "\\bottomrule\n")

txt = ""
for cohort, d in [("Messidor-2", m), ("G1020", g)]:
    first = True
    for k, n in NAME.items():
        a = d.loc[k]
        grades = [a.get("recallD_grade1"), a.get("recallD_grade2"), a.get("recallD_grade3"), a.get("recallD_grade4")] if cohort == "Messidor-2" else [float("nan")] * 4
        cells = [fx(a["recall_N"])] + ["" if pd.isna(v) else fx(v) for v in grades] + [fx(a[x]) for x in ["cov80", "risk80", "nll_ts", "spec@calB90", "sens@calB90", "spec@calB95", "sens@calB95"]]
        txt += (cohort if first else "") + " & " + n.replace(", ", " ").replace("Group DRO", "GDRO").replace("MedSigLIP zero-shot", "Zero-shot") + " & " + " & ".join(cells) + "\\\\\n"
        first = False
    if cohort == "Messidor-2":
        txt += "\\midrule\n"
(OUT / "s10_ext2.tex").write_text(txt + "\\bottomrule\n")

d = pd.concat([pd.read_csv(f) for f in glob.glob("results/recipe_ablation/w*.csv")])
d = d[d.split != 0]
p = d.pivot_table(index="split", columns="variant", values="test_bacc")
s = d.groupby("variant")[["test_bacc", "ece_ts", "nll_ts", "recall_G"]].mean()
LAB = {"reference": "Reference", "no_balancing": "No class balancing", "lr_1e-3": "lr $10^{-3}$", "epochs_24": "24 epochs", "last_epoch": "Last epoch (no selection)",
       "wd_0": "Weight decay 0", "no_smoothing": "No smoothing", "wd_1e-1": "Weight decay $10^{-1}$", "epochs_4": "4 epochs"}
ref = s.loc["reference", "test_bacc"]
txt = ""
for k in sorted(LAB, key=lambda k: (k != "reference", s.loc[k, "test_bacc"])):
    a = s.loc[k]
    dl = "--" if k == "reference" else f3s(round(a.test_bacc - ref, 3))
    hi = "--" if k == "reference" else f"{int((p['reference'] > p[k]).sum())}/{len(p)}"
    txt += f"{LAB[k]} & {fx(a.test_bacc)} & {dl} & {fx(a.ece_ts)} & {fx(a.nll_ts)} & {fx(a.recall_G)} & {hi}\\\\\n"
(OUT / "s9_recipe.tex").write_text(txt + "\\bottomrule\n")
print("[ok] s10_ext, s10_ext2, s9_recipe")
