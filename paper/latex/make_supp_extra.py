"""Extra supplement tables: source hierarchy and label scope, image quality, duplicate links, per-source promises.
FS_DATA points at the folder holding bench/ (split table and near_duplicates.csv); run from the repository root.
python paper/latex/make_supp_extra.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = os.environ.get("FS_DATA", "data")
OUT = Path(os.environ.get("FS_OUT", "paper/latex/generated/supp"))
OUT.mkdir(parents=True, exist_ok=True)
SRC = {"afio_composite": "AFIO", "fives": "FIVES", "hrf": "HRF", "hygd": "HYGD", "palm": "PALM", "papila": "PAPILA", "paraguay_dr": "Paraguay", "bangladesh_eye": "Bangladesh Eye", "ddr": "OIA-DDR", "deepdrid": "DeepDRiD", "grape": "GRAPE", "idrid": "IDRiD", "jichi": "Jichi", "sustech_sysu": "SUSTech-SYSU", "chaksu": "Ch\\'ak\\d{s}u", "rfmid2": "RFMiD 2.0"}
DEV = ["bangladesh_eye", "ddr", "deepdrid", "grape", "idrid", "jichi", "sustech_sysu"]
TEST = ["afio_composite", "fives", "hrf", "hygd", "palm", "papila", "paraguay_dr"]
ORDER = DEV + TEST + ["chaksu"]
ROLE = {**{s: "Dev" for s in DEV}, **{s: "Test" for s in TEST}, "chaksu": "OOD"}


def esc(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def f3(x):
    if x is None or pd.isna(x):
        return "--"
    if abs(x) < 1 and round(abs(x), 3) < 1:
        return f"{x:.3f}"[1:] if x >= 0 else f"{x:.3f}".replace("-0.", "$-$.")
    return f"{x:.2f}"


def write(name, rows):
    (OUT / name).write_text("".join(" & ".join(r) + " \\\\\n" for r in rows) + "\\bottomrule\n")


t = pd.read_csv(Path(DATA) / "bench" / "splits_final.csv", dtype=str, keep_default_na=False)
used = t[t.split.isin(["train", "validation", "calibration", "test", "test_ood"])]

# Source hierarchy, patients and label scope
rows = []
for s in ORDER:
    q = used[used.source == s]
    has = q.patient_id != ""
    pats = q.patient_id[has].nunique()
    pat_txt = "n/a" if not pats else (f"{pats:,}" if has.all() else f"{pats:,} ({int(has.sum()):,} img.)")
    classes = "/".join(c for c in "NDG" if (q.label == c).any())
    scope = "no DR (DR-only grading)" if q.normal_scope.iloc[0] == "dr_only" else "no graded disease"
    rows.append([ROLE[s], SRC[s], esc(q.country.iloc[0]), f"{len(q):,}", pat_txt, classes, scope])
write("s1_hierarchy.tex", rows)

# Image quality by source (processed images, used rows)
rows = []
for s in ORDER:
    q = used[used.source == s]
    num = lambda c: pd.to_numeric(q[c], errors="coerce")
    flag = (q.quality_flags != "").mean()
    rows.append([SRC[s], f"{int(num('orig_width').median()):,}$\\times${int(num('orig_height').median()):,}", f3(num("fov_fraction").median()), f"{num('brightness').median():.1f}", f"{num('contrast').median():.1f}", f"{num('sharpness').median():.1f}", f"{100 * flag:.1f}\\%"])
write("s1_quality.tex", rows)

# Duplicate links by source pair
nd = pd.read_csv(Path(DATA) / "bench" / "near_duplicates.csv")
nd = nd[nd.linked]
nd["pair"] = [" -- ".join(sorted([SRC.get(a, a), SRC.get(b, b)])) if a != b else SRC.get(a, a) for a, b in zip(nd.source_a, nd.source_b)]
nd["cross"] = nd.source_a != nd.source_b
g = nd.groupby(["cross", "pair"]).agg(n=("pair", "size"), exact=("exact", "sum"), conflict=("label_a", lambda x: 0)).reset_index()
g["conflict"] = [int(((nd.pair == p) & (nd.label_a != nd.label_b)).sum()) for p in g.pair]
g = g.sort_values(["cross", "n"], ascending=[False, False])
write("s1_duplicates.tex", [["across" if r.cross else "within", r.pair, f"{int(r.n):,}", f"{int(r.exact):,}", f"{int(r.conflict):,}"] for r in g.itertuples()])
print("linked pairs", len(nd), "exact", int(nd.exact.sum()), "across sources", int(nd.cross.sum()))

# Per-source promises of LANTERN on the published split (TS, thresholds from Cal-B)
o = pd.read_csv("results/top_dev/ops.csv")
o = o[(o.config == "last_reg") & (o.setting == "posthoc") & (o.posthoc == "ts")]
rows = []
for s in TEST + ["chaksu"]:
    w = "chaksu" if s == "chaksu" else f"src:{s}"
    g_ = lambda op, c: o[(o["view"] == w) & (o.op == op)][c].mean()
    held = g_("risk10", "risk") <= .10
    rows.append([SRC[s], f3(g_("cov80", "coverage")), f3(g_("cov80", "risk")), f3(g_("cov90", "coverage")), f3(g_("cov90", "risk")), f3(g_("risk10", "coverage")), f3(g_("risk10", "risk")), "yes" if held else "no"])
write("s5_persource_ops.tex", rows)
print("[ok extra]")


# ---------------- LOTUS: revision runs (results/revision) and published-split logits (FS_RUNS/lotus) ----------------
import glob
REV = Path("results/revision")
rv = pd.concat([pd.read_csv(f) for f in glob.glob(str(REV / "rotation" / "w*.csv"))])
rot = rv[rv.split > 0]
ARMNAME = {"no_projection": "\\textbf{LOTUS} (bag + text)", "full": "+ projection (initial design)", "single_text": "single head + text", "bag_only": "bag only",
           "no_bag": "single head + projection + text", "no_text": "bag + projection", "single_head": "single head", "proj_only": "single head + projection", "text_only": "text only"}
per = rot.groupby(["arm", "split"]).mean(numeric_only=True).reset_index()
base = per[per.arm == "no_projection"].set_index("split")
rows = []
order = per.groupby("arm").test_bacc.mean().sort_values(ascending=False).index
for arm in order:
    q = per[per.arm == arm].set_index("split")
    d = base.test_bacc - q.test_bacc
    tail = ["--", "--"] if arm == "no_projection" else [f"{d.mean():+.3f}", f"{int((d > 0).sum())}/50"]
    rows.append([ARMNAME[arm], *[f3(q[c].mean()) for c in ["test_bacc", "test_macro_f1", "test_auroc", "recall_G", "false_G", "ece_ts", "nll_ts", "aurc_ts", "risk_test", "chaksu_bacc"]], *tail])
write("s6_factorial.tex", rows)

lp0 = pd.read_csv("results/supplement/rotation.csv")
lp0 = lp0[(lp0.model == "medsiglip") & (lp0.kind == "full_class") & (lp0.split > 0)].assign(seed=0, arm="lp")
lpx = pd.read_csv("results/supplement/rotation_lp_seeds.csv")
lpx = lpx[lpx.split > 0].assign(arm="lp")
sx = pd.concat([rot, lp0, lpx], ignore_index=True)
fullp = sx[sx.arm == "no_projection"].groupby("split")
rows = []
for arm, n in [("no_projection", "\\textbf{LOTUS}"), ("full", "+ projection (initial design)"), ("single_text", "single head + text"), ("bag_only", "bag only"), ("text_only", "text only"), ("lp", "MedSigLIP linear probe")]:
    q = sx[sx.arm == arm]
    ps = q.groupby("split")
    m_, e_ = ps.test_bacc.mean(), ps.ece_ts.mean()
    d, de = fullp.test_bacc.mean() - m_, fullp.ece_ts.mean() - e_
    tail = ["--", "--", "--"] if arm == "no_projection" else [f"{d.mean():+.3f} $\\pm$ {d.std():.3f}", f"{int((d > 0).sum())}/50", f"{int((de < 0).sum())}/50"]
    rows.append([n, str(q.split.nunique()), f3(m_.mean()) + " $\\pm$ " + f3(m_.std()), f3(ps.test_bacc.std().mean()), f3(e_.mean()), f3(ps.recall_G.mean().mean()), *tail])
write("s6_seeds.tex", rows)

G_SRC = set(used[used.label == "G"].source) - {"chaksu"}
rows = []
for name, d in [("LOTUS", rot[rot.arm == "no_projection"]), ("Initial design", rot[rot.arm == "full"]), ("MedSigLIP LP", pd.concat([lp0, lpx]))]:
    d = d.groupby("split").agg(test_sources=("test_sources", "first"), recall_G=("recall_G", "mean"), test_bacc=("test_bacc", "mean")).reset_index()
    d["g_dev"] = d.test_sources.map(lambda s: len(G_SRC - set(s.split("|"))))
    for k, q in d.groupby("g_dev"):
        rows.append([name, str(k), str(len(q)), f3(q.recall_G.mean()), f3(q.test_bacc.mean())])
write("s6_gsrc.tex", rows)

rg = pd.read_csv(REV / "region.csv")
RN = {"full": "Initial design", "no_projection": "\\textbf{LOTUS}", "medsiglip_lp": "MedSigLIP LP"}
rows = []
for R, q in rg.groupby("region"):
    for arm in ["no_projection", "full", "medsiglip_lp"]:
        qq = q[q.arm == arm]
        rows.append([R if arm == "no_projection" else "", ", ".join(SRC[s] for s in qq.test_sources.iloc[0].split("|")) if arm == "no_projection" else "", f"{int(qq.n_test.iloc[0]):,}" if arm == "no_projection" else "", RN[arm], f3(qq.test_bacc.mean()), f3(qq.test_bacc_source_equal.mean()), f3(qq.recall_G.mean()), f3(qq.ece_ts.mean()), f3(qq.chaksu_bacc.mean())])
mean_rows = [["Mean", "", "", RN[arm], f3(rg[rg.arm == arm].groupby("region").test_bacc.mean().mean()), f3(rg[rg.arm == arm].groupby("region").test_bacc_source_equal.mean().mean()), "--", f3(rg[rg.arm == arm].groupby("region").ece_ts.mean().mean()), "--"] for arm in ["no_projection", "full", "medsiglip_lp"]]
write("s6_region.tex", rows + [["\\midrule " + mean_rows[0][0]] + mean_rows[0][1:]] + mean_rows[1:])

cB = pd.concat([pd.read_csv(f) for f in glob.glob(str(REV / "certify_no_projection" / "rot2_w*.csv"))])
cB = cB[cB.split > 0]
rows = []
for arm, nm in [("calB", "Cal-B"), ("loso", "LOSO reference")]:
    for cert in ["", "_cert"]:
        for tgt in ["c80", "c90", "r10"]:
            p = f"{arm}_{tgt}{cert}"
            rows.append([nm + (" + cert." if cert else ""), {"c80": "cov. .80", "c90": "cov. .90", "r10": "risk .10"}[tgt], f3(cB[f"{p}_test_cov"].mean()), f3(cB[f"{p}_test_risk"].mean()),
                         f"{int((cB[f'{p}_test_risk'] <= .10).sum())}/50" if tgt == "r10" else "--", f3(cB[f"{p}_test_fr"].mean()), f3(cB[f"{p}_test_GtoN"].mean()), f3(cB[f"{p}_chaksu_cov"].mean()), f3(cB[f"{p}_chaksu_risk"].mean())])
write("s8_cert.tex", rows)

# per-source ECE after TS with bootstrap over leakage groups, LOTUS seeds pooled (as in experiments.review_checks)
RUNS = os.environ.get("FS_RUNS")
if RUNS:
    import sys
    sys.path.insert(0, ".")
    from experiments.calib import Calibrator, ece as ece_
    from experiments.data import load_table
    tt, _ = load_table(Path(DATA) / "bench")
    sp, yy, ss, grp = tt.split.to_numpy(), tt.y.to_numpy(), tt.source.to_numpy(), tt.group_id.to_numpy()
    ca, te = (sp == "calibration") & (tt.cal_subset.to_numpy() == "cal_A"), sp == "test"
    Ps = []
    for s in range(3):
        Z = np.load(Path(RUNS) / "lotus" / f"logits_s{s}.npy").astype(np.float64)
        Ps.append(Calibrator("ts").fit(Z[ca], yy[ca])(Z))
    rng = np.random.default_rng(0)
    rows = []
    for s in ["paraguay_dr", "hygd", "fives", "papila", "palm", "hrf", "afio_composite", "chaksu"]:
        m = np.flatnonzero((ss == s) & (te | (sp == "test_ood")))
        ug, inv = np.unique(grp[m], return_inverse=True)
        point = np.mean([ece_(p[m], yy[m])[0] for p in Ps])
        bs = []
        for _ in range(1000):
            w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv]
            idx = np.repeat(m, w)
            bs.append(np.mean([ece_(p[idx], yy[idx])[0] for p in Ps]))
        rows.append([SRC[s], f"{len(m):,}", f"{len(ug):,}", f3(point), f"[{f3(np.quantile(bs, .025))}, {f3(np.quantile(bs, .975))}]"])
    write("s4_ececi.tex", rows)

# LOTUS per-source promises on the published split
oo = pd.read_csv("results/lotus/ops.csv")
oo = oo[(oo.config == "lotus") & (oo.setting == "posthoc") & (oo.posthoc == "ts")]
rows = []
for s in TEST + ["chaksu"]:
    w = "chaksu" if s == "chaksu" else f"src:{s}"
    g_ = lambda op, c: oo[(oo["view"] == w) & (oo.op == op)][c].mean()
    rows.append([SRC[s], f3(g_("cov80", "coverage")), f3(g_("cov80", "risk")), f3(g_("cov90", "coverage")), f3(g_("cov90", "risk")), f3(g_("risk10", "coverage")), f3(g_("risk10", "risk")), "yes" if g_("risk10", "risk") <= .10 else "no"])
write("s5_persource_ops.tex", rows)
print("[ok lotus extra]")
