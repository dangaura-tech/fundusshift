"""Builds every table of the supplementary material from saved results, so no number is typed by hand.
FS_DATA points at the folder holding bench/ (split table); run from the repository root after paper/make_assets.py.
python paper/make_supp.py
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from experiments.backbones import VLM, VM
from experiments.data import load_table
from experiments.prompts import CUPL, NAMES, SINGLE, TEMPLATES

DATA = os.environ.get("FS_DATA", "data")
OUT = Path("paper/generated/supp")
OUT.mkdir(parents=True, exist_ok=True)
t, classes = load_table(f"{DATA}/bench")
SRC = {"afio_composite": "AFIO", "fives": "FIVES", "hrf": "HRF", "hygd": "HYGD", "palm": "PALM", "papila": "PAPILA", "paraguay_dr": "Paraguay", "bangladesh_eye": "Bangladesh Eye", "ddr": "OIA-DDR", "deepdrid": "DeepDRiD", "grape": "GRAPE", "idrid": "IDRiD", "jichi": "Jichi", "sustech_sysu": "SUSTech-SYSU", "chaksu": "Ch\\'ak\\d{s}u"}
SET = {"16-shot-probe": "16-shot probe", "linear-probe": "Linear probe", "mlp-probe": "MLP probe", "zero-shot": "Zero-shot", "finetune": "Fine-tune", "finetune-nll": "Fine-tune, NLL rule", "erm": "ERM", "groupdro": "Group DRO", "coral": "CORAL", "swad": "SWAD", "wise-ft": "WiSE-FT", "coop": "CoOp", "coop-ctxinit": "CoOp ctx-init", "coop-lp": "CoOp + LP", "tip-adapter": "Tip-Adapter", "tip-adapter-f": "Tip-Adapter-F", "tip-adapter-acc": "Tip-Adapter (acc. rule)", "tip-adapter-f-acc": "Tip-Adapter-F (acc. rule)", "mcq": "Multiple choice", "proposed": "LANTERN"}
TEST = ["paraguay_dr", "hygd", "fives", "papila", "palm", "hrf", "afio_composite"]


def f3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return (f"{x:.3f}"[1:] if 0 <= x < 1 else f"{x:.3f}".replace("-0.", "$-$.")) if abs(x) < 1 else f"{x:.2f}"


def esc(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def write(name, rows):
    (OUT / name).write_text("".join(rows) + "\\bottomrule\n")


def line(*cells):
    return " & ".join(cells) + " \\\\\n"


v = pd.read_csv("results/full_eval/views.csv.gz", low_memory=False)
v["config"] = v.config.fillna("")
v = v[~v.setting.str.endswith("+ens")]
L = pd.read_csv("results/top_dev/views.csv", low_memory=False)
L = L[(L.setting == "posthoc") & (L.config == "last_reg")].assign(setting="proposed", model="medsiglip_tokens")
reps = pd.read_csv("results/paired_tests/representatives.csv")
reps["config"] = reps.config.fillna("")


def mean(q, view, ph, col):
    return q[(q["view"] == view) & (q.posthoc == ph)][col].mean()


# S1: sources, splits and classes
lic = {"bangladesh_eye": "CC BY 4.0", "ddr": "CC BY-NC-SA 4.0", "deepdrid": "CC BY-SA 4.0", "grape": "CC0", "idrid": "CC BY 4.0", "jichi": "CC BY 4.0", "sustech_sysu": "CC BY 4.0", "afio_composite": "CC BY 4.0", "fives": "CC BY 4.0", "hrf": "CC BY 4.0", "hygd": "ODC-By 1.0", "palm": "CC0", "papila": "GPL-3.0", "paraguay_dr": "CC BY 4.0", "chaksu": "CC BY 4.0"}
rows = []
order = ["bangladesh_eye", "ddr", "deepdrid", "grape", "idrid", "jichi", "sustech_sysu"] + sorted(TEST, key=lambda s: SRC[s]) + ["chaksu"]
for s in order:
    q = t[(t.source == s) & (t.split != "test_openset")]
    cnt = lambda sp: "/".join(str(int(((q.split == sp) & (q.y == c)).sum())) for c in range(3))
    sps = ["train", "validation", "calibration"] if s in order[:7] else (["test"] if s != "chaksu" else ["test_ood"])
    rows.append(line(SRC[s], esc(q.country.iloc[0])[:28], "DR only" if q.normal_scope.iloc[0] == "dr_only" else "full", lic[s], *[cnt(sp) if sp in sps else "--" for sp in ["train", "validation", "calibration"]], cnt("test") if s != "chaksu" else cnt("test_ood")))
write("s1_sources.tex", rows)

rows = []
for s in order:
    for c, cn in enumerate(classes):
        q = t[(t.source == s) & (t.y == c)]
        if len(q):
            labs = sorted(set(str(x).split("|")[0] for x in q.native_label.unique()))
            rows.append(line(SRC[s], cn, esc(", ".join(labs[:6]) + (", ..." if len(labs) > 6 else ""))))
write("s1_labels.tex", rows)

# S2: encoders and prompts
rows = []
for k, (_, data, ft) in VM.items():
    rows.append(line(esc(k), esc(data), "yes" if ft else "no"))
for k, (_, _, data) in VLM.items():
    rows.append(line(esc(k), esc(data), "no"))
rows.append(line("medsiglip", "Medical image-text (MedGemma report); fundus: EyePACS", "no"))
rows.append(line("flair", "Retinal image-text; includes five FundusShift test sources", "no"))
write("s2_encoders.tex", rows)
pr = [line("CLIP ensemble", esc("; ".join(TEMPLATES)) + " with class names: " + esc(", ".join(f"{k}: {v_}" for k, v_ in NAMES.items())))]
for c in classes:
    pr.append(line(f"CuPL ({c})", esc(" ".join(CUPL[c]))))
for c in classes:
    pr.append(line(f"Single ({c})", esc(SINGLE[c])))
write("s2_prompts.tex", pr)

# S3: every setting and encoder, best configuration by validation
rows = []
val = v[(v["view"] == "validation") & (v.posthoc == "none")].groupby(["setting", "model", "config"]).bacc.mean().reset_index()
best = val.loc[val.groupby(["setting", "model"]).bacc.idxmax()]
for r in best.sort_values(["setting", "bacc"], ascending=[True, False]).itertuples():
    q = v[(v.setting == r.setting) & (v.model == r.model) & (v.config == r.config)]
    rows.append(line(esc(SET.get(r.setting, r.setting)), esc(r.model) + ("$^\\dagger$" if r.model == "flair" else ""), esc(r.config) or "--", f3(r.bacc), f3(mean(q, "test", "none", "bacc")), f3(mean(q, "test", "none", "eq_bacc")), f3(mean(q, "chaksu", "none", "bacc")),
                     "/".join(f3(mean(q, "test", p, "ece")) for p in ["none", "ts", "fts"]), "/".join(f3(mean(q, "test", p, "nll")) for p in ["none", "ts", "fts"])))
q = v[v.setting == "mcq"]
for m_ in sorted(q.model.unique()):
    qq = q[q.model == m_]
    rows.append(line("Multiple choice", esc(m_), "avg6", "--", f3(mean(qq, "test", "none", "bacc")), f3(mean(qq, "test", "none", "eq_bacc")), f3(mean(qq, "chaksu", "none", "bacc")), "--", "--"))
write("s3_all.tex", rows)


def rep_frames():
    out = [("LANTERN", L)]
    for r in reps.itertuples():
        if r.setting.endswith("-acc"):
            continue
        out.append((f"{SET[r.setting]}, {esc(r.model)}", v[(v.setting == r.setting) & (v.model == r.model) & (v.config == r.config)]))
    return out


# S4: per source and per class
rows, rec = [], []
for n, q in rep_frames():
    rows.append(line(n, *[f3(mean(q, f"src:{s}", "none", "bacc")) for s in TEST], f3(mean(q, "chaksu", "none", "bacc")), f3(mean(q, "test", "none", "eq_bacc")), f3(np.nanmin([mean(q, f"src:{s}", "none", "bacc") for s in TEST]))))
    rec.append(line(n, *[f3(mean(q, f"src:{s}", "none", f"recall_{c}")) for s, c in [("fives", 0), ("fives", 1), ("fives", 2), ("afio_composite", 0), ("afio_composite", 1), ("afio_composite", 2), ("papila", 0), ("papila", 2), ("paraguay_dr", 0), ("paraguay_dr", 1)]]))
write("s4_sources.tex", rows)
write("s4_recall.tex", rec)
rows = []
for n, q in rep_frames():
    rows.append(line(n, *[f3(mean(q, f"src:{s}", "ts", "ece")) for s in TEST], f3(mean(q, "chaksu", "ts", "ece")), f3(mean(q, "test", "ts", "eq_ece"))))
write("s4_ece.tex", rows)

# S5: reliability of representatives in all three states
o = pd.read_csv("results/full_eval/ops.csv" if Path("results/full_eval/ops.csv").exists() else "results/full_eval/ops.csv.gz", low_memory=False)
o["config"] = o.config.fillna("")
oL = pd.read_csv("results/top_dev/ops.csv")
oL = oL[(oL.config == "last_reg") & (oL.setting == "posthoc")]
c_ = pd.read_csv("results/full_eval/conformal.csv" if Path("results/full_eval/conformal.csv").exists() else "results/full_eval/conformal.csv.gz", low_memory=False)
c_["config"] = c_.config.fillna("")
cL = pd.read_csv("results/top_dev/conformal.csv")
cL = cL[(cL.config == "last_reg") & (cL.setting == "posthoc")]
rows, conf = [], []
for r in [None] + [x for x in reps.itertuples() if not x.setting.endswith("-acc")]:
    if r is None:
        oo, cc, n = oL, cL, "LANTERN"
    else:
        oo = o[(o.setting == r.setting) & (o.model == r.model) & (o.config == r.config)]
        cc = c_[(c_.setting == r.setting) & (c_.model == r.model) & (c_.config == r.config)]
        n = f"{SET[r.setting]}, {esc(r.model)}"
    for ph, pn in [("none", "R"), ("ts", "T"), ("fts", "F")]:
        g = lambda op, w, col: oo[(oo.op == op) & (oo["view"] == w) & (oo.posthoc == ph)][col].mean()
        rows.append(line(n if ph == "none" else "", pn, *[f3(g(op, "test", col)) for op, col in [("cov80", "coverage"), ("cov80", "risk"), ("cov80", "G_to_N_rate"), ("cov80", "D_to_N_rate"), ("cov90", "coverage"), ("cov90", "risk"), ("risk10", "coverage"), ("risk10", "risk")]], f3(g("cov80", "chaksu", "coverage")), f3(g("cov80", "chaksu", "risk"))))
        h = lambda me, a_, w, col: cc[(cc.method == me) & (cc.alpha == a_) & (cc["view"] == w) & (cc.posthoc == ph)][col].mean()
        conf.append(line(n if ph == "none" else "", pn, *[f3(h(me, .1, "test", "coverage")) + "/" + f"{h(me, .1, 'test', 'size'):.2f}" for me in ["LAC", "APS", "ClassLAC"]], *[f3(h(me, .05, "test", "coverage")) for me in ["LAC", "APS", "ClassLAC"]], f3(h("LAC", .1, "chaksu", "coverage")) + "/" + f"{h('LAC', .1, 'chaksu', 'size'):.2f}"))
write("s5_ops.tex", rows)
write("s5_conformal.tex", conf)

hv = pd.read_csv("results/reliability/halving_summary.csv")
hv["config"] = hv.config.fillna("")
rows = []
for r in [x for x in reps.itertuples() if not x.setting.endswith("-acc")]:
    q = hv[(hv.setting == r.setting) & (hv.model == r.model) & (hv.config == r.config) & (hv.posthoc == "ts")]
    if not len(q):
        continue
    g = lambda op, w, col: q[(q.op == op) & (q["view"] == w)][col].mean()
    rows.append(line(f"{SET[r.setting]}, {esc(r.model)}", *[f3(g("cov80", w, "coverage")) for w in ["other_half", "test", "chaksu"]], *[f3(g("risk10", w, "risk")) for w in ["other_half", "test", "chaksu"]], *[f3(g("lac90", w, "coverage")) for w in ["other_half", "test", "chaksu"]]))
write("s5_halving.tex", rows)

b = pd.read_csv("results/full_eval/budget.csv")
rows = []
for (s, m_), q in b.groupby(["setting", "model"]):
    full = q[q.policy == "full"]
    r_ = [f3(full["T"].mean()) if len(full) else "--", f3(full.T_oracle.mean()) if len(full) else "--"]
    for size in [50, 200, 1000]:
        qq = q[(q.policy == "natural") & (q["size"] == size)]
        r_.append(f3(qq.test_nll.mean()) if len(qq) else "--")
    r_.append(f3(full.test_nll.mean()) if len(full) else "--")
    rows.append(line(esc(SET.get(s, s)), esc(m_), *r_))
write("s5_budget.tex", rows)

bl = pd.read_csv("results/supplement/baseline.csv")
write("s5_trivial.tex", [line("Test" if r.view == "test" else "Ch\\'ak\\d{s}u", esc(r.predictor), f3(r.acc), f3(r.bacc), f3(r.bacc_source_equal), f3(r.ece), f3(r.nll)) for r in bl.itertuples()])
vr = pd.read_csv("results/followup/val_test_rank.csv")
write("s5_rank.tex", [line(esc(SET.get(r.setting, r.setting)) if r.setting != "ALL" else "All configurations", str(r.n), f3(r.spearman)) for r in vr.itertuples()])

# S6: rotations, discrimination, ceiling, sources, curve
rot = pd.read_csv("results/supplement/rotation.csv")
rot = rot[rot.split > 0]
lr = pd.read_csv("results/lantern_rotation.csv")
lr = lr[lr.split > 0]
rows = []
for n, q in [("LANTERN", lr)] + [(f"{esc(m_)}, {'16-shot' if k == '16_shot' else 'full'}", rot[(rot.model == m_) & (rot.kind == k)]) for m_ in ["medsiglip", "dinov2_b14", "siglip2_l16", "convnext_t"] for k in ["full_class", "16_shot"]]:
    rows.append(line(n, f3(q.val_bacc.mean()), f3(q.test_bacc.mean()), f3(q.test_bacc.quantile(.05)) + "--" + f3(q.test_bacc.quantile(.95)), f3(q.test_bacc_source_equal.mean()), f3(q.recall_G.mean()), f3(q.ece_raw.mean()), f3(q.ece_ts.mean()), f3(q.risk_internal.mean()) + "$\\to$" + f3(q.risk_test.mean()), f3(q.lac_internal.mean()) + "$\\to$" + f3(q.lac_test.mean()), f3(q.cov_internal.mean()) + "$\\to$" + f3(q.cov_test.mean())))
write("s6_rotation.tex", rows)
ab = pd.concat([pd.read_csv(f_) for f_ in glob.glob("results/ablation_rotation/w*.csv")])
ab = ab[ab.split > 0]
rows = []
for k, n in [("full", "LANTERN"), ("no_projection", "without projection"), ("no_bag", "without bag"), ("no_text", "without text"), ("single_head", "single head only"), ("text_only", "text only")]:
    q = ab[ab.arm == k]
    rows.append(line(n, f3(q.test_bacc.mean()), f3(q.test_bacc.std()), f3(q.test_bacc_source_equal.mean()), f3(q.recall_N.mean()), f3(q.recall_D.mean()), f3(q.recall_G.mean()), f3(q.false_G.mean()), f3(q.ece_ts.mean()), f3(q.chaksu_bacc.mean())))
write("s6_ablation.tex", rows)
db = pd.read_csv("results/supplement/discrim_boot.csv")
write("s6_discrim.tex", [line(esc(r.pair).replace("\\_", "--"), esc(SET.get(r.a, r.a)), esc(SET.get(r.b, r.b)), str(r.n_sources), f3(r.auroc_a), f3(r.auroc_b), f"{r.diff:+.3f}", f"[{r.lo:+.3f}, {r.hi:+.3f}]") for r in db.itertuples()])
ce_ = pd.read_csv("results/supplement/ceiling.csv")
cb = pd.read_csv("results/supplement/ceiling_boot.csv")
rows = []
for s in ["afio_composite", "chaksu", "fives", "hrf", "papila", "paraguay_dr"]:
    q = ce_[ce_.source == s]
    g = lambda k, col: q[q.kind == k][col].mean()
    gg = cb[cb.source == s]
    rows.append(line(SRC[s], f3(g("dev_raw", "bacc")), f3(g("within", "bacc")), f3(g("dev_ts", "ece")), f3(g("within", "ece")), f3(g("dev_raw", "auroc_N_G")), f3(g("within", "auroc_N_G")), f3(g("dev_raw", "auroc_N_D")), f3(g("within", "auroc_N_D")), f"{gg.gap.mean():+.3f} [{gg.lo.mean():+.3f}, {gg.hi.mean():+.3f}]" if len(gg) else "--"))
write("s6_ceiling.tex", rows)
sd = pd.read_csv("results/per_source/source_diversity_3class_N300.csv", skiprows=3, header=None)
rows = []
for r in sd.itertuples(index=False):
    if str(r[0]) in ("nan", "model"):
        continue
    vals = [float(x) for x in r[1:9]]
    rows.append(line(esc(r[0]) + ("$^\\dagger$" if r[0] == "flair" else ""), *[f3(x) for x in vals[4:]], *[f3(x) for x in vals[:4]]))
write("s6_sources.tex", rows)
cv = pd.read_csv("results/followup/curve_source_boot.csv")
write("s6_curve.tex", [line(esc(r.model), str(r.k), f"{r.diff:+.3f}", f"[{r.lo:+.3f}, {r.hi:+.3f}]") for r in cv.itertuples()])

# S7: attempted fixes and other probes
tt = pd.read_csv("results/supplement/tta.csv")
rows = []
for (s, m_), q in tt.groupby(["setting", "model"], sort=False):
    g = lambda me, col, w="test": q[(q.method == me) & (q["view"] == w)][col].mean()
    rows.append(line(esc(SET.get(s, s)), esc(m_), *[f3(g(me, "bacc")) for me in ["none", "bn", "tent", "t3a_20", "t3a_100"]], *[f3(g(me, "bacc", "test_ood")) for me in ["none", "tent", "t3a_100"]]))
write("s7_tta.tex", rows)
pr_ = pd.read_csv("results/followup/prior.csv")
pr_["config"] = pr_.config.fillna("")
rows = []
for (s, m_), q in pr_.groupby(["setting", "model"], sort=False):
    g = lambda arm, col: q[q.arm == arm][col].mean()
    rows.append(line(esc(SET.get(s, s)), esc(m_), *[f3(g(a_, "bacc")) for a_ in ["ts", "em_pooled", "em_per_source", "true_pooled", "true_per_source"]], f3(g("ts", "ece")), f3(g("em_per_source", "ece")), f3(g("true_per_source", "ece"))))
write("s7_prior.tex", rows)
tg = pd.read_csv("results/followup/target.csv.gz")
tg = tg[tg.k > 0]
rows = []
for (s_, m_), q in tg.groupby(["setting", "model"], sort=False):
    for i, k in enumerate(sorted(q.k.unique())):
        qq = q[q.k == k]
        g = lambda arm, col: qq[qq.arm == arm][col].mean()
        low = qq[qq.arm == "q_thr_from_K"].lac_cov.quantile(.05)
        rows.append(line(esc(SET.get(s_, s_)) if i == 0 else "", esc(m_) if i == 0 else "", str(int(k)), f3(g("frozen", "ece")), f3(g("T_from_K", "ece")), f3(g("frozen", "lac_cov")), f3(g("q_thr_from_K", "lac_cov")), f3(low)))
write("s7_target.tex", rows)
print("target arms", sorted(tg.arm.unique()))
pp = pd.read_csv("results/supplement/preprocess.csv")
rows = []
for (s, cf), q in pp.groupby(["setting", "config"], sort=False):
    for p_ in ["none", "clahe"]:
        qq = q[q.prep == p_]
        rows.append(line(esc(SET.get(s, s)), esc(cf), "CLAHE" if p_ == "clahe" else "none", f3(qq.val_bacc.mean()), f3(qq.test_bacc.mean()), f3(qq.test_bacc_source_equal.mean()), f3(qq.chaksu_bacc.mean()), f3(qq.ece_ts.mean()), f3(qq.recall_G.mean())))
write("s7_clahe.tex", rows)
gt = pd.read_csv("results/followup/gate.csv")
rows = []
for (s, m_, sc_), q in gt[gt.source == "all"].groupby(["setting", "model", "score"], sort=False):
    rows.append(line(esc(SET.get(s, s)), esc(m_), sc_, f3(q.coverage.mean()), f3(q.risk.mean()), f3(q.calB_coverage.mean()), f3(q.open_accepted.mean()), f3(q.G_to_N_rate.mean())))
write("s7_gate.tex", rows)
fo = pd.read_csv("results/reliability/feature_ood.csv")
fo = fo[fo.setting != "swad"]
rows = []
for (s, m_), q in fo.groupby(["setting", "model"], sort=False):
    rows.append(line(esc(SET.get(s, s)), esc(m_), *[f3(q[q.score == sc_].auroc.mean()) for sc_ in ["MSP", "MaxLogit", "Energy", "KNN", "Mahalanobis"]]))
write("s7_ood.tex", rows)
mq = v[v.setting == "mcq"]
mo = pd.read_csv("results/reliability/mcq_openset_scores.csv")
rows = []
for m_ in sorted(mq.model.unique()):
    q = mq[mq.model == m_]
    rows.append(line(esc(m_), f3(mean(q, "test", "none", "bacc")), *[f3(mean(q, "test", "none", f"recall_{c}")) for c in range(3)], f3(mean(q, "chaksu", "none", "bacc")), f3(mean(q, "chaksu", "none", "recall_2"))))
write("s7_mcq.tex", rows)
dg = pd.read_csv("results/followup/degrade_pred.csv")
rows = []
for (s, m_), q in dg.groupby(["setting", "model"], sort=False):
    cells = []
    for kind in ["clean", "blur", "jpeg"]:
        qq = q[(q.kind == kind) & (q.true == "G") & (q.pred == "G")]
        lv = qq.level.max() if len(qq) else None
        cells.append(f3(qq[qq.level == lv].rate.mean()) if len(qq) else "--")
    rows.append(line(esc(SET.get(s, s)), esc(m_), *cells))
write("s7_degrade.tex", rows)
dr = pd.read_csv("results/followup/dr_grade.csv")
rows = []
for (s, m_), q in dr.groupby(["setting", "model"], sort=False):
    rows.append(line(esc(SET.get(s, s)), esc(m_), *[f3(q[q.grade == g_].recall_D.mean()) if (q.grade == g_).any() else "--" for g_ in sorted(dr.grade.unique())]))
write("s7_grade.tex", rows)
print("grades", sorted(dr.grade.unique()))

# S8: LANTERN development
g7 = pd.read_csv("results/strata/gate7.csv")
rows = []
for arm, q in g7.groupby("arm"):
    rows.append((q.bacc.mean(), line(esc(arm).replace("\\_", " "), f3(q.bacc.mean()), f3(q.ece.mean()), f3(q.nll.mean()), f3(q.recall_G.mean()))))
write("s8_gate7.tex", [r for _, r in sorted(rows, reverse=True)])
g12 = pd.read_csv("results/strata/gate12.csv")
g10 = pd.read_csv("results/strata/gate10.csv")
rows = []
for d_, nm in [(g10, "input"), (g12, "recipe")]:
    for arm, q in d_.groupby("arm"):
        if arm.endswith("bag_nap+text") or arm.endswith("single+text"):
            rows.append((q.bacc.mean(), line(nm, esc(arm).replace("\\_", " "), f3(q.bacc.mean()), f3(q.ece.mean()), f3(q.nll.mean()), f3(q.recall_G.mean()), f3(q.false_G.mean()))))
write("s8_inputs.tex", [r for _, r in sorted(rows, reverse=True)])
q = g12[g12.arm == "last_reg_bag_nap+text"]
write("s8_dev_sites.tex", [line(SRC[s], f3(q[q.source == s].bacc.mean()), f3(q[q.source == s].recall_G.mean()) if q[q.source == s].recall_G.notna().any() else "--", f3(q[q.source == s].false_G.mean()) if q[q.source == s].false_G.notna().any() else "--") for s in sorted(q.source.unique(), key=lambda x: SRC[x])])
lu = pd.read_csv("results/strata/last_upgrade_dev/dev.csv")
rows = []
for arm, q in lu.groupby("arm"):
    rows.append((q.bacc.mean(), line(esc(arm).replace("\\_", " "), f3(q.bacc.mean()), f3(q.nll.mean()), f3(q.recall_G.mean()), f3(q.false_G.mean()), f3(q.N_G_left.mean()), f3(q.N_G_right.mean()))))
write("s8_upgrade.tex", [r for _, r in sorted(rows, reverse=True)])
pp2 = pd.read_csv("results/strata/last_postprocess.csv")
rows = []
for ph in ["none", "ts", "fts"]:
    q = pp2[pp2.posthoc == ph]
    rows.append(line({"none": "Raw", "ts": "TS", "fts": "FTS"}[ph], f3(q.ece.mean()), f3(q.nll.mean()), f3(q.cov80_cov.mean()), f3(q.cov80_risk.mean()), f3(q.risk10_risk.mean()), f3(q.LAC_cov.mean()), f"{q.LAC_size.mean():.2f}", f3(q.ClassLAC_cov.mean()), f3(q.APS_cov.mean()) + "/" + f"{q.APS_size.mean():.2f}"))
write("s8_post.tex", rows)
sc = pd.read_csv("results/scope/dev.csv")
rows = [line(arm, f3(q.bacc.mean()), f3(q.bacc_scope.mean()), f3(q.ece.mean()), f3(q.recall_G.mean()), f3(q.NtoG_dronly.mean()), f3(q.NtoG_full.mean())) for arm, q in sc.groupby("arm")]
se, ss = pd.read_csv("results/scope/rotation_exact.csv"), pd.read_csv("results/scope/rotation_scope.csv")
for n, q in [("exact (rotations)", se[se.split > 0]), ("scope (rotations)", ss[ss.split > 0])]:
    rows.append(line(n, f3(q.test_bacc.mean()), f3(q.test_bacc_scope.mean()), f3(q.ece_ts.mean()), f3(q.recall_G.mean()), "--", f3(q.NtoG_full.mean())))
write("s8_scope.tex", rows)
c1 = pd.concat([pd.read_csv(f_) for f_ in glob.glob("results/certify/rot_w*.csv")])
c2 = pd.concat([pd.read_csv(f_) for f_ in glob.glob("results/certify/rot2_w*.csv")])
c1, c2 = c1[c1.split > 0], c2[c2.split > 0]
rows = []
for arm, nm in [("calB", "Cal-B"), ("loso", "LOSO reference")]:
    for cert in ["", "_cert"]:
        for tgt, d_ in [("c80", c1), ("c90", c1), ("r10", c2)]:
            p = f"{arm}_{tgt}{cert}"
            rows.append(line(nm + (" + cert." if cert else ""), {"c80": "cov. .80", "c90": "cov. .90", "r10": "risk .10"}[tgt], f3(d_[f"{p}_test_cov"].mean()), f3(d_[f"{p}_test_risk"].mean()), f3(d_[f"{p}_test_fr"].mean()), f3(d_[f"{p}_test_GtoN"].mean()), f3(d_[f"{p}_chaksu_cov"].mean()), f3(d_[f"{p}_chaksu_risk"].mean())))
write("s8_cert.tex", rows)
print("[ok supp]")

# Reviewer checks
pj = pd.read_csv("results/review/projection.csv")
write("s6_projection.tex", [line("none (all)" if r.left_out == "none" else SRC[r.left_out], str(int(r.k)), f3(r.ND_removed), f3(r.NG_removed), f3(r.DG_removed), f3(r.NG_removed_random), f3(r.cos_Gsite_NG) if not pd.isna(r.cos_Gsite_NG) else "--") for r in pj.itertuples()])
gs = pd.read_csv("results/review/glaucoma_sources.csv")
write("s6_gsrc.tex", [line(r.method, str(int(r.g_dev_sources)), str(int(r.rotations)), f3(r.recall_G), f3(r.test_bacc)) for r in gs.itertuples()])
ec = pd.read_csv("results/review/ece_ci.csv")
write("s4_ececi.tex", [line(SRC[r.source], f"{int(r.n):,}", f"{int(r.groups):,}", f3(r.ece_ts), f"[{f3(r.lo)}, {f3(r.hi)}]") for r in ec.itertuples()])
seeds = sorted(glob.glob("results/ablation_rotation/seed*_w*.csv"))
if seeds:
    s0 = pd.concat([pd.read_csv(f_) for f_ in glob.glob("results/ablation_rotation/w*.csv")]).assign(seed=0)
    sx = pd.concat([s0] + [pd.read_csv(f_) for f_ in seeds])
    sx = sx[sx.split > 0]
    lp0 = pd.read_csv("results/supplement/rotation.csv")
    lp0 = lp0[(lp0.model == "medsiglip") & (lp0.kind == "full_class") & (lp0.split > 0)].assign(seed=0, arm="lp")
    lpx = pd.read_csv("results/supplement/rotation_lp_seeds.csv") if Path("results/supplement/rotation_lp_seeds.csv").exists() else lp0.iloc[:0]
    sx = pd.concat([sx, lp0, lpx[lpx.split > 0].assign(arm="lp")], ignore_index=True)
    full = sx[sx.arm == "full"].groupby("split").test_bacc.mean()
    rows = []
    for arm, n in [("full", "LANTERN"), ("no_projection", "without projection"), ("no_bag", "without bag"), ("no_text", "without text"), ("text_only", "text only"), ("lp", "MedSigLIP linear probe")]:
        q = sx[sx.arm == arm]
        complete = q.groupby("split").seed.nunique()
        q = q[q.split.isin(complete[complete == 3].index)]
        if not len(q):
            continue
        per_split = q.groupby("split").test_bacc
        m_ = per_split.mean()
        common = full.index.intersection(m_.index)
        d_ = (full[common] - m_[common])
        tail = "-- & --" if arm == "full" else f"{d_.mean():+.3f} $\\pm$ {d_.std():.3f} & {int((d_ > 0).sum())}/{len(common)}"
        rows.append(line(n, str(q.split.nunique()), f3(m_.mean()) + " $\\pm$ " + f3(m_.std()), f3(per_split.std().mean()), f3(q.groupby("split").ece_ts.mean().mean()), f3(q.groupby("split").recall_G.mean().mean()), tail))
    write("s6_seeds.tex", rows)
    print("".join(rows))
ps_ = pd.read_csv("results/review/per_source.csv")
write("s4_persource_full.tex", [line(r.method if i % 8 == 0 else "", SRC[r.source] + ("$^\\ast$" if "/" not in r.classes else ""), f"{int(r.n):,}", r.classes, f3(r.acc), f3(r.bacc), f3(r.macro_f1), f3(r.ece_ts)) for i, r in enumerate(ps_.itertuples())])
dl = pd.read_csv("results/review/disease_level.csv")
write("s4_disease.tex", [line(r.method, "Test" if r.view == "test" else "Ch\\'ak\\d{s}u", r.disease, f3(r.sensitivity), f3(r.specificity), f3(r.ppv), f3(r.auroc)) for r in dl.itertuples()])
sc_ = pd.read_csv("results/review/scope_subset.csv")
write("s4_scope.tex", [line(r.method, f3(r.all_test), f3(r.without_single_class), f3(r.multiclass_full_scope)) for r in sc_.itertuples()])
print("[ok review]")
sp_ = pd.read_csv("results/review/site_probe.csv")
write("s6_siteprobe.tex", [line(esc(r.features), r.view, f"{int(r.n):,}", f3(r.disease), f3(r.site_all), f3(r.site_within_N), f3(r.site_within_D)) for r in sp_.itertuples()])
print("[ok site]")
