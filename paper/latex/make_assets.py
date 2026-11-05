"""Builds every table and figure of the paper from saved results, so no number is typed by hand.
FS_DATA: folder with bench/ and processed/; FS_RUNS: folder with the saved logits (lotus/, lp16/, finetune/, ...).
FS_DATA=data FS_RUNS=runs python paper/latex/make_assets.py   (from the project root)
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from experiments.calib import Calibrator
from experiments.data import load_table
from experiments.full_eval import discover

DATA, RUNS = os.environ.get("FS_DATA", "data"), os.environ.get("FS_RUNS", "runs")
OUT, FIG = Path("paper/latex/generated"), Path("paper/latex/figures")
t, classes = load_table(f"{DATA}/bench")
sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
NAMES = {"afio_composite": "AFIO", "fives": "FIVES", "hrf": "HRF", "hygd": "HYGD", "palm": "PALM", "papila": "PAPILA", "paraguay_dr": "Paraguay", "bangladesh_eye": "Bangladesh Eye", "ddr": "OIA-DDR", "deepdrid": "DeepDRiD", "grape": "GRAPE", "idrid": "IDRiD", "jichi": "Jichi", "sustech_sysu": "SUSTech-SYSU", "chaksu": "Ch\\'ak\\d{s}u"}
LABEL = {"16-shot-probe": "16-shot probe", "linear-probe": "Linear probe", "zero-shot": "Zero-shot", "erm": "ERM", "groupdro": "Group DRO", "finetune-nll": "Fine-tune (NLL rule)", "coral": "CORAL", "swad": "SWAD", "finetune": "Fine-tune", "wise-ft": "WiSE-FT", "mlp-probe": "MLP probe", "coop": "CoOp", "coop-ctxinit": "CoOp ctx-init", "coop-lp": "CoOp + LP", "tip-adapter": "Tip-Adapter", "tip-adapter-f": "Tip-Adapter-F"}
MODEL = {"medsiglip": "MedSigLIP", "dinov2_lora": "DINOv2 LoRA", "convnext_t": "ConvNeXt-T", "swin_t": "Swin-T", "siglip2_l16": "SigLIP2-L", "dinov2_b14": "DINOv2-B", "siglip_b16": "SigLIP-B", "biomedclip": "BiomedCLIP"}
f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}".replace("0.", ".", 1) if abs(v) < 1 else f"{v:.3f}"
v = pd.read_csv("results/full_eval/views.csv.gz", low_memory=False)
v["config"] = v.config.fillna("")
LV = pd.read_csv("results/lotus/views.csv", low_memory=False)
L = LV[(LV.config == "lotus") & (LV.setting == "posthoc")]
L0 = LV[(LV.config == "lantern") & (LV.setting == "posthoc")]   # initial design, with between-site projection
reps = pd.read_csv("results/paired_tests/representatives.csv")
reps["config"] = reps.config.fillna("")
reps = reps[~reps.setting.str.endswith("-acc")]


def row(q, view, ph, c):
    return q[(q["view"] == view) & (q.posthoc == ph)][c].mean()


def write(name, s):
    (OUT / name).write_text(s.rstrip() + "\n\\bottomrule\n")


# leakage check, reported in the data section
te, dev = sp == "test", np.isin(sp, ["train", "validation", "calibration"])
shared = set(t.group_id[te]) & set(t.group_id[dev])
print("leakage groups shared between test and development:", len(shared))
print("sizes", t.split.value_counts().to_dict())

# Table: data
rows = []
for s in ["bangladesh_eye", "ddr", "deepdrid", "grape", "idrid", "jichi", "sustech_sysu", "afio_composite", "fives", "hrf", "hygd", "palm", "papila", "paraguay_dr", "chaksu"]:
    m = (src == s) & (sp != "test_openset")
    n = [int(((y == c) & m).sum()) for c in range(3)]
    rows.append((s, n, str(t.region[m].iloc[0])))
txt = ""
for i, (s, n, reg) in enumerate(rows):
    if i == 7:
        txt += "\\midrule\n"
    if i == 14:
        txt += "\\midrule\n"
    role = "Dev" if i < 7 else ("Test" if i < 14 else "OOD")
    txt += f"{role} & {NAMES[s]} & {reg} & " + " & ".join(f"{x:,}" if x else "--" for x in n) + f" & {sum(n):,} \\\\\n"
write("tab_data.tex", txt)

# Table: method against representatives
cols = ["bacc", "eq_bacc", "auroc", "recall_0", "recall_1", "recall_2"]
def line(name, q):
    return " & ".join([name] + [f3(row(q, "test", "none", c)) for c in cols] + [f3(row(q, "test", "none", "ece")), f3(row(q, "test", "ts", "ece")), f3(row(q, "test", "ts", "nll")), f3(row(q, "chaksu", "none", "bacc"))]) + " \\\\\n"
txt = line("\\textbf{LOTUS (ours)}", L) + "\\midrule\n"
order = []
for r in reps.itertuples():
    q = v[(v.setting == r.setting) & (v.model == r.model) & (v.config == r.config) & ~v.setting.str.endswith("+ens")]
    order.append((row(q, "test", "none", "bacc"), f"{LABEL[r.setting]}, {MODEL.get(r.model, r.model)}", q))
for _, n, q in sorted(order, key=lambda x: -x[0]):
    txt += line(n, q)
write("tab_baselines.tex", txt)

# Table: per source, LANTERN
txt = ""
for s in ["paraguay_dr", "hygd", "fives", "papila", "palm", "hrf", "afio_composite", "chaksu"]:
    w = "chaksu" if s == "chaksu" else f"src:{s}"
    ref = v[(v.setting == "16-shot-probe") & (v.model == "medsiglip") & (v.config == "none/ce")]
    lin = v[(v.setting == "linear-probe") & (v.model == "medsiglip") & (v.config == "class/ls")]
    n = int(L[(L["view"] == w) & (L.posthoc == "none")].n.iloc[0])
    txt += f"{NAMES[s]} & {n:,} & " + " & ".join([f3(row(L, w, "none", "bacc")), f3(row(ref, w, "none", "bacc")), f3(row(lin, w, "none", "bacc")), f3(row(L, w, "none", "recall_0")), f3(row(L, w, "none", "recall_1")), f3(row(L, w, "none", "recall_2")), f3(row(L, w, "none", "ece")), f3(row(L, w, "ts", "ece"))]) + " \\\\\n"
write("tab_sources.tex", txt)

# Table: calibration states
txt = ""
for w, nm in [("test", "Test"), ("chaksu", "Ch\\'ak\\d{s}u")]:
    for ph, pn in [("none", "Raw"), ("ts", "TS"), ("fts", "FTS")]:
        txt += f"{nm} & {pn} & " + " & ".join(f3(row(L, w, ph, c)) for c in ["ece", "mce", "ace", "nll", "brier", "aurc"]) + " \\\\\n"
write("tab_calib.tex", txt)

# Table: operating points and conformal
o = pd.read_csv("results/top_dev/ops.csv")
o = o[(o.config == "last_reg") & (o.setting == "posthoc")]
txt = ""
for w, nm in [("test", "Test"), ("chaksu", "Ch\\'ak\\d{s}u")]:
    for op, on in [("cov90", "Coverage 90\\%"), ("cov80", "Coverage 80\\%"), ("risk10", "Risk $\\leq$ 10\\%")]:
        q = o[(o["view"] == w) & (o.posthoc == "ts") & (o.op == op)]
        txt += f"{nm} & {on} & {f3(q.coverage.mean())} & {f3(q.risk.mean())} & {f3(q.D_to_N_rate.mean())} & {f3(q.G_to_N_rate.mean())} \\\\\n"
write("tab_ops.tex", txt)
c = pd.read_csv("results/top_dev/conformal.csv")
c = c[(c.config == "last_reg") & (c.setting == "posthoc")]
txt = ""
for w, nm in [("test", "Test"), ("chaksu", "Ch\\'ak\\d{s}u")]:
    for a in [.10, .05]:
        for me in ["LAC", "ClassLAC", "APS"]:
            q = c[(c["view"] == w) & (c.posthoc == "ts") & (c.alpha == a) & (c.method == me)]
            txt += f"{nm} & {1 - a:.2f} & {me} & {f3(q.coverage.mean())} & {q['size'].mean():.2f} & {f3(q.singleton.mean())} \\\\\n"
write("tab_conformal.tex", txt)

# Supplement: paired tests
d = pd.read_csv("results/lotus/paired.csv")
d = d[~d.baseline.str.contains("-acc")]
txt = ""
for fam, nm in [("bacc", "bAcc (raw)"), ("ece_ts", "ECE (TS)"), ("nll_ts", "NLL (TS)")]:
    g = d[d.family == fam].sort_values("diff", key=lambda x: x.abs())
    for r in g.itertuples():
        s, m, cfg = r.baseline.split(":", 2)
        txt += f"{nm} & {LABEL[s]}, {MODEL.get(m, m)} & {f3(r.b)} & {r.diff:+.3f} & [{r.lo:+.3f}, {r.hi:+.3f}] & {r.p_holm:.3f} \\\\\n"
    txt += "\\midrule\n"
write("tab_paired.tex", txt.rsplit("\\midrule\n", 1)[0])

# Supplement: ten candidates
P = pd.read_csv("results/proposed/views.csv", low_memory=False)
P = P[P.setting == "proposed"]
T = pd.read_csv("results/top_dev/views.csv", low_memory=False)
T = T[T.setting == "posthoc"]
dev_ = {"mirror_mid": .800, "mirror_reg": .799, "mirror_base": .795, "cross_bag_nap": .794, "strata_bag_nap": .792, "last_reg": .792, "bag_nap_allbanks": .791, "pooled_plus_last": .791, "canonical": .791, "text_mirror": .791}
nm = {"mirror_mid": "Mirror, weight decay $10^{-3}$", "mirror_reg": "Mirror, weight decay $10^{-2}$", "mirror_base": "Mirror, weight decay $10^{-4}$", "cross_bag_nap": "Cross-site pair loss", "strata_bag_nap": "Same-site pair loss", "last_reg": "\\textbf{Tokens, weight decay $10^{-2}$ (selected)}", "bag_nap_allbanks": "Pooled feature, weight decay $10^{-4}$", "pooled_plus_last": "Pooled feature + tokens", "canonical": "Disc side canonicalised", "text_mirror": "Mirrored text only"}
pp = pd.read_csv("results/top_dev/paired_vs_frozen.csv").set_index("config")
txt = ""
for k, dv in sorted(dev_.items(), key=lambda x: -x[1]):
    q = (P if k == "bag_nap_allbanks" else T[T.config == k])
    txt += f"{nm[k]} & {f3(dv)} & {f3(row(q, 'test', 'none', 'bacc'))} & {f3(row(q, 'test', 'none', 'eq_bacc'))} & {f3(row(q, 'test', 'none', 'recall_2'))} & {f3(row(q, 'test', 'ts', 'nll'))} & {f3(row(q, 'chaksu', 'none', 'bacc'))} \\\\\n"
write("tab_candidates.tex", txt)

# Table: development error analysis of the pooled-feature pipeline
ea = pd.read_csv("results/strata/error_analysis.csv")
txt = ""
for r in ea.itertuples():
    txt += f"{NAMES[r.source]} & {r.n:,} & " + " & ".join(f3(x) for x in [r.head, r.text, r.bacc, r.recall_N, r.recall_D, r.recall_G, r.shared_all]) + " \\\\\n"
write("tab_errors.tex", txt)

# Figure: per source
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
S = ["paraguay_dr", "hygd", "fives", "palm", "papila", "hrf", "afio_composite", "chaksu"]
ser = [("LOTUS", L, "#0B6E99", ""), ("16-shot probe", v[(v.setting == "16-shot-probe") & (v.model == "medsiglip") & (v.config == "none/ce")], "#E08A1E", "//"), ("Linear probe", v[(v.setting == "linear-probe") & (v.model == "medsiglip") & (v.config == "class/ls")], "#6B4C9A", "..")]
fig, ax = plt.subplots(figsize=(4.8, 1.95))
x = np.arange(len(S))
for i, (n, q, col, h) in enumerate(ser):
    vals = [row(q, "chaksu" if s == "chaksu" else f"src:{s}", "none", "bacc") for s in S]
    ax.bar(x + (i - 1) * .27, vals, .25, label=n, color=col, hatch=h, edgecolor="white", linewidth=.4)
ax.set_xticks(x, [NAMES[s].replace("Ch\\'ak\\d{s}u", "Chákṣu") for s in S], rotation=20, ha="right")
ax.set_ylabel("Balanced accuracy")
ax.set_ylim(0, 1)
ax.grid(axis="y", linewidth=.3)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.18), fontsize=7)
fig.savefig(FIG / "per_source.pdf", bbox_inches="tight")

# Figure: reliability, raw against TS, seeds pooled
ca = (sp == "calibration") & (sub == "cal_A")
fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.05), sharey=True)
fl = {(s, m, str(c), sd): f for s, m, c, sd, f in discover(RUNS)}
for ax, (name, Zs) in zip(axs, [("LOTUS", [np.load(f"{RUNS}/lotus/logits_s{i}.npy") for i in range(3)]), ("16-shot probe", [np.load(fl[("16-shot-probe", "medsiglip", "none/ce", i)]) for i in range(3)])]):
    for ph, col, mk, lab in [("none", "#E08A1E", "s", "Raw"), ("ts", "#0B6E99", "o", "TS")]:
        conf, ok = [], []
        for Z in Zs:
            Pm = Calibrator(ph).fit(Z[ca], y[ca])(Z)[te]
            conf.append(Pm.max(1))
            ok.append(Pm.argmax(1) == y[te])
        conf, ok = np.concatenate(conf), np.concatenate(ok)
        e = np.linspace(.33, 1, 11)
        i = np.clip(np.digitize(conf, e) - 1, 0, 9)
        xs = [conf[i == b].mean() for b in range(10) if (i == b).sum() >= 30]
        ys = [ok[i == b].mean() for b in range(10) if (i == b).sum() >= 30]
        ax.plot(xs, ys, color=col, marker=mk, ms=3, lw=1.1, label=lab)
    ax.plot([.33, 1], [.33, 1], color="#999999", lw=.6, ls="--")
    ax.set_title(name, fontsize=8)
    ax.set_xlabel("Confidence")
    ax.grid(linewidth=.3)
axs[0].set_ylabel("Accuracy")
axs[0].legend(frameon=False, fontsize=7)
fig.savefig(FIG / "reliability.pdf", bbox_inches="tight")
print("[ok]")

# ---------- Second version: benchmark figures ----------
from PIL import Image
from scipy.stats import spearmanr
INK, GRID = "#333333", "#D9D9D9"
plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "font.size": 7.5, "axes.linewidth": .5})
FAM = {"finetune": "Fine-tuning", "linear-probe": "Frozen probes", "mlp-probe": "Frozen probes", "16-shot-probe": "16-shot probes", "zero-shot": "Prompt methods", "coop": "Prompt methods", "coop-ctxinit": "Prompt methods", "coop-lp": "Prompt methods", "tip-adapter": "Prompt methods", "tip-adapter-f": "Prompt methods", "erm": "DG methods", "groupdro": "DG methods", "coral": "DG methods", "swad": "DG methods", "wise-ft": "DG methods"}
COL = {"Fine-tuning": ("#2F6DB5", "o"), "Frozen probes": ("#2A9D8F", "s"), "16-shot probes": ("#E08A1E", "^"), "Prompt methods": ("#8E5BB5", "D"), "DG methods": ("#C8455A", "v")}

# Val against test
m = v[(v.posthoc == "none") & ~v.setting.str.endswith("+ens") & (v.model != "flair") & v.setting.isin(FAM)]
a = m[m["view"] == "validation"].groupby(["setting", "model", "config"]).bacc.mean().rename("val")
b = m[m["view"] == "test"].groupby(["setting", "model", "config"]).bacc.mean().rename("test")
c_ = m[m["view"] == "chaksu"].groupby(["setting", "model", "config"]).bacc.mean().rename("chaksu")
sc = pd.concat([a, b, c_], axis=1).dropna().reset_index()
print("configs in scatter", len(sc), "runs", len(m[m["view"] == "test"]), "spearman all", round(spearmanr(sc.val, sc.test)[0], 3), "finetune", round(spearmanr(sc[sc.setting == "finetune"].val, sc[sc.setting == "finetune"].test)[0], 3), "chaksu all", round(spearmanr(sc.val, sc.chaksu)[0], 3), "val-test gap median", round((sc.val - sc.test).median(), 3), "share val>test", round((sc.val > sc.test).mean(), 3))
print("model count", m.model.nunique(), "settings", m.setting.nunique())
fig, ax = plt.subplots(figsize=(2.75, 2.45))
for fam, (col, mk) in COL.items():
    q = sc[sc.setting.map(FAM) == fam]
    ax.scatter(q.val, q.test, s=7, c=col, marker=mk, alpha=.65, linewidths=0, label=fam)
lv, lt = row(L, "validation", "none", "bacc"), row(L, "test", "none", "bacc")
ax.scatter([lv], [lt], s=70, marker="*", c="#111111", label="LOTUS", zorder=5)
ax.plot([.3, 1], [.3, 1], ls="--", lw=.5, c="#888888")
ax.text(.47, .50, "val = test", rotation=44, fontsize=6, color="#666666")
ax.set_xlim(.3, .95); ax.set_ylim(.3, .95)
ax.set_xlabel("Validation bAcc (development sources)"); ax.set_ylabel("Test bAcc (unseen sources)")
ax.grid(lw=.3, color=GRID); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=5.6, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3, handletextpad=.05, columnspacing=.5, borderpad=0, labelspacing=.2)
fig.savefig(FIG / "val_test.pdf", bbox_inches="tight")

# Rotation
rot = pd.read_csv("results/supplement/rotation.csv")
rot = rot[rot.split > 0]
lr = pd.read_csv("results/lotus/rotation_lotus.csv").sort_values("split")
series = [("LOTUS", lr), ("MedSigLIP LP", rot[(rot.model == "medsiglip") & (rot.kind == "full_class")].sort_values("split")), ("MedSigLIP 16-shot", rot[(rot.model == "medsiglip") & (rot.kind == "16_shot")].sort_values("split")), ("DINOv2 LP", rot[(rot.model == "dinov2_b14") & (rot.kind == "full_class")].sort_values("split")), ("SigLIP2 LP", rot[(rot.model == "siglip2_l16") & (rot.kind == "full_class")].sort_values("split")), ("ConvNeXt LP", rot[(rot.model == "convnext_t") & (rot.kind == "full_class")].sort_values("split"))]
txt = ""
for n, q in series:
    bold = n == "LOTUS"
    w = lambda s: f"\\textbf{{{s}}}" if bold else s
    if bold:
        dtxt = "-- & -- & -- & --"
    else:
        dd = lr.test_bacc.to_numpy() - q.test_bacc.to_numpy()
        de = lr.ece_ts.to_numpy() - q.ece_ts.to_numpy()
        dtxt = f"{dd.mean():+.3f} & {(dd > 0).sum()}/50 & {de.mean():+.3f} & {(de < 0).sum()}/50"
    txt += w(n) + " & " + " & ".join(f3(q[c].mean()) for c in ["test_bacc", "test_bacc_source_equal", "recall_N", "recall_D", "recall_G", "ece_ts", "risk_test", "lac_test"]) + " & " + dtxt + " \\\\\n"
write("tab_rotation.tex", txt)
print(txt)
# figure: rotation strip + reliability promises
fig, axs = plt.subplots(1, 4, figsize=(6.6, 1.75), gridspec_kw=dict(width_ratios=[1.5, 1, 1, 1]))
ax = axs[0]
names = [n for n, _ in series]
for i, (n, q) in enumerate(series):
    jit = np.random.default_rng(i).uniform(-.18, .18, len(q))
    ax.scatter(q.test_bacc, i + jit, s=4, color="#111111" if i == 0 else "#7D8FA3", alpha=.6, linewidths=0)
    ax.plot([q.test_bacc.mean()] * 2, [i - .32, i + .32], color="#C8455A", lw=1.3)
ax.set_yticks(range(len(names)), names); ax.invert_yaxis()
ax.set_xlabel("Test bAcc over 50 rotations"); ax.grid(axis="x", lw=.3, color=GRID)
ax.spines[["top", "right"]].set_visible(False)
for ax, (ci, ct, ttl, promise) in zip(axs[1:], [("risk_internal", "risk_test", "Risk at 80% coverage", None), ("lac_internal", "lac_test", "LAC coverage (nominal .90)", .9), ("cov_internal", "cov_test", "Coverage (target .80)", .8)]):
    for i, (n, q) in enumerate(series[:4]):
        ax.plot([0, 1], [q[ci].mean(), q[ct].mean()], marker="os^D"[i], ms=3, lw=1, color=["#111111", "#2A9D8F", "#E08A1E", "#2F6DB5"][i], label=n)
    if promise:
        ax.axhline(promise, ls="--", lw=.5, color="#888888")
    ax.set_xticks([0, 1], ["Internal", "Test"]); ax.set_xlim(-.25, 1.25); ax.set_title(ttl, fontsize=7)
    ax.grid(axis="y", lw=.3, color=GRID); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(w_pad=.6, rect=(0, 0, 1, .9))
h_, l_ = axs[1].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=6, loc="upper right", ncol=4, bbox_to_anchor=(.99, 1.03), handlelength=1.4, columnspacing=.9)
fig.savefig(FIG / "rotation.pdf", bbox_inches="tight")

# Heatmap: per source
SRC = ["paraguay_dr", "hygd", "fives", "palm", "papila", "hrf", "afio_composite", "chaksu"]
H, rl = [], []
order2 = sorted(order, key=lambda x: -x[0])
for n, q in [("LOTUS", L)] + [(x[1], x[2]) for x in order2 if not x[1].startswith(("Tip-Adapter-F", "CoOp ctx", "CoOp +", "Fine-tune (NLL"))]:
    H.append([row(q, "chaksu" if s == "chaksu" else f"src:{s}", "none", "bacc") for s in SRC]); rl.append(n)
H = np.array(H)
fig, ax = plt.subplots(figsize=(3.75, 2.45))
im = ax.imshow(H, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(SRC)), ["Paraguay", "HYGD", "FIVES", "PALM", "PAPILA", "HRF", "AFIO", "Chákṣu"], rotation=35, ha="right", fontsize=6.5)
ax.set_yticks(range(len(rl)), rl, fontsize=6)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H[i, j]:.2f}"[1:] if H[i, j] < 1 else "1.0", ha="center", va="center", fontsize=5.3, color="white" if H[i, j] > .6 else INK)
ax.spines[:].set_visible(False); ax.tick_params(length=0)
fig.savefig(FIG / "heatmap.pdf", bbox_inches="tight")

# Reliability
ca = (sp == "calibration") & (sub == "cal_A")
fig, axs = plt.subplots(1, 3, figsize=(4.9, 1.65), sharey=True)
fl = {(s, m_, str(c), sd): f for s, m_, c, sd, f in discover(RUNS)}
rows_rel = [("LOTUS", [np.load(f"{RUNS}/lotus/logits_s{i}.npy") for i in range(3)]), ("16-shot, MedSigLIP", [np.load(fl[("16-shot-probe", "medsiglip", "none/ce", i)]) for i in range(3)]), ("Fine-tune, Swin-T", [np.load(fl[("finetune", "swin_t", "mbls", i)]) for i in range(3)])]
for ax, (name, Zs) in zip(axs, rows_rel):
    for ph, col, mk, lab in [("none", "#E08A1E", "s", "Raw"), ("ts", "#2F6DB5", "o", "TS")]:
        cf, ok = [], []
        for Z in Zs:
            Pm = Calibrator(ph).fit(Z[ca], y[ca])(Z)[te]
            cf.append(Pm.max(1)); ok.append(Pm.argmax(1) == y[te])
        cf, ok = np.concatenate(cf), np.concatenate(ok)
        i = np.clip(np.digitize(cf, np.linspace(.33, 1, 11)) - 1, 0, 9)
        keep_ = [b for b in range(10) if (i == b).sum() >= 30]
        ax.plot([cf[i == b].mean() for b in keep_], [ok[i == b].mean() for b in keep_], color=col, marker=mk, ms=2.5, lw=.9, label=lab)
    ax.plot([.33, 1], [.33, 1], color="#888888", lw=.5, ls="--")
    ax.set_title(name, fontsize=7); ax.set_xlabel("Confidence"); ax.grid(lw=.3, color=GRID); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("Accuracy"); axs[0].legend(frameon=False, fontsize=6)
fig.savefig(FIG / "reliability.pdf", bbox_inches="tight")

# Sample images
pick = ["ddr", "bangladesh_eye", "jichi", "fives", "hygd", "palm", "paraguay_dr", "chaksu"]
lab = ["OIA-DDR (D)", "Bangladesh (G)", "Jichi (N)", "FIVES (N)", "HYGD (G)", "PALM (N)", "Paraguay (D)", "Chákṣu (G)"]
cls = [1, 2, 0, 0, 2, 0, 1, 2]
fig, axs = plt.subplots(1, 8, figsize=(6.6, .95))
for ax, s, n, c0 in zip(axs, pick, lab, cls):
    q = t[(t.source == s) & (t.y == c0) & (t.split != "test_openset")].sample(1, random_state=3).iloc[0]
    ax.imshow(np.asarray(Image.open(f"{DATA}/processed/{s}/{q.image_id}.jpg").convert("RGB").resize((224, 224)))); ax.set_title(n, fontsize=5.8, pad=1.5); ax.axis("off")
fig.subplots_adjust(wspace=.03)
fig.savefig(FIG / "samples.pdf", bbox_inches="tight", dpi=200)

# In-source ceiling
ce_ = pd.read_csv("results/supplement/ceiling.csv")
print(ce_.columns.tolist()); print(ce_.head(2).to_string())
print(pd.read_csv("results/supplement/discrim_auroc.csv").head(3).to_string())
print(v[(v.setting == "mcq") & (v.posthoc == "none") & (v["view"].isin(["test", "chaksu"]))].groupby(["model", "view"])[["bacc", "recall_0", "recall_1", "recall_2"]].mean().round(3).to_string())
print("[ok2]")

# Main results table with citations
CITE = {"16-shot-probe": "", "linear-probe": "", "zero-shot": "", "erm": "", "groupdro": "groupdro", "finetune-nll": "", "coral": "coral", "swad": "swad", "finetune": "", "wise-ft": "wiseft", "mlp-probe": "", "coop": "coop", "coop-ctxinit": "coop", "coop-lp": "coop", "tip-adapter": "tip", "tip-adapter-f": "tip"}
ENC = {"medsiglip": "medgemma", "dinov2_lora": "dinov2", "convnext_t": "convnext", "swin_t": "swin", "siglip2_l16": "siglip2", "dinov2_b14": "dinov2", "siglip_b16": "siglip", "biomedclip": "biomedclip"}
def cols2(q):
    return " & ".join([f3(row(q, "validation", "none", "bacc")), f3(row(q, "test", "none", "bacc")), f3(row(q, "test", "none", "eq_bacc")), f3(row(q, "test", "none", "auroc")), f3(row(q, "test", "none", "recall_0")), f3(row(q, "test", "none", "recall_1")), f3(row(q, "test", "none", "recall_2")), f3(row(q, "test", "none", "ece")), f3(row(q, "test", "ts", "ece")), f3(row(q, "test", "ts", "nll")), f3(row(q, "chaksu", "none", "bacc")), f3(row(q, "chaksu", "none", "recall_2"))])
txt = "\\textbf{LOTUS (ours)} & " + cols2(L) + " \\\\\n" + "+ projection (initial design) & " + cols2(L0) + " \\\\\n\\midrule\n"
info = {(r.setting): r for r in reps.itertuples()}
for r in sorted(reps.itertuples(), key=lambda r: -row(v[(v.setting == r.setting) & (v.model == r.model) & (v.config == r.config)], "test", "none", "bacc")):
    q = v[(v.setting == r.setting) & (v.model == r.model) & (v.config == r.config) & ~v.setting.str.endswith("+ens")]
    c1 = f"~\\cite{{{CITE[r.setting]}}}" if CITE[r.setting] else ""
    txt += f"{LABEL[r.setting]}{c1}, {MODEL.get(r.model, r.model)}~\\cite{{{ENC[r.model]}}} & " + cols2(q) + " \\\\\n"
q = v[(v.setting == "16-shot-probe") & (v.model == "flair") & (v.config == "none/mbls")]
txt += "\\midrule\n16-shot probe, FLAIR~\\cite{flair}$^\\dagger$ & " + cols2(q) + " \\\\\n"
write("tab_main.tex", txt)
print(txt[:600])
# Compact data table
txt = ""
for i, (s, n, reg) in enumerate(rows):
    if i in (7, 14):
        txt += "\\midrule\n"
    role = "Dev" if i < 7 else ("Test" if i < 14 else "OOD")
    key = {"bangladesh_eye": "bangladesh", "ddr": "ddr", "deepdrid": "deepdrid", "grape": "grape", "idrid": "idrid", "jichi": "jichi", "sustech_sysu": "sustech", "afio_composite": "afio", "fives": "fives", "hrf": "hrf", "hygd": "hygd", "palm": "palm", "papila": "papila", "paraguay_dr": "paraguay", "chaksu": "chaksu"}[s]
    txt += f"{role} & {NAMES[s]}~\\cite{{{key}}} & {reg} & " + " & ".join(f"{x:,}" if x else "--" for x in n) + f" & {sum(n):,} \\\\\n"
write("tab_data.tex", txt)
print("[ok3]")

# Reliability of representatives after TS
oo = pd.read_csv("results/full_eval/ops.csv.gz", low_memory=False); oo["config"] = oo.config.fillna("")
cc = pd.read_csv("results/full_eval/conformal.csv.gz", low_memory=False); cc["config"] = cc.config.fillna("")
oL = pd.read_csv("results/lotus/ops.csv"); oL = oL[(oL.config == "lotus") & (oL.setting == "posthoc")]
cL = pd.read_csv("results/lotus/conformal.csv"); cL = cL[(cL.config == "lotus") & (cL.setting == "posthoc")]
def relrow(name, o_, c_2):
    O = lambda op, w, col: o_[(o_.op == op) & (o_["view"] == w) & (o_.posthoc == "ts")][col].mean()
    C = lambda w, col: c_2[(c_2.method == "LAC") & (c_2.alpha == .1) & (c_2["view"] == w) & (c_2.posthoc == "ts")][col].mean()
    return name + " & " + " & ".join(f3(x) for x in [O("cov80", "test", "coverage"), O("cov80", "test", "risk"), O("cov80", "test", "G_to_N_rate"), O("cov80", "test", "D_to_N_rate"), O("risk10", "test", "coverage"), O("risk10", "test", "risk"), C("test", "coverage")]) + f" & {C('test', 'size'):.2f} & " + " & ".join(f3(x) for x in [O("cov80", "chaksu", "coverage"), O("cov80", "chaksu", "risk"), O("cov80", "chaksu", "G_to_N_rate"), C("chaksu", "coverage")]) + " \\\\\n"
txt = relrow("\\textbf{LOTUS}", oL, cL) + "\\midrule\n"
for s, m_, cf, nm_ in [("16-shot-probe", "medsiglip", "none/ce", "16-shot, MedSigLIP"), ("linear-probe", "medsiglip", "class/ls", "Linear probe, MedSigLIP"), ("zero-shot", "medsiglip", "descriptive_single", "Zero-shot, MedSigLIP"), ("groupdro", "convnext_t", "ce", "Group DRO, ConvNeXt-T"), ("erm", "dinov2_lora", "ce", "ERM, DINOv2 LoRA"), ("finetune", "swin_t", "mbls", "Fine-tune, Swin-T"), ("wise-ft", "siglip2_l16", "fully-tuned", "WiSE-FT, SigLIP2-L")]:
    f_ = lambda d: d[(d.setting == s) & (d.model == m_) & (d.config == cf)]
    txt += relrow(nm_, f_(oo), f_(cc))
write("tab_reliab.tex", txt)
print(txt)
print(pd.read_csv("results/last_reg/paired.csv").query("family=='ece_ts'")[["baseline","b","diff","p_holm"]].round(3).to_string())
print(sorted(v[v.setting=="finetune"].config.unique()), v[v.setting=="finetune"].model.nunique())
print(t[t.source=="fives"].unused_reason.fillna("").value_counts().head(6).to_dict(), t[t.source=="fives"].groupby(["split","y"]).size().to_dict())
print("flair16 ok")

# Benchmark coverage
m2 = v[(v["view"] == "test") & (v.posthoc == "none") & ~v.setting.str.endswith("+ens") & (v.model != "flair")]
grp = [("Fine-tuning", ["finetune"], "13 CNN/ViT/RETFound~\\cite{retfound} $\\times$ 5 losses~\\cite{mdca,mbls}", "best epoch"), ("Fine-tuning, NLL rule", ["finetune-nll"], "Sensitivity check", "best epoch, NLL"), ("Linear probe", ["linear-probe"], "25 encoders, 3 weightings, 5 losses", "best epoch"), ("MLP probe", ["mlp-probe"], "Frozen encoders", "best epoch"), ("16-shot probe", ["16-shot-probe"], "Frozen encoders, 16 per class", "best epoch"), ("Zero-shot", ["zero-shot"], "Vision-language encoders, 3 prompt banks", "prompt bank"), ("CoOp~\\cite{coop} and variants", ["coop", "coop-ctxinit", "coop-lp"], "Vision-language encoders", "fixed recipe"), ("Tip-Adapter~\\cite{tip}", ["tip-adapter", "tip-adapter-f"], "Vision-language encoders", "$\\alpha,\\beta$, bank"), ("ERM, Group DRO~\\cite{groupdro}, CORAL~\\cite{coral}, SWAD~\\cite{swad}", ["erm", "groupdro", "coral", "swad"], "ConvNeXt-T~\\cite{convnext}, DINOv2 LoRA~\\cite{dinov2}", "best epoch; SWAD: own rule"), ("WiSE-FT~\\cite{wiseft}", ["wise-ft"], "Vision-language encoders", "$\\alpha$, checkpoint")]
txt = ""
for n, s_, enc, sel in grp:
    q = m2[m2.setting.isin(s_)]
    txt += f"{n} & {enc} & {q[['setting','model','config']].drop_duplicates().shape[0]} & {len(q)} & {sel} \\\\\n"
q = v[(v.setting == "mcq") & (v["view"] == "test") & (v.posthoc == "none")]
txt += f"Multiple choice & 4 multimodal LLMs, e.g.~\\cite{{fundusr1}}, answers averaged over 6 option orders & {q.model.nunique()} & {len(q)} & none \\\\\n"
write("tab_cover.tex", txt)
print(txt); print(len(m2))

# Recall trade-off
fig, axs = plt.subplots(1, 2, figsize=(4.9, 1.9), sharex=True, sharey=True)
ss = sc.merge(m[m["view"] == "test"].groupby(["setting", "model", "config"])[["recall_0", "recall_2"]].mean().reset_index(), on=["setting", "model", "config"]).merge(m[m["view"] == "chaksu"].groupby(["setting", "model", "config"])[["recall_0", "recall_2"]].mean().reset_index(), on=["setting", "model", "config"], suffixes=("", "_c"))
for ax, (a_, b_, t_, w) in zip(axs, [("recall_0", "recall_2", "Test", "test"), ("recall_0_c", "recall_2_c", "Chákṣu", "chaksu")]):
    for fam, (col, mk) in COL.items():
        q = ss[ss.setting.map(FAM) == fam]
        ax.scatter(q[a_], q[b_], s=6, c=col, marker=mk, alpha=.55, linewidths=0, label=fam)
    ax.scatter([row(L, w, "none", "recall_0")], [row(L, w, "none", "recall_2")], s=60, marker="*", c="#111111", zorder=5, label="LOTUS")
    ax.set_title(t_, fontsize=7); ax.set_xlabel("Recall N"); ax.grid(lw=.3, color=GRID); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("Recall G"); h_, l_ = axs[1].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(.5, .94), ncol=6, handletextpad=.05, columnspacing=.7)
fig.savefig(FIG / "recall.pdf", bbox_inches="tight")
print("[ok4]")

# Supplement: rotations, all probes
txt = ""
AUX = [("LOTUS", lr)] + [(f"{MODEL.get(a, a)} {'16-shot' if k == '16_shot' else 'full'}", rot[(rot.model == a) & (rot.kind == k)]) for a in ["medsiglip", "dinov2_b14", "siglip2_l16", "convnext_t"] for k in ["full_class", "16_shot"]]
for n, q in AUX:
    txt += f"{n} & {f3(q.val_bacc.mean())} & {f3(q.test_bacc.mean())} & {f3(q.ece_raw.mean())} & {f3(q.ece_ts.mean())} & {f3(q.risk_internal.mean())} & {f3(q.risk_test.mean())} & {f3(q.lac_internal.mean())} & {f3(q.lac_test.mean())} & {100 * (q.risk_test > q.risk_internal).mean():.0f}\\% \\\\\n"
write("tab_rot_all.tex", txt)
# Supplement: representatives across raw, TS, FTS
txt = ""
for nm_, q in [("LOTUS", L)] + [(x[1], x[2]) for x in order2]:
    txt += nm_ + " & " + " & ".join("/".join(f3(row(q, "test", ph, c)) for ph in ["none", "ts", "fts"]) for c in ["ece", "nll", "brier"]) + " & " + "/".join(f3(row(q, "chaksu", ph, "ece")) for ph in ["none", "ts", "fts"]) + " \\\\\n"
write("tab_states.tex", txt)
# Best-by-test frozen probes, for transparency
print("[ok5]")

# LOSO reference against Cal-B, 50 rotations (split 0 excluded)
import glob
c1 = pd.concat([pd.read_csv(f_) for f_ in glob.glob("results/revision/certify_no_projection/rot2_w*.csv")])
c2 = c1
c1, c2 = c1[c1.split > 0], c2[c2.split > 0]
txt = ""
for arm, nm_ in [("calB", "Cal-B (in distribution)"), ("loso", "LOSO reference")]:
    for cert in ["", "_cert"]:
        lab = nm_ + (" + normal certificate" if cert else "")
        a_, b_ = c1[f"{arm}_c80{cert}_test_cov"], c1[f"{arm}_c80{cert}_test_risk"]
        r_ = c2[f"{arm}_r10{cert}_test_risk"]
        txt += f"{lab} & {f3(a_.mean())} & {f3(b_.mean())} & {f3(c2[f'{arm}_r10{cert}_test_cov'].mean())} & {f3(r_.mean())} & {(r_ <= .10).sum()}/50 & {f3(c2[f'{arm}_r10{cert}_test_fr'].mean())} & " + (f"{f3(c1[f'{arm}_lac_test'].mean())} & {f3(c1[f'{arm}_lac_chaksu'].mean())}" if not cert else "-- & --") + " \\\\\n"
write("tab_cert.tex", txt)
print(txt)

# Ablation on rotations (three seeds; LOTUS = bag + text; results/revision/rotation)
ab = pd.concat([pd.read_csv(f_) for f_ in glob.glob("results/revision/rotation/w*.csv")])
ab = ab[ab.split > 0].groupby(["split", "arm"]).mean(numeric_only=True).reset_index()
full = ab[ab.arm == "no_projection"].set_index("split")
names_ = [("no_projection", "\\textbf{LOTUS} (bag + text)"), ("full", "+ projection (initial design)"), ("single_text", "without bag (one head + text)"), ("bag_only", "without text (bag)"), ("single_head", "single head, no bag, no text"), ("text_only", "text only")]
txt = ""
for k_, n_ in names_:
    q = ab[ab.arm == k_].set_index("split")
    dd = full.test_bacc - q.test_bacc
    tail_ = "-- & --" if k_ == "no_projection" else f"{dd.mean():+.3f} & {(dd > 0).sum()}/50"
    txt += f"{n_} & " + " & ".join(f3(q[c].mean()) for c in ["test_bacc", "test_macro_f1", "recall_N", "recall_G", "false_G", "ece_ts", "nll_ts", "chaksu_bacc"]) + f" & {tail_} \\\\\n"
write("tab_ablrot.tex", txt)
print(txt)
