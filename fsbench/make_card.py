"""Step 6a: render the Hugging Face dataset card from hf/card_template.md + computed tables.

python -m fsbench.make_card --bench data/bench --sources configs/fundusshift_open.yaml --out hf/README.md
Placeholders: {{SPLIT_TABLE}} {{SOURCE_TABLE}} {{UNUSED_TABLE}} {{REUSE_TABLE}} {{CHECKS_TABLE}} {{N_*}} {{SHORTCUT}}
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from .common import CLASSES

SPLITS = ["train", "validation", "calibration", "test", "test_ood", "test_openset"]


def md(df):
    df = df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def probe_section(path):
    r = json.loads(Path(path).read_text())
    rows = [("validation (in-distribution)", "validation"), ("calibration", "calibration"),
            ("**test** (held-out collections)", "test"), ("**test_ood** (Chákṣu, handheld)", "test_ood")]
    classes = list(r["test"]["recall"]) if "classes" not in r else r["classes"]
    t = f"| split | balanced acc | macro-F1 | recall {' / '.join(classes)} | worst collection (bAcc) |\n|---|---|---|---|---|\n"
    for name, k in rows:
        m = r[k]
        rec = " / ".join(f"{m['recall'][c]:.2f}" if c in m["recall"] else "–" for c in classes)
        w = min(m["per_source_bacc"], key=m["per_source_bacc"].get)
        t += f"| {name} | {m['balanced_accuracy']:.3f} | {m['macro_f1']:.3f} | {rec} | {w} ({m['per_source_bacc'][w]:.3f}) |\n"
    ps = "| test collection | bAcc |\n|---|---|\n" + "".join(
        f"| {k} | {v:.3f} |\n" for k, v in sorted(r["test"]["per_source_bacc"].items(), key=lambda x: x[1]))
    drop = r["validation"]["balanced_accuracy"] - r["test"]["balanced_accuracy"]
    return (f"## Reference result: frozen-feature linear probe\n\n"
            f"Frozen {r['model'].split('/')[-1]} features (CLS + mean patch token, 224 px), logistic regression with "
            f"(collection, class)-balanced weights, C selected on validation (`code/baselines/linear_probe.py`). "
            f"A difficulty check, not a tuned baseline.\n\n{t}\n{ps}\n"
            f"Open-set AUROC (max-softmax, `test_openset` vs `test`): **{r['test_openset_auroc_msp']:.3f}**.\n\n"
            f"Balanced accuracy drops by **{drop * 100:.1f} points** from validation to the held-out collections. "
            f"Single-class collections (e.g. HYGD) report recall of that class."
            + (f"\n\n**Task 3 (DR grading, same frozen features):** quadratic weighted kappa "
               f"{r['dr_grading']['validation']['quadratic_kappa']:.3f} on validation vs "
               f"**{r['dr_grading']['test']['quadratic_kappa']:.3f}** on held-out "
               f"{', '.join(r['dr_grading']['test']['sources'])} (n = {r['dr_grading']['test']['n']})."
               if "dr_grading" in r else ""))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "hf/card_template.md"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", help="probe_metrics.json from baselines/linear_probe.py")
    a = ap.parse_args(argv)
    bench = Path(a.bench)
    src = {s["id"]: s for s in yaml.safe_load(open(a.sources))["sources"]}
    d = pd.read_csv(bench / "splits_final.csv", dtype=str, keep_default_na=False)
    u = d[d.split.isin(SPLITS)]

    st = pd.crosstab(u.split, u.label).reindex(index=SPLITS, fill_value=0)
    st = st[[c for c in [*CLASSES, "OPENSET"] if c in st.columns]]
    st["total"] = st.sum(axis=1)
    st.loc["**all**"] = st.sum()
    st.index.name = "split"

    side = u.groupby("source").split.agg(lambda s: "/".join(sorted(set(s))))
    sc = pd.crosstab(u.source, u.label).reindex(columns=[c for c in [*CLASSES, "OPENSET"] if c in set(u.label)],
                                                fill_value=0)
    inv = d.groupby("source").size().rename("inventory")
    sc = sc.join(inv)
    sc.insert(0, "splits", side)
    sc.insert(1, "country", [src[s].get("country", "") for s in sc.index])
    sc.insert(2, "device", [src[s].get("device", "") for s in sc.index])
    sc["licence"] = [src[s].get("licence", "") for s in sc.index]
    order = {"train/validation/calibration": 0}
    sc = sc.sort_values(["splits", "inventory"], ascending=[True, False])
    sc.index.name = "source"

    rg = pd.crosstab(u.region, u.split).reindex(columns=SPLITS, fill_value=0)
    rg.index.name = "region"
    un = d[d.split == "unused"].unused_reason.value_counts().rename("images")
    un.index.name = "reason"
    reuse = pd.read_csv(bench / "cross_source_reuse.csv")
    checks = re.search(r"## Leakage checks\n\n(.*?)\n\n", (bench / "audit_report.md").read_text(), re.S).group(1)
    sc_auc = pd.read_csv(bench / "eda/shortcut_auc.csv")
    eda = (bench / "eda/eda_report.md").read_text()
    m = re.search(r"balanced accuracy \*\*([0-9.]+)\*\*\s*\(chance ([0-9.]+)", eda)

    rep = {"{{SPLIT_TABLE}}": md(st), "{{PROBE_SECTION}}": probe_section(a.probe) if a.probe else "", "{{REGION_TABLE}}": md(rg), "{{SOURCE_TABLE}}": md(sc), "{{UNUSED_TABLE}}": md(un),
           "{{REUSE_TABLE}}": md(reuse) if len(reuse) else "none detected", "{{CHECKS_TABLE}}": checks,
           "{{N_INVENTORY}}": f"{len(d):,}", "{{N_RELEASED}}": f"{len(u):,}", "{{N_SOURCES}}": str(u.source.nunique()),
           "{{SRC_BACC}}": m.group(1) if m else "?", "{{SRC_CHANCE}}": m.group(2) if m else "?",
           "{{SHORTCUT_TABLE}}": md(sc_auc.sort_values("auc_trivial_features", ascending=False).round(3))}
    text = Path(a.template).read_text()
    for k, v in rep.items():
        text = text.replace(k, v)
    Path(a.out).write_text(text)
    print(f"card -> {a.out}")


if __name__ == "__main__":
    main()
