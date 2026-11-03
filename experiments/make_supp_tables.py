"""Supplementary tables for the data experiments, from data_exps outputs only.
python -m experiments.make_supp_tables --bench data/bench --exps runs/data_exps --out docs/supplementary_data_experiments.md
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def md(d):
    return d.round(3).to_markdown(index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--exps", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    s = pd.read_csv(Path(a.bench) / "splits_final.csv", dtype=str, keep_default_na=False)
    s = s[s.split != "unused"]
    ex, sm = Path(a.exps), json.loads((Path(a.exps) / "summary.json").read_text())
    role = s.groupby("source").split.agg(lambda x: "+".join(sorted(set(x))))
    t1 = pd.crosstab(s.source, s.label).reset_index()
    t1.insert(1, "country", t1.source.map(s.groupby("source").country.first()))
    t1.insert(2, "splits", t1.source.map(role))
    out = ["# FundusShift v1.1: supplementary data tables", "", "Linear heads: logistic regression, C = 0.05, (source, class)-balanced weights, features standardised on train.",
           "95 % CIs resample leakage groups (500 draws).", "", "## Table S1. Sources", "", md(t1), "",
           "## Table S2. Split protocol (E1), balanced accuracy", ""]
    e1 = [{"model": m, "protocol": p, **v} for m, r in sm.items() for p, v in r["e1"].items()]
    out += [md(pd.DataFrame(e1)), "", "The source-held-out row is one deterministic fit, so its std is 0.", "",
            "## Table S3. Known reuse (E2): FIVES glaucoma images with a GRAPE copy", ""]
    out += [md(pd.concat([pd.read_csv(f) for f in sorted(ex.glob("*/e2_leaky.csv"))])), "", "## Table S4. Source identifiability (E3)", ""]
    out += [md(pd.DataFrame([{"model": m, **r["e3"]} for m, r in sm.items()])), ""]
    for tag, title, f in [("S5", "Leave-region-out (E4)", "e4_leave_region_out"), ("S6", "Removing Bangladesh (E5)", "e5_label_noise"), ("S7", "Domain distance (E6)", "e6_domain_distance")]:
        d = pd.concat([pd.read_csv(ex / m / f"{f}.csv").assign(model=m) for m in sm])
        out += [f"## Table {tag}. {title}", "", md(d), ""]
    out += [md(pd.DataFrame([{"model": m, "spearman": r["e6_spearman_distance_vs_bacc"], "p": r["e6_spearman_p"], "n_sources": r["e6_n_sources"]} for m, r in sm.items()])), "",
            "## Table S8. Training-set scaling (E7), mean balanced accuracy over 3 seeds", ""]
    e7 = pd.concat([pd.read_csv(ex / m / "e7_scaling.csv").assign(model=m) for m in sm])
    out += [md(e7.groupby(["model", "fraction", "split"]).bacc.mean().unstack().reset_index()), ""]
    for tag, title, f in [("S9", "Same held-out images: pooled out-of-fold vs held-out (A1)", "a1_same_images_per_source"), ("S10", "Confusion and recall (A2)", "a2_confusion"),
                          ("S11", "D recall per DR grade; -1 = no grade published (A2)", "a2_dr_grade"), ("S12", "Separating Bangladesh effects (A3)", "a3_bangladesh"),
                          ("S13", "Leave one development source out (A4)", "a4_leave_one_source")]:
        fs = sorted(ex.glob(f"*/{f}.csv"))
        out += [f"## Table {tag}. {title}", "", md(pd.concat([pd.read_csv(x).assign(model=x.parent.name) for x in fs])), ""]
    gaps = [{"model": x.parent.name, **json.loads(x.read_text())} for x in sorted(ex.glob("*/a1_same_images.json"))]
    out += ["A1 overall (gap CI from 500 group resamples):", "", md(pd.DataFrame(gaps).astype({"gap_ci": str})), "",
            "## Table S14. Head on 12 trivial acquisition statistics (A5)", "", md(pd.read_csv(ex / "a5_trivial_shortcut.csv")), ""]
    Path(a.out).write_text("\n".join(out))


if __name__ == "__main__":
    main()
