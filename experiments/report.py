"""Aggregate every results.jsonl under runs/ into paper tables (mean +- std over seeds).

python -m experiments.report --runs runs --out runs/tables
Writes main_table.{csv,md,tex} (test / test_ood / open-set per model x loss x post-hoc) and calibration_effect.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLS = [("test", "bacc"), ("test", "worst_source_bacc"), ("test", "macro_f1"), ("test", "ece"), ("test", "ace"),
        ("test", "nll"), ("test", "brier"), ("test", "aurc"), ("test", "coverage_at_thr"), ("test", "risk_at_thr"),
        ("test_ood", "bacc"), ("test_ood", "ece"), ("openset", "auroc_msp"), ("validation", "bacc")]


def flatten(r, setting):
    row = {"setting": r.get("setting", setting), "model": r["model"], "loss": r["loss"], "weighting": r.get("weighting", "-"), "posthoc": r["posthoc"],
           "seed": r.get("seed", 0)}
    for s, k in COLS:
        v = r.get(s, {}).get(k) if isinstance(r.get(s), dict) else None
        row[f"{s}.{k}"] = v
    return row


def fmt(m, s):
    return f"{m:.3f}" if np.isnan(s) or s == 0 else f"{m:.3f}±{s:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = []
    for f in Path(a.runs).rglob("results.jsonl"):
        setting = {"probe_calib": "frozen probe", "zeroshot": "zero-shot", "vlm": "zero-shot MCQ"}.get(f.parent.name, "finetune")
        rows += [flatten(json.loads(l), setting) for l in f.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_runs.csv", index=False)
    keys = ["setting", "model", "loss", "weighting", "posthoc"]
    metric_cols = [f"{s}.{k}" for s, k in COLS]
    g = df.groupby(keys)[metric_cols]
    mean, std = g.mean(), g.std().fillna(0)
    n = df.groupby(keys).seed.nunique().rename("seeds")
    tab = pd.concat([n, mean.round(4)], axis=1).reset_index()
    tab.to_csv(out / "main_table.csv", index=False)
    pretty = mean.copy().astype(object)
    for c in metric_cols:
        pretty[c] = [fmt(m, s) if pd.notna(m) else "–" for m, s in zip(mean[c], std[c])]
    pretty = pd.concat([n, pretty], axis=1).reset_index()
    (out / "main_table.md").write_text(pretty.to_markdown(index=False))
    (out / "main_table.tex").write_text(pretty.to_latex(index=False, escape=True))
    # effect of post-hoc calibration per (setting, model, loss): ECE none -> ts -> fts
    eff = mean["test.ece"].unstack("posthoc").reset_index()
    eff.to_csv(out / "calibration_effect.csv", index=False)
    print(pretty.to_markdown(index=False)[:4000])


if __name__ == "__main__":
    main()
