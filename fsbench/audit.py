"""Step 4: leakage audit + count tables -> audit_report.md. Exit code 1 if any leakage check fails.

python -m fsbench.audit --bench data/bench
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from .common import CLASSES

USED = ["train", "val", "cal", "test", "test_ood", "test_openset"]


def _md(df):
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    a = ap.parse_args(argv)
    bench = Path(a.bench)
    s = pd.read_csv(bench / "splits.csv", dtype=str, keep_default_na=False)
    u = s[s.split.isin(USED)]
    pairs = pd.read_csv(bench / "near_duplicates.csv", dtype=str, keep_default_na=False)
    split_of = dict(zip(s.image_id, s.split))

    checks = []

    def check(name, bad):
        checks.append((name, len(bad) == 0, len(bad)))
        return bad

    # held-out splits are never trained on, so test / test_ood / test_openset count as one side for leakage
    u = u.assign(leak_key=u.split.where(~u.split.str.startswith("test"), "held_out"))
    split_of = {i: (k if k != "held_out" else "held_out") for i, k in zip(u.image_id, u.leak_key)}
    check("leakage group in >1 split", u.groupby("group_id").leak_key.nunique().loc[lambda x: x > 1])
    check("identical file (SHA-256) in >1 split", u[u.sha256 != ""].groupby("sha256").leak_key.nunique().loc[lambda x: x > 1])
    pp = u[u.patient_id != ""].groupby(["source", "patient_id"]).leak_key.nunique()
    check("patient in >1 split", pp[pp > 1])
    xp = pairs[[x in split_of and y in split_of and split_of[x] != split_of[y]
                for x, y in zip(pairs.image_id_a, pairs.image_id_b)]]
    check("near-duplicate pair across splits", xp)
    fam = u.assign(side=u.split.str.startswith("test")).groupby("family").side.nunique()
    check("acquisition family on both dev and test side", fam[fam > 1])
    dev_src, test_src = set(u.source[u.split.isin(["train", "val", "cal"])]), set(u.source[u.split.str.startswith("test")])
    check("source on both dev and test side", sorted(dev_src & test_src))
    check("cal image without cal_A/cal_B", u[(u.split == "cal") & (u.cal_subset == "")])

    classes = [c for c in [*CLASSES, "OPENSET"] if c in set(u.label)]
    by_split = pd.crosstab(u.split, u.label).reindex(index=USED, columns=classes, fill_value=0)
    by_split["total"] = by_split.sum(axis=1)
    by_src = pd.crosstab([u.source, u.split], u.label).reindex(columns=classes, fill_value=0)
    ncls = (by_src.groupby(level=0).sum() > 0).sum(axis=1).value_counts().sort_index()
    cal = pd.crosstab(u.cal_subset[u.split == "cal"], u.label[u.split == "cal"]).reindex(columns=classes, fill_value=0)
    unused = s[s.split == "unused"].unused_reason.value_counts().rename("images")
    dr = u[u.dr_grade != ""]
    dr_tab = pd.crosstab(dr.split, dr.dr_grade) if len(dr) else pd.DataFrame()

    lines = ["# FundusShift — audit report", "",
             f"Inventory rows: {len(s)} | used in protocol: {len(u)} | unused: {len(s) - len(u)} | "
             f"sources used: {u.source.nunique()}", "", "## Leakage checks", "",
             "| check | result | violations |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {k} |" for n, ok, k in checks]
    lines += ["", "## Images per split and class", "", _md(by_split),
              "", "## Calibration halves", "", _md(cal),
              "", "## Mapped classes per source (count of sources)", "", _md(ncls.rename("sources").rename_axis("n_classes")),
              "", "## Source x split x class", "", _md(by_src),
              "", "## Unused images by reason", "", _md(unused.rename_axis("reason"))]
    if len(dr_tab):
        lines += ["", "## Secondary task: ICDR DR grade per split", "", _md(dr_tab)]
    (bench / "audit_report.md").write_text("\n".join(lines) + "\n")

    for n, ok, k in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {n}  ({k})")
    print(f"\n{by_split.to_string()}\n\nreport -> {bench / 'audit_report.md'}")
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)


if __name__ == "__main__":
    main()
