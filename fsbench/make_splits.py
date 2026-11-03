"""Step 3: source-held-out, group-aware splits.

  test / test_ood : every core image of a held-out source (role test / ood_test)
  test_openset    : held-out images whose condition is outside the core classes (label OPENSET)
  train/val/cal   : dev-source images, split by leakage group, stratified by (source, label)
  cal -> cal_A / cal_B : disjoint halves (temperature scaling vs. acceptance thresholds)
  unused          : outside the core task, with the reason kept

python -m fsbench.make_splits --bench data/bench [--fractions 0.8 0.1 0.1 --seed 2027]
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _greedy_assign(groups, sizes, names, fracs, rng):
    """Assign groups to splits so each split's image share tracks its target fraction (largest deficit first)."""
    order = rng.permutation(len(groups))
    count = dict.fromkeys(names, 0)
    total, out = 0, {}
    for k in order:
        total += sizes[k]
        deficit = {s: f * total - count[s] for s, f in zip(names, fracs)}
        s = max(names, key=lambda x: (deficit[x], -names.index(x)))
        out[groups[k]] = s
        count[s] += sizes[k]
    return out


def _stratified(df, names, fracs, rng):
    """df: one row per image with group_id, source, label. Returns {group_id: split}."""
    g = df.groupby("group_id").agg(source=("source", lambda s: Counter(s).most_common(1)[0][0]),
                                   label=("label", lambda s: Counter(s).most_common(1)[0][0]),
                                   n=("image_id", "size"))
    out = {}
    for _, sub in g.groupby(["source", "label"]):
        out.update(_greedy_assign(list(sub.index), list(sub.n), names, fracs, rng))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "CAL"))
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args(argv)
    bench = Path(a.bench)
    rng = np.random.default_rng(a.seed)
    m = pd.read_csv(bench / "manifest.csv", dtype=str, keep_default_na=False)
    m = m.merge(pd.read_csv(bench / "groups.csv", dtype=str), on="image_id", how="left", validate="1:1")
    m["side"] = np.where(m.role.isin(["dev", "inventory"]), "dev", "test")

    reason = pd.Series("", index=m.index)
    reason[m.role == "inventory"] = "source_excluded_from_core"
    reason[(reason == "") & (m.file_missing == "True")] = "file_missing"
    reason[(reason == "") & (m.read_error != "")] = "unreadable"
    reason[(reason == "") & (m.label == "")] = "label:" + m.label_reason
    reason[(reason == "") & (m.label == "OPENSET") & (m.side == "dev")] = "label:openset_class_in_dev_source"

    # any group touching a held-out source is removed from development (test images are kept)
    test_groups = set(m.group_id[(m.side == "test") & (reason == "")])
    leak = (m.side == "dev") & m.group_id.isin(test_groups) & (reason == "")
    reason[leak] = "duplicate_of_test_side_image"
    # a group shared by test and ood_test sources stays only in test
    ood_leak = (m.role == "ood_test") & m.group_id.isin(set(m.group_id[(m.role == "test") & (reason == "")])) & (reason == "")
    reason[ood_leak] = "duplicate_of_test_image"
    # copies of the same picture must carry one harmonized label
    core = reason == ""
    nlab = m[core].groupby("dup_group_id").label.nunique()
    reason[m.dup_group_id.isin(nlab[nlab > 1].index) & core] = "duplicate_label_conflict"

    m["split"] = "unused"
    core = reason == ""
    m.loc[core & (m.role == "test"), "split"] = "test"
    m.loc[core & (m.role == "ood_test"), "split"] = "test_ood"
    m.loc[core & (m.side == "test") & (m.label == "OPENSET"), "split"] = "test_openset"
    dev = m[core & (m.side == "dev")]
    m.loc[dev.index, "split"] = dev.group_id.map(_stratified(dev, ["train", "val", "cal"], a.fractions, rng))
    cal = m[m.split == "cal"]
    m["cal_subset"] = ""
    m.loc[cal.index, "cal_subset"] = cal.group_id.map(_stratified(cal, ["cal_A", "cal_B"], [0.5, 0.5], rng))
    m["unused_reason"] = reason

    cols = ["image_id", "split", "cal_subset", "source", "family", "region", "country", "rel_path", "label", "native_label",
            "label_reason", "normal_scope", "dr_grade", "patient_id", "eye", "group_id", "dup_group_id", "sha256", "unused_reason"]
    m[cols].to_csv(bench / "splits.csv", index=False)
    for s in ["train", "val", "cal", "test", "test_ood", "test_openset"]:
        m[m.split == s][cols].to_csv(bench / f"split_{s}.csv", index=False)

    print(m.split.value_counts().to_string())
    print("\nunused reasons:\n" + m[m.split == "unused"].unused_reason.value_counts().to_string())


if __name__ == "__main__":
    main()
