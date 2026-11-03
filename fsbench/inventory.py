"""Export the source registry to a flat CSV table: python -m fsbench.inventory > source_inventory.csv"""
import sys

import pandas as pd

from .common import load_sources, project_root

COLS = ["id", "name", "role", "family", "country", "device", "n_approx", "access", "license", "url",
        "normal_scope", "exclude_reason", "notes"]

if __name__ == "__main__":
    src = load_sources(sys.argv[1] if len(sys.argv) > 1 else project_root() / "configs/sources.yaml")
    df = pd.DataFrame(src).reindex(columns=COLS)
    cls = lambda s: set(s["label_map"].values()) | {v for _, v in s["label_map_regex"]}
    df["classes"] = [",".join(c for c in "NDGA" if c in cls(s)) for s in src]
    df.to_csv(sys.stdout, index=False)
