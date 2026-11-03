"""Prints the effective preprocessing and regularisation of every backbone, for the paper's methods table.
python -m experiments.recipe_audit --out runs/recipe_audit.csv
"""
import argparse

import pandas as pd
import timm

from .backbones import VM


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = []
    for name, (tid, data, ft) in VM.items():
        m = timm.create_model(tid, pretrained=False, num_classes=3, **({"global_pool": "avg"} if name == "retfound_mae" else {}))
        c = timm.data.resolve_model_data_config(m)
        rows.append({"model": name, "timm_id": tid, "pretrain": data, "finetuned": ft, "input": c["input_size"][-1], "crop_pct": c["crop_pct"],
                     "interpolation": c["interpolation"], "mean": tuple(round(x, 3) for x in c["mean"]),
                     "dropout": sum(isinstance(x, __import__("torch").nn.Dropout) and x.p > 0 for x in m.modules()),
                     "drop_path": sum(type(x).__name__ == "DropPath" and x.drop_prob > 0 for x in m.modules()),
                     "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 1)})
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(d.drop(columns=["timm_id", "pretrain"]).to_string(index=False))


if __name__ == "__main__":
    main()
