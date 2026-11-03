"""Linear probes on the vision-encoder features of multimodal LLMs (experiments.mllm_vision), with the benchmark's linear-probe recipe:
class-balanced label-smoothed head, best epoch by validation bAcc, TS on Cal-A, three seeds. Compared with the same models answering multiple choice.
python -m experiments.mllm_probe --bench data/bench --features runs/features --out results/review/mllm_probe.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .calib import Calibrator, ece, nll
from .data import balanced_weights, load_table
from .probe_calib import train_head

MODELS = {"mllm_fundusr1": "Fundus-R1-7B", "mllm_qwen25vl": "Qwen2.5-VL-7B-Instruct", "mllm_llava_med": "llava-med-v1.5-mistral-7b-hf", "mllm_llava_ov": "llava-onevision-qwen2-7b-ov-hf", "medsiglip": "MedSigLIP (reference)"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ["bench", "features", "out"]:
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    t, classes = load_table(a.bench)
    sp, y, sub, src = t.split.to_numpy(), t.y.to_numpy(), t.cal_subset.to_numpy(), t.source.to_numpy()
    tr, va, ca, te, ch = sp == "train", sp == "validation", (sp == "calibration") & (sub == "cal_A"), sp == "test", sp == "test_ood"
    rows = []
    for f, name in MODELS.items():
        p = Path(a.features) / f"{f}.npy"
        if not p.exists():
            print("missing", f)
            continue
        F = np.load(p).astype(np.float32)
        if not np.abs(F[tr]).sum(1).all():
            print("incomplete", f)
            continue
        X = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-6)
        for seed in range(3):
            h = train_head(X[tr], y[tr], balanced_weights(y[tr].tolist()), X[va], y[va], "ls", seed)
            with torch.no_grad():
                Z = h(torch.tensor(X)).numpy()
            c = Calibrator("ts").fit(Z[ca], y[ca])
            for view, m in [("validation", va), ("test", te), ("chaksu", ch)]:
                P = c(Z[m])
                pred = P.argmax(1)
                r = dict(model=name, seed=seed, view=view, bacc=balanced_accuracy_score(y[m], pred), ece_ts=ece(P, y[m])[0], nll_ts=nll(P, y[m]))
                for k, cn in enumerate(classes):
                    r[f"recall_{cn}"] = float((pred[y[m] == k] == k).mean()) if (y[m] == k).any() else np.nan
                if view == "test":
                    r["eq_src"] = float(np.mean([balanced_accuracy_score(y[m & (src == s)], pred[(src == s)[m]]) for s in np.unique(src[m])]))
                rows.append(r)
        print(name, flush=True)
    d = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)
    print(d.groupby(["model", "view"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())


if __name__ == "__main__":
    main()
