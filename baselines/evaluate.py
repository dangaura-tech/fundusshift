"""FundusShift evaluation protocol (model-agnostic).

Input: per-split prediction files (parquet/csv) with columns image_id, logit_N, logit_D, logit_G.
  1. temperature T fitted on calibration cal_A (NLL)
  2. referral threshold on max-probability chosen on cal_B for a target accepted-risk (default 5% error)
  3. report on test / test_ood: accuracy, balanced accuracy, macro-F1, ECE, per-source balanced accuracy,
     worst-source balanced accuracy, coverage and risk at the cal_B threshold, AURC
  4. test_openset: AUROC of (1 - max prob) for separating out-of-label-space images from core test images

python baselines/evaluate.py --preds runs/resnet50 --repo le-dang1haha/FundusShift
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

CANON = ["N", "D", "G", "A", "M"]
META = ["image_id", "label", "source", "cal_subset"]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    nll = lambda t: -np.log(softmax(logits / t)[np.arange(len(y)), y] + 1e-12).mean()
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def threshold_for_risk(p, y, target):
    """Smallest max-prob threshold whose accepted-set error <= target (falls back to accept-all)."""
    conf, err = p.max(1), p.argmax(1) != y
    order = np.argsort(-conf)
    cum_risk = np.cumsum(err[order]) / np.arange(1, len(y) + 1)
    ok = np.flatnonzero(cum_risk <= target)
    return float(conf[order][ok.max()]) if len(ok) else 1.0


def aurc(p, y):
    conf, err = p.max(1), (p.argmax(1) != y).astype(float)
    order = np.argsort(-conf)
    return float((np.cumsum(err[order]) / np.arange(1, len(y) + 1)).mean())


def report(p, y, src, thr):
    pred = p.argmax(1)
    per_src = {s: balanced_accuracy_score(y[src == s], pred[src == s]) for s in np.unique(src)}
    acc = p.max(1) >= thr
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"), "ece": ece(p, y), "aurc": aurc(p, y),
            "coverage_at_thr": float(acc.mean()), "risk_at_thr": float((pred[acc] != y[acc]).mean()) if acc.any() else None,
            "worst_source_bacc": min(per_src.values()), "per_source_bacc": per_src}


def load_preds(d, split, meta):
    f = Path(d) / f"{split}.csv"
    if not f.exists():
        return None
    return meta[split].merge(pd.read_csv(f, dtype={"image_id": str}), on="image_id", validate="1:1")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--repo", default="le-dang1haha/FundusShift")
    ap.add_argument("--target-risk", type=float, default=0.05)
    ap.add_argument("--splits-csv", help="local mode: metadata/splits_final.csv instead of the Hub")
    a = ap.parse_args(argv)
    head = pd.read_csv(Path(a.preds) / "calibration.csv", nrows=1)
    CLASSES = [c[len("logit_"):] for c in head.columns if c.startswith("logit_")]
    if a.splits_csv:
        d = pd.read_csv(a.splits_csv, dtype=str, keep_default_na=False)
        d["split"] = d.split.replace({"val": "validation", "cal": "calibration"})
        d["label"] = [CLASSES.index(l) if l in CLASSES else -1 for l in d.label]
        meta = {s: g[META].reset_index(drop=True) for s, g in d.groupby("split")}
    else:
        from datasets import load_dataset
        ds = load_dataset(a.repo)
        meta = {s: ds[s].remove_columns("image").select_columns(META).to_pandas() for s in ds}
    L = [f"logit_{c}" for c in CLASSES]

    cal = load_preds(a.preds, "calibration", meta)
    ca, cb = cal[cal.cal_subset == "cal_A"], cal[cal.cal_subset == "cal_B"]
    T = fit_temperature(ca[L].to_numpy(), ca.label.to_numpy())
    thr = threshold_for_risk(softmax(cb[L].to_numpy() / T), cb.label.to_numpy(), a.target_risk)
    out = {"temperature": T, "threshold_maxprob": thr, "target_risk": a.target_risk}
    for split in ["validation", "test", "test_ood"]:
        d = load_preds(a.preds, split, meta)
        if d is not None:
            p = softmax(d[L].to_numpy() / T)
            out[split] = report(p, d.label.to_numpy(), d.source.to_numpy(), thr)
            out[split]["ece_uncalibrated"] = ece(softmax(d[L].to_numpy()), d.label.to_numpy())
    o, t = load_preds(a.preds, "test_openset", meta), load_preds(a.preds, "test", meta)
    if o is not None and t is not None:
        po, pt = softmax(o[L].to_numpy() / T), softmax(t[L].to_numpy() / T)
        score = np.r_[1 - pt.max(1), 1 - po.max(1)]
        out["test_openset"] = {"auroc_msp": roc_auc_score(np.r_[np.zeros(len(pt)), np.ones(len(po))], score),
                               "fraction_referred_at_thr": float((po.max(1) < thr).mean())}
    print(json.dumps(out, indent=2, default=float))
    (Path(a.preds) / "metrics.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
