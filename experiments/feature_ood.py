"""Feature-based open-set detection for trained models: KNN and Mahalanobis on the penultimate features, against MSP, MaxLogit and Energy.

Same rules as full_eval.ood_scores: validation features are the bank, open-set images against held-out test images.
Features come from the saved checkpoint; the recomputed logits are compared with the saved ones as a check.
python -m experiments.feature_ood --bench data/bench --images data/processed --ckpt ckpt --runs runs --out results/reliability/feature_ood.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from .calib import softmax
from .data import ImageTable, load_table
from .finetune import transforms
from .robust_train import build

RUNS = [("finetune", "swin_t", "mbls", "finetune/swin_t_mbls_s{}"), ("groupdro", "convnext_t", "ce", "robust/convnext_t_groupdro_s{}"),
        ("swad", "convnext_t", "ce", "robust/convnext_t_erm_swad_s{}"), ("erm", "dinov2_lora", "ce", "robust/dinov2_lora_erm_s{}")]


def make(model):
    if model == "swin_t":
        from .backbones import build_vm
        return build_vm(model, 3)
    return build(model, 3)


@torch.no_grad()
def extract(net, dl, dev):
    F, L = [], []
    head = net.get_classifier()
    for x, _ in dl:
        pre = net.forward_head(net.forward_features(x.to(dev)), pre_logits=True)
        F.append(pre.float().cpu())
        L.append(head(pre).float().cpu())
    return torch.cat(F).numpy(), torch.cat(L).numpy()


def scores(Fv, yv, Fq, Zq):
    s = {"MSP": -softmax(Zq).max(1), "MaxLogit": -Zq.max(1), "Energy": -np.log(np.exp(Zq - Zq.max(1, keepdims=True)).sum(1)) - Zq.max(1)}
    norm = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    s["KNN"] = 1 - np.sort(norm(Fq) @ norm(Fv).T, 1)[:, -50]
    mu = np.stack([Fv[yv == c].mean(0) for c in range(3)])
    prec = LedoitWolf().fit(np.concatenate([Fv[yv == c] - mu[c] for c in range(3)])).precision_
    s["Mahalanobis"] = np.min([(((Fq - m) @ prec) * (Fq - m)).sum(1) for m in mu], 0)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    t, _ = load_table(a.bench)
    sp, y = t.split.to_numpy(), t.y.to_numpy()
    keep = np.flatnonzero(np.isin(sp, ["validation", "test", "test_openset"]))
    if a.limit:
        keep = np.concatenate([np.flatnonzero(sp == s)[:a.limit] for s in ["validation", "test", "test_openset"]])
    va, te, op = sp[keep] == "validation", sp[keep] == "test", sp[keep] == "test_openset"
    rows = []
    for setting, model, cfg, pat in RUNS:
        for seed in range(3):
            run = pat.format(seed)
            net, c = make(model)
            sd = torch.load(Path(a.ckpt) / run / "best.pt", map_location="cpu")
            bad = net.load_state_dict(sd, strict=False)
            if bad.unexpected_keys:
                raise SystemExit(f"{run}: unexpected {bad.unexpected_keys[:4]}")
            net = net.to(dev).eval()
            dl = torch.utils.data.DataLoader(ImageTable(t.iloc[keep], a.images, transforms(c, False)), 64, num_workers=6)
            F, Z = extract(net, dl, dev)
            saved = np.load(Path(a.runs) / run / "logits.npy")[keep].astype(np.float32)
            check = float((Z.argmax(1) == saved.argmax(1)).mean())
            s = scores(F[va], y[keep][va], np.r_[F[te], F[op]], np.r_[Z[te], Z[op]])
            lab = np.r_[np.zeros(te.sum()), np.ones(op.sum())]
            for name, v in s.items():
                fpr = float((v[lab == 0] >= np.quantile(v[lab == 1], .05)).mean())
                rows.append(dict(setting=setting, model=model, config=cfg, seed=seed, score=name, auroc=float(roc_auc_score(lab, v)), fpr95=fpr, pred_match_saved=check))
            print(run, "prediction match with saved logits", round(check, 4), {k: round(float(roc_auc_score(lab, v)), 3) for k, v in s.items()}, flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
            del net
            torch.mps.empty_cache() if dev == "mps" else None


if __name__ == "__main__":
    main()
