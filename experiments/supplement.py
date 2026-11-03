"""Reviewer checks added after reading test results, so everything here is post hoc and exploratory.

baseline: constant and prior-only predictors, the floor for every metric.
discrim: pairwise AUROC per source and sensitivity at 90 % specificity, threshold frozen on Cal-B against oracle on the target.
ceiling: logistic regression cross-validated inside each target source against the same model trained on the development sources.
rotation: random 7/7 splits of the sources into development and test, probes only, including internal against external reliability.
tta: prediction-time BatchNorm, TENT and T3A, adapting per held-out source, T frozen from clean calibration data.
preprocess: CLAHE and Ben Graham runs against the unprocessed runs of the same seed on the same hardware.
python -m experiments.supplement baseline --bench data/bench --runs runs --features runs/features --ckpt ckpt --images data/processed --views results/full_eval/views.csv --out results/supplement
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .backbones import build_vm
from .calib import Calibrator, ece, nll, softmax
from .coop import sample_shots
from .data import ImageTable, balanced_weights, load_table
from .finetune import transforms
from .full_eval import coverage_threshold, load_logits, quantile
from .probe_calib import train_head
from .reliability_extras import rep_files
from .robust_train import build

PAIRS = [(0, 1, "N_D"), (0, 2, "N_G"), (1, 2, "D_G")]
PROBE_MODELS = ["biomedclip", "clip_b16", "convnext_t", "dinov2_b14", "medsiglip", "resnet50", "siglip2_b16", "siglip2_l16", "vit_b16_in21k"]
TARGETS = ["afio_composite", "fives", "hrf", "papila", "paraguay_dr", "chaksu"]


def source_equal_bacc(y, pred, src):
    return float(np.mean([np.mean([(pred[(src == s) & (y == c)] == c).mean() for c in np.unique(y[src == s])]) for s in np.unique(src)]))


def baseline(a):
    t, classes = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    prior = np.bincount(y[sp == "train"], minlength=3) / (sp == "train").sum()
    rows = []
    for view in ["test", "test_ood"]:
        m = sp == view
        for name, p in [("uniform", np.full(3, 1 / 3)), ("train_prior", prior), *[(f"always_{c}", np.eye(3)[i]) for i, c in enumerate(classes)]]:
            P = np.tile(p, (m.sum(), 1))
            pred = P.argmax(1)
            r = dict(view=view, predictor=name, acc=float((pred == y[m]).mean()), bacc=balanced_accuracy_score(y[m], pred), bacc_source_equal=source_equal_bacc(y[m], pred, src[m]))
            if name in ("uniform", "train_prior"):
                r.update(ece=ece(P, y[m])[0], nll=nll(P, y[m]))
            rows.append(r)
    pd.DataFrame(rows).to_csv(a.out / "baseline.csv", index=False)


def discrim(a, n_boot=5000):
    t, classes = load_table(a.bench)
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    ca, cb = (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B")
    _, files = rep_files(a.runs, a.views)
    rows, screen = [], []
    for setting, runs in files.items():
        for model, cfg, seed, f in runs:
            Z = load_logits(f, setting, t).astype(np.float64)
            P = Calibrator("ts").fit(Z[ca], y[ca])(Z)
            for view, mask in [("test", sp == "test"), ("test_ood", sp == "test_ood")]:
                for s in sorted(set(src[mask])):
                    m = mask & (src == s)
                    for i, j, name in PAIRS:
                        mi, mj = m & (y == i), m & (y == j)
                        if mi.sum() >= 10 and mj.sum() >= 10:
                            sel = mi | mj
                            rows.append(dict(setting=setting, model=model, seed=seed, view=view, source=s, pair=name, auroc=roc_auc_score(y[sel] == j, Z[sel, j] - Z[sel, i]), pred_rate_j=float((P[sel].argmax(1) == j).mean())))
                    score = 1 - P[:, 0]
                    if (m & (y == 0)).sum() >= 10:
                        frozen = np.quantile(score[cb & (y == 0)], .9)
                        oracle = np.quantile(score[m & (y == 0)], .9)
                        for c, cn in [(1, "D"), (2, "G")]:
                            md = m & (y == c)
                            if md.sum() >= 10:
                                screen.append(dict(setting=setting, model=model, seed=seed, view=view, source=s, disease=cn, spec_frozen=float((score[m & (y == 0)] < frozen).mean()), sens_frozen=float((score[md] >= frozen).mean()), sens_oracle=float((score[md] >= oracle).mean())))
    rows, screen = pd.DataFrame(rows), pd.DataFrame(screen)
    rows.to_csv(a.out / "discrim_auroc.csv", index=False)
    screen.to_csv(a.out / "discrim_screen.csv", index=False)
    # within-source AUROC is free of the class mix between sources; resample sources and seeds together
    te = rows[rows.view == "test"]
    rng = np.random.default_rng(0)
    out = []
    for pair, g in te.groupby("pair"):
        cube = g.pivot_table(index=["setting", "seed"], columns="source", values="auroc")
        for x, y_ in [("16-shot-probe", "finetune"), ("16-shot-probe", "linear-probe"), ("16-shot-probe", "groupdro"), ("linear-probe", "finetune"), ("groupdro", "erm")]:
            if x not in cube.index.get_level_values(0) or y_ not in cube.index.get_level_values(0):
                continue
            A, B = cube.loc[x].to_numpy(), cube.loc[y_].to_numpy()
            k = A.shape[1]
            boots = [A[rng.integers(0, len(A), len(A))][:, ids].mean() - B[rng.integers(0, len(B), len(B))][:, ids].mean() for ids in rng.integers(0, k, (n_boot, k))]
            out.append(dict(pair=pair, a=x, b=y_, n_sources=k, auroc_a=A.mean(), auroc_b=B.mean(), diff=A.mean() - B.mean(), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5)))
    pd.DataFrame(out).to_csv(a.out / "discrim_boot.csv", index=False)


def fit_lr(X, y, C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=500, tol=1e-3).fit(X, y)


def full_proba(clf, X):
    P = np.zeros((len(X), 3))
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


def choose_C(X, y, groups, grid=(1e-3, 1e-2, 1e-1, 1.0)):
    n = min(3, *[len(np.unique(groups[y == c])) for c in np.unique(y)])
    scores = []
    for C in grid:
        s = []
        for tr, te in StratifiedGroupKFold(n, shuffle=True, random_state=0).split(X, y, groups):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            s.append(balanced_accuracy_score(y[te], fit_lr((X[tr] - mu) / sd, y[tr], C).predict((X[te] - mu) / sd)))
        scores.append(np.mean(s))
    return grid[int(np.argmax(scores))]


def within_source(X, y, groups, rep):
    P = np.zeros((len(y), 3))
    n = min(5, *[len(np.unique(groups[y == c])) for c in np.unique(y)])
    for tr, te in StratifiedGroupKFold(n, shuffle=True, random_state=rep).split(X, y, groups):
        C = choose_C(X[tr], y[tr], groups[tr])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        P[te] = full_proba(fit_lr((X[tr] - mu) / sd, y[tr], C), (X[te] - mu) / sd)
    return P


def read_target(P, y, groups, rng, n_boot=0):
    r = dict(bacc=balanced_accuracy_score(y, P.argmax(1)), ece=ece(P, y)[0], nll=nll(np.clip(P, 1e-8, 1), y))
    for i, j, name in PAIRS:
        sel = np.isin(y, [i, j])
        if len(np.unique(y[sel])) == 2:
            r[f"auroc_{name}"] = roc_auc_score(y[sel] == j, np.log(P[sel, j] + 1e-8) - np.log(P[sel, i] + 1e-8))
    return r


def ceiling(a, repeats=3):
    t, classes = load_table(a.bench)
    sp, y, src, groups, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.group_id.to_numpy(), t.cal_subset.to_numpy()
    tr, va, ca = sp == "train", sp == "validation", (sp == "calibration") & (sub == "cal_A")
    rows, boots = [], []
    rng = np.random.default_rng(0)
    for model in PROBE_MODELS:
        X = np.load(Path(a.features) / f"{model}.npy").astype(np.float32)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = (X - mu) / sd
        scores = [(C, balanced_accuracy_score(y[va], (clf := fit_lr(Xs[tr], y[tr], C)).predict(Xs[va])), clf) for C in (1e-3, 1e-2, 1e-1, 1.0)]
        C, val_bacc, clf = max(scores, key=lambda s: s[1])
        Pd = full_proba(clf, Xs)
        Zd = np.log(Pd + 1e-8)
        Pts = Calibrator("ts").fit(Zd[ca], y[ca])(Zd)
        targets = [s for s in TARGETS if (src == s).any()]
        masks = {s: (src == s) & np.isin(sp, ["test", "test_ood"]) for s in targets}
        jobs = [(s, r) for s in targets for r in range(repeats)]
        results = Parallel(n_jobs=6)(delayed(within_source)(X[masks[s]], y[masks[s]], groups[masks[s]], r) for s, r in jobs)
        oof = {}
        for (s, r), P in zip(jobs, results):
            oof.setdefault(s, []).append(P)
            rows.append(dict(model=model, source=s, repeat=r, kind="within", **read_target(P, y[masks[s]], groups[masks[s]], rng)))
        for s in targets:
            m = masks[s]
            rows.append(dict(model=model, source=s, repeat=0, kind="dev_raw", dev_val_bacc=val_bacc, **read_target(Pd[m], y[m], groups[m], rng)))
            rows.append(dict(model=model, source=s, repeat=0, kind="dev_ts", dev_val_bacc=val_bacc, **read_target(Pts[m], y[m], groups[m], rng)))
            # group bootstrap of the gap, out-of-fold predictions averaged over repeats
            Pw = np.mean(oof[s], 0)
            ug, inv = np.unique(groups[m], return_inverse=True)
            yy, gaps = y[m], []
            pw, pd_ = Pw.argmax(1), Pd[m].argmax(1)
            idx = [np.flatnonzero(inv == g) for g in range(len(ug))]
            for _ in range(1000):
                b = np.concatenate([idx[i] for i in rng.integers(0, len(ug), len(ug))])
                if len(np.unique(yy[b])) == len(np.unique(yy)):
                    gaps.append(balanced_accuracy_score(yy[b], pw[b]) - balanced_accuracy_score(yy[b], pd_[b]))
            boots.append(dict(model=model, source=s, gap=balanced_accuracy_score(yy, pw) - balanced_accuracy_score(yy, pd_), lo=np.percentile(gaps, 2.5), hi=np.percentile(gaps, 97.5)))
        print("ceiling", model, "C", C, "val", round(val_bacc, 3), flush=True)
        pd.DataFrame(rows).to_csv(a.out / "ceiling.csv", index=False)
        pd.DataFrame(boots).to_csv(a.out / "ceiling_boot.csv", index=False)


def draw_splits(t, n, seed=0):
    use = t[t.split.isin(["train", "validation", "calibration", "test"])]
    classes_of = use.groupby("source").y.agg(lambda v: set(v)).to_dict()
    sources = sorted(classes_of)
    rng = np.random.default_rng(seed)
    fixed = tuple(sorted(t[t.split == "test"].source.unique()))
    out, seen = [fixed], {fixed}
    while len(out) < n + 1:
        test = tuple(sorted(rng.choice(sources, 7, replace=False).tolist()))
        dev = [s for s in sources if s not in test]
        if test in seen or set().union(*[classes_of[s] for s in dev]) != {0, 1, 2} or set().union(*[classes_of[s] for s in test]) != {0, 1, 2}:
            continue
        seen.add(test)
        out.append(test)
    return out


def rotation(a, n_splits=50):
    t, classes = load_table(a.bench)
    keep = t.split.isin(["train", "validation", "calibration", "test"]).to_numpy()
    base_sp, y, src, groups = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.group_id.to_numpy()
    splits = draw_splits(t, n_splits)
    f = a.out / "rotation.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["split"], r["model"], r["kind"]) for r in rows}
    feats = {m: np.load(Path(a.features) / f"{m}.npy").astype(np.float32) for m in ["medsiglip", "dinov2_b14", "convnext_t", "siglip2_l16"]}
    for k, test in enumerate(splits):
        rng = np.random.default_rng(k)
        te = keep & np.isin(src, test)
        dev = np.flatnonzero(keep & ~np.isin(src, test))
        if k == 0:     # the published split, for reference
            sp = base_sp.copy()
            sub = t.cal_subset.to_numpy().astype(object).copy()
        else:
            sp = np.full(len(t), "other", dtype=object)
            sub = np.full(len(t), "", dtype=object)
            sp[te] = "test"
            ug = np.unique(groups[dev])
            u = rng.random(len(ug))
            side = dict(zip(ug, np.where(u < .8, "train", np.where(u < .9, "validation", "calibration"))))
            half = dict(zip(ug, rng.integers(0, 2, len(ug))))
            for i in dev:
                sp[i] = side[groups[i]]
                if sp[i] == "calibration":
                    sub[i] = "cal_A" if half[groups[i]] else "cal_B"
        tr, va, ca, cb = sp == "train", sp == "validation", (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B")
        tk = t.copy()
        tk["split"] = sp
        for model, F in feats.items():
            X = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-6)
            for kind in ["full_class", "16_shot"]:
                if (k, model, kind) in done:
                    continue
                if kind == "full_class":
                    head = train_head(X[tr], y[tr], balanced_weights(y[tr].tolist()), X[va], y[va], "ce", k)
                else:
                    idx = sample_shots(tk, k, 16)
                    head = train_head(X[idx], y[idx], np.ones(len(idx)), X[va], y[va], "ce", k, epochs=300)
                with torch.no_grad():
                    Z = head(torch.tensor(X)).numpy()
                P = Calibrator("ts").fit(Z[ca], y[ca])(Z)
                pred, conf = P.argmax(1), P.max(1)
                cbi = np.flatnonzero(cb)
                side = np.random.default_rng(k).integers(0, 2, len(cbi)).astype(bool)
                b1, b2 = cbi[side], cbi[~side]
                thr, q = coverage_threshold(conf[b1], .8), quantile(1 - P[b1, y[b1]], .10)
                def read(m):
                    ok = conf[m] >= thr
                    return float(ok.mean()), float((pred[m][ok] != y[m][ok]).mean()) if ok.any() else np.nan, float(((1 - P[np.flatnonzero(m), y[m]]) <= q).mean())
                mb2 = np.zeros(len(y), bool)
                mb2[b2] = True
                ci, ri, li = read(mb2)
                ce_, re_, le_ = read(te)
                r = dict(split=k, test_sources="|".join(test), model=model, kind=kind, n_train=int(tr.sum()) if kind == "full_class" else len(idx), val_bacc=balanced_accuracy_score(y[va], pred[va]), test_bacc=balanced_accuracy_score(y[te], pred[te]), test_bacc_source_equal=source_equal_bacc(y[te], pred[te], src[te]), T=Calibrator("ts").fit(Z[ca], y[ca]).T, val_ece_raw=ece(softmax(Z[va]), y[va])[0], val_ece_ts=ece(P[va], y[va])[0], ece_raw=ece(softmax(Z[te]), y[te])[0], ece_ts=ece(P[te], y[te])[0], cov_internal=ci, risk_internal=ri, lac_internal=li, cov_test=ce_, risk_test=re_, lac_test=le_)
                for c, cn in enumerate(classes):
                    r[f"recall_{cn}"] = float((pred[te & (y == c)] == c).mean())
                rows.append(r)
        print("rotation", k, test, flush=True)
        pd.DataFrame(rows).to_csv(f, index=False)


def make_net(model, ckpt):
    net, c = build_vm(model, 3) if model in ("swin_t", "resnet50") else build(model, 3)
    bad = net.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    if bad.unexpected_keys:
        raise SystemExit(f"{ckpt}: unexpected {bad.unexpected_keys[:4]}")
    return net, c


NORMS = (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.GroupNorm)


def t3a(z, W, b, K, bs=64):
    """Iwasawa & Matsuo 2021. Supports start from the head weights; the K lowest-entropy supports per class form the centroids."""
    head = lambda v: v @ W.T + b
    ent = lambda v: -(softmax(v) * np.log(softmax(v) + 1e-12)).sum(1)
    S, L, E = W.copy(), np.arange(len(W)), ent(head(W))
    out = np.zeros((len(z), 3), np.float32)
    for s in range(0, len(z), bs):
        batch = z[s:s + bs]
        p = head(batch)
        S, L, E = np.r_[S, batch], np.r_[L, p.argmax(1)], np.r_[E, ent(p)]
        keep = np.concatenate([np.flatnonzero(L == c)[np.argsort(E[L == c])[:K]] for c in range(3)])
        S, L, E = S[keep], L[keep], E[keep]
        C = np.stack([(S[L == c] / (np.linalg.norm(S[L == c], axis=1, keepdims=True) + 1e-9)).sum(0) for c in range(3)])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        out[s:s + bs] = batch @ C.T
    return out


def tta(a):
    t, classes = load_table(a.bench)
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    ca = (sp == "calibration") & (sub == "cal_A")
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    idx = np.flatnonzero(np.isin(sp, ["test", "test_ood"]))
    rows, by_source = [], []

    def record(setting, model, seed, method, Zt, T):
        for view in ["test", "test_ood"]:
            m = sp[idx] == view
            yy, ss, pred = y[idx][m], src[idx][m], Zt[m].argmax(1)
            r = dict(setting=setting, model=model, seed=seed, method=method, view=view, bacc=balanced_accuracy_score(yy, pred), bacc_source_equal=source_equal_bacc(yy, pred, ss))
            if T is not None:
                r.update(ece_raw=ece(softmax(Zt[m]), yy)[0], ece_ts=ece(softmax(Zt[m] / T), yy)[0], nll_ts=nll(softmax(Zt[m] / T), yy))
            for c, cn in enumerate(classes):
                if (yy == c).any():
                    r[f"recall_{cn}"] = float((pred[yy == c] == c).mean())
            rows.append(r)
            for s in np.unique(ss):
                by_source.append(dict(setting=setting, model=model, seed=seed, method=method, source=s, bacc=float(np.mean([(pred[(ss == s) & (yy == c)] == c).mean() for c in np.unique(yy[ss == s])]))))

    def flush():
        pd.DataFrame(rows).to_csv(a.out / "tta.csv", index=False)
        pd.DataFrame(by_source).to_csv(a.out / "tta_by_source.csv", index=False)

    RUNS = [("finetune", "swin_t", "finetune/swin_t_mbls_s{}"), ("finetune", "resnet50", "finetune/resnet50_ce_s{}"), ("groupdro", "convnext_t", "robust/convnext_t_groupdro_s{}"), ("erm", "dinov2_lora", "robust/dinov2_lora_erm_s{}")]
    for setting, model, pat in RUNS:
        for seed in range(3):
            run = pat.format(seed)
            saved = np.load(Path(a.runs) / run / "logits.npy").astype(np.float32)
            T = Calibrator("ts").fit(saved[ca], y[ca]).T
            net, c = make_net(model, Path(a.ckpt) / run / "best.pt")
            init = {k: v.clone() for k, v in net.state_dict().items()}
            net = net.to(dev)
            tf = transforms(c, False)
            has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules())
            outs = {k: np.zeros((len(idx), 3), np.float32) for k in ["none", "tent", *(["bn"] if has_bn else []), "t3a_20", "t3a_100"]}
            W = net.get_classifier().weight.detach().cpu().numpy()
            b = net.get_classifier().bias.detach().cpu().numpy()
            for s in sorted(set(src[idx])):
                loc = np.flatnonzero(src[idx] == s)
                order = np.random.default_rng(seed).permutation(loc)
                dl = lambda: torch.utils.data.DataLoader(ImageTable(t.iloc[idx[order]], a.images, tf), 64, num_workers=4)
                # unadapted pass, also the features for T3A
                net.load_state_dict(init)
                net.eval()
                F, Zn = [], []
                with torch.no_grad():
                    for x, _ in dl():
                        pre = net.forward_head(net.forward_features(x.to(dev)), pre_logits=True)
                        F.append(pre.float().cpu().numpy())
                        Zn.append(net.get_classifier()(pre).float().cpu().numpy())
                F, Zn = np.concatenate(F), np.concatenate(Zn)
                outs["none"][order] = Zn
                outs["t3a_20"][order] = t3a(F, W, b, 20)
                outs["t3a_100"][order] = t3a(F, W, b, 100)
                if has_bn:
                    net.load_state_dict(init)
                    net.eval()
                    for m in net.modules():
                        if isinstance(m, torch.nn.BatchNorm2d):
                            m.train()
                            m.track_running_stats = False
                            m.running_mean, m.running_var = None, None
                    with torch.no_grad():
                        outs["bn"][order] = np.concatenate([net(x.to(dev)).float().cpu().numpy() for x, _ in dl()])
                    net, c = make_net(model, Path(a.ckpt) / run / "best.pt")
                    net = net.to(dev)
                # TENT, Wang et al. 2021, ImageNet settings: SGD 2.5e-4, momentum .9, batch 64, online
                net.load_state_dict(init)
                net.eval()
                net.requires_grad_(False)
                params = []
                for m in net.modules():
                    if isinstance(m, NORMS):
                        if isinstance(m, torch.nn.BatchNorm2d):
                            m.train()
                            m.track_running_stats = False
                            m.running_mean, m.running_var = None, None
                        for p in m.parameters(recurse=False):
                            p.requires_grad_(True)
                            params.append(p)
                opt = torch.optim.SGD(params, lr=2.5e-4, momentum=.9)
                Zt = []
                for x, _ in dl():
                    z = net(x.to(dev))
                    loss = -(z.softmax(1) * z.log_softmax(1)).sum(1).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    Zt.append(z.detach().float().cpu().numpy())
                outs["tent"][order] = np.concatenate(Zt)
                net, c = make_net(model, Path(a.ckpt) / run / "best.pt")
                net = net.to(dev)
                print(run, s, flush=True)
            print(run, "prediction match with saved logits", round(float((outs["none"].argmax(1) == saved[idx].argmax(1)).mean()), 4), flush=True)
            for k, Zk in outs.items():
                record(setting, model, seed, k, Zk, None if k.startswith("t3a") else T)
            flush()
            del net
            torch.mps.empty_cache() if dev == "mps" else None
    # T3A on the MedSigLIP probes; heads retrained with the original recipe
    F = np.load(Path(a.features) / "medsiglip.npy").astype(np.float32)
    tr, va = sp == "train", sp == "validation"
    X = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-6)
    for seed in range(3):
        sh = sample_shots(t, seed, 16)
        heads = [("16-shot-probe", train_head(X[sh], y[sh], np.ones(len(sh)), X[va], y[va], "ce", seed, epochs=300)), ("linear-probe", train_head(X[tr], y[tr], balanced_weights(y[tr].tolist()), X[va], y[va], "ls", seed))]
        for setting, head in heads:
            W, b = head.weight.detach().numpy(), head.bias.detach().numpy()
            with torch.no_grad():
                Z = head(torch.tensor(X)).numpy()
            T = Calibrator("ts").fit(Z[ca], y[ca]).T
            outs = {"none": Z[idx], "t3a_20": np.zeros((len(idx), 3), np.float32), "t3a_100": np.zeros((len(idx), 3), np.float32)}
            for s in sorted(set(src[idx])):
                loc = np.flatnonzero(src[idx] == s)
                order = np.random.default_rng(seed).permutation(loc)
                outs["t3a_20"][order] = t3a(X[idx[order]], W, b, 20)
                outs["t3a_100"][order] = t3a(X[idx[order]], W, b, 100)
            for k, Zk in outs.items():
                record(setting, "medsiglip", seed, k, Zk, None if k.startswith("t3a") else T)
    flush()


def preprocess(a):
    t, classes = load_table(a.bench)
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    ca, va, te, oo = (sp == "calibration") & (sub == "cal_A"), sp == "validation", sp == "test", sp == "test_ood"
    rows = []
    sets = [("finetune", "convnext_t_ce", "finetune/convnext_t_ce_s{}/logits.npy", "preprocess/{}/finetune/convnext_t_ce_s{}/logits.npy"),
            ("finetune", "swin_t_mbls", "finetune/swin_t_mbls_s{}/logits.npy", "preprocess/{}/finetune/swin_t_mbls_s{}/logits.npy"),
            ("linear-probe", "medsiglip_class_ls", "linear_probe/logits_medsiglip_class_ls_s{}.npy", "preprocess/{}/linear_probe/logits_medsiglip_class_ls_s{}.npy"),
            ("16-shot-probe", "medsiglip_none_ce", "lp16/logits_medsiglip_none_ce_s{}.npy", "preprocess/{}/lp16/logits_medsiglip_none_ce_s{}.npy")]
    for setting, name, base, var in sets:
        for prep in ["none", "clahe", "graham"]:
            for seed in range(3):
                f = Path(a.runs) / (base.format(seed) if prep == "none" else var.format(prep, seed))
                if not f.exists():
                    continue
                Z = np.load(f).astype(np.float32)
                P = Calibrator("ts").fit(Z[ca], y[ca])(Z)
                pred = P.argmax(1)
                r = dict(setting=setting, config=name, prep=prep, seed=seed, val_bacc=balanced_accuracy_score(y[va], pred[va]), test_bacc=balanced_accuracy_score(y[te], pred[te]), test_bacc_source_equal=source_equal_bacc(y[te], pred[te], src[te]), chaksu_bacc=balanced_accuracy_score(y[oo], pred[oo]), ece_raw=ece(softmax(Z[te]), y[te])[0], ece_ts=ece(P[te], y[te])[0])
                for c, cn in enumerate(classes):
                    r[f"recall_{cn}"] = float((pred[te & (y == c)] == c).mean())
                rows.append(r)
    pd.DataFrame(rows).to_csv(a.out / "preprocess.csv", index=False)


TASKS = {"baseline": baseline, "discrim": discrim, "ceiling": ceiling, "rotation": rotation, "tta": tta, "preprocess": preprocess}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", default="")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--features", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--views", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.out = Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    TASKS[a.task](a)


if __name__ == "__main__":
    main()
