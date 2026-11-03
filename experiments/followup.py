"""Post hoc checks on the benchmark, decided after reading test results, so everything here is exploratory.

prior: label-prior adjustment per held-out source (true prior, EM of Saerens et al. 2002) on top of TS. True priors use test labels: a check, not a method.
paired: differences recomputed per source, leave-one-source-out, source + seed bootstrap; same-backbone objective comparisons.
target: recalibration with K labelled images of the new source, read on the rest; T from each development source.
curve: linear probes against labels per class and weight decay. Same shots rule as probe_calib: 300 epochs, one image per leakage group.
loso: pick probe configurations by holding out each development source against ordinary validation.
nll: NLL against bAcc checkpoint rule inside the same HPC runs, and HPC against Mac.
extras: entropy and margin as selection scores, D recall per DR grade, validation-test rank correlation, ensembles.
referral: KNN and Mahalanobis as the referral rule, thresholds frozen on Cal-B.
degrade: controlled blur, JPEG, brightness and contrast on test images, thresholds and T frozen from clean calibration data.
gate: referral per source, and MSP and KNN required together, thresholds frozen on Cal-B.
mondrian: LAC sets with one threshold against one per class, per class and source.
curve_source: labels per class against pooled bAcc, source + seed bootstrap of the difference to the full train set.
degrade_pred: confusion matrices under the harshest corruptions.
python -m experiments.followup prior --bench data/bench --images data/processed --runs runs --features runs/features --ckpt runs --views results/full_eval/views.csv --out results/followup
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.metrics import balanced_accuracy_score

from .backbones import GATED, build_vm
from .calib import Calibrator, ece, nll, softmax
from .coop import sample_shots
from .data import balanced_weights, load_table
from .finetune import transforms
from .full_eval import coverage_threshold, discover, load_logits, operating, posthoc_states, quantile
from .paired_tests import PAIRS, bacc
from .paired_tests import ece as w_ece
from .probe_calib import train_head
from .reliability_extras import rep_files
from .robust_train import build

KEY = ["16-shot-probe", "linear-probe", "zero-shot", "erm", "groupdro", "finetune"]


def load(a):
    t, classes = load_table(a.bench)
    reps, files = rep_files(a.runs, a.views)
    return t, classes, reps, files


def ts(Z, y, ca):
    c = Calibrator("ts").fit(Z[ca], y[ca])
    return c(Z), c.T


def adjust(P, source, target):
    q = P * (target + 1e-3) / (source + 1e-3)
    return q / q.sum(1, keepdims=True)


def em_prior(P, source, iters=200):
    pi = source.copy()
    for _ in range(iters):
        new = adjust(P, source, pi).mean(0)
        if abs(new - pi).max() < 1e-5:
            return new
        pi = new
    return pi


def prior(a):
    t, classes, reps, files = load(a)
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    ca, cb = (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B")
    te = np.flatnonzero(sp == "test")
    rows, per = [], []
    for setting, rs in files.items():
        for model, cfg, seed, f in rs:
            Z = load_logits(f, setting, t)
            P, _ = ts(Z, y, ca)
            base_prior = P[ca].mean(0)
            thr = coverage_threshold(P[cb].max(1), .8)
            arms = {k: P[te].copy() for k in ["ts", "true_per_source", "em_per_source"]}
            for s in np.unique(src[te]):
                m = src[te] == s
                arms["true_per_source"][m] = adjust(P[te][m], base_prior, np.bincount(y[te][m], minlength=3) / m.sum())
                arms["em_per_source"][m] = adjust(P[te][m], base_prior, em_prior(P[te][m], base_prior))
            arms["true_pooled"] = adjust(P[te], base_prior, np.bincount(y[te], minlength=3) / len(te))
            arms["em_pooled"] = adjust(P[te], base_prior, em_prior(P[te], base_prior))
            for arm, Q in arms.items():
                r = dict(setting=setting, model=model, config=cfg, seed=seed, arm=arm, bacc=balanced_accuracy_score(y[te], Q.argmax(1)), ece=ece(Q, y[te])[0], nll=nll(Q, y[te]))
                r.update({k: v for k, v in operating(Q, y[te], thr, classes).items() if k in ("coverage", "risk", "G_to_N_rate", "D_to_N_rate")})
                rows.append(r)
                for s in np.unique(src[te]):
                    m = src[te] == s
                    per.append(dict(setting=setting, seed=seed, arm=arm, source=s, ece=ece(Q[m], y[te][m])[0], nll=nll(Q[m], y[te][m])))
        print("prior", setting, flush=True)
    pd.DataFrame(rows).to_csv(a.out / "prior.csv", index=False)
    pd.DataFrame(per).to_csv(a.out / "prior_by_source.csv", index=False)


def paired(a):
    t, classes, reps, files = load(a)
    all_runs = {}
    for setting, model, cfg, seed, f in discover(a.runs):
        all_runs.setdefault((setting, model, str(cfg)), []).append((seed, f))
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    ca = (sp == "calibration") & (sub == "cal_A")
    te = sp == "test"
    yt, st = y[te], src[te]
    us = np.unique(st)
    rk = {r.setting: (r.setting, r.model, r.config) for r in reps.itertuples()}
    pairs = [("rep", rk[x], rk[z]) for x, z in PAIRS if x in rk and z in rk]
    for s in ["groupdro", "coral", "swad"]:
        pairs.append(("same_backbone", (s, "dinov2_lora", "ce"), ("erm", "dinov2_lora", "ce")))
        pairs.append(("same_backbone", (s, "convnext_t", "ce"), ("finetune", "convnext_t", "ce")))
    pairs.append(("same_backbone", ("groupdro", "dinov2_lora", "ce"), ("groupdro", "convnext_t", "ce")))
    cache = {}

    def probs(k):
        if k not in cache:
            raw, cal = {}, {}
            for seed, f in all_runs[k]:
                Z = load_logits(f, k[0], t)
                raw[seed] = softmax(Z[te].astype(np.float64))
                cal[seed] = ts(Z, y, ca)[0][te]
            cache[k] = (raw, cal)
        return cache[k]

    fns = {"bacc": lambda P, w, _: bacc(P.argmax(1), yt, w), "ece_raw": lambda P, w, _: w_ece(P, yt, w), "ece_ts": lambda _, w, Q: w_ece(Q, yt, w)}
    rng = np.random.default_rng(0)
    rows, per = [], []
    for family, A, B in pairs:
        (ra, ca_), (rb, cb_) = probs(A), probs(B)
        name = dict(family=family, a="/".join(A), b="/".join(B))
        one = np.ones(len(yt))
        for metric, fn in fns.items():
            va = {s: fn(ra[s], one, ca_[s]) for s in ra}
            vb = {s: fn(rb[s], one, cb_[s]) for s in rb}
            d = np.mean(list(va.values())) - np.mean(list(vb.values()))
            shared = sorted(set(va) & set(vb))
            seed_d = [va[s] - vb[s] for s in shared]
            bs = []
            for _ in range(1000):
                w = np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us))[np.searchsorted(us, st)].astype(float)
                sa, sb = rng.choice(list(ra), len(ra)), rng.choice(list(rb), len(rb))
                bs.append(np.mean([fn(ra[s], w, ca_[s]) for s in sa]) - np.mean([fn(rb[s], w, cb_[s]) for s in sb]))
            source_d, loso = {}, []
            for u in us:
                m = (st == u).astype(float)
                source_d[u] = np.mean([fn(ra[s], m, ca_[s]) for s in ra]) - np.mean([fn(rb[s], m, cb_[s]) for s in rb])
                loso.append(np.mean([fn(ra[s], 1 - m, ca_[s]) for s in ra]) - np.mean([fn(rb[s], 1 - m, cb_[s]) for s in rb]))
                per.append(dict(name, metric=metric, source=u, diff=source_d[u]))
            rows.append(dict(name, metric=metric, diff=d, boot_lo=np.quantile(bs, .025), boot_hi=np.quantile(bs, .975), seed_min=min(seed_d) if seed_d else np.nan, seed_max=max(seed_d) if seed_d else np.nan,
                             sources_positive=int(sum(v > 0 for v in source_d.values())), sources=len(us), loso_min=min(loso), loso_max=max(loso)))
        print("paired", name["a"], name["b"], flush=True)
        pd.DataFrame(rows).to_csv(a.out / "paired_by_source.csv", index=False)
    pd.DataFrame(per).to_csv(a.out / "paired_per_source.csv", index=False)


def target(a, repeats=50):
    t, classes, reps, files = load(a)
    sp, y, src, sub, groups = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy(), t.group_id.to_numpy()
    ca, cb = (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B")
    rows = []
    for setting in KEY:
        for model, cfg, seed, f in files[setting]:
            Z = load_logits(f, setting, t)
            dev = Calibrator("ts").fit(Z[ca], y[ca])
            Ts = [Calibrator("ts").fit(Z[ca & (src == s)], y[ca & (src == s)]).T for s in np.unique(src[ca])]
            P = dev(Z)
            q0, thr0 = quantile(1 - P[cb, y[cb]], .10), coverage_threshold(P[cb].max(1), .8)
            rng = np.random.default_rng(seed)
            for s in np.unique(src[(sp == "test") | (sp == "test_ood")]):
                idx = np.flatnonzero(((sp == "test") | (sp == "test_ood")) & (src == s))
                ug = np.unique(groups[idx])

                def read(Q, q, thr, ev):
                    op = operating(Q[ev], y[ev], thr, classes)
                    return dict(ece=ece(Q[ev], y[ev])[0], nll=nll(Q[ev], y[ev]), lac_cov=float(((1 - Q[ev, y[ev]]) <= q).mean()), cov80=op["coverage"], risk80=op["risk"])

                for k in [10, 25, 50, 100]:
                    if len(idx) < 2 * k:
                        continue
                    for rep in range(repeats):
                        chosen, n = [], 0
                        for g in rng.permutation(ug):
                            chosen.append(g)
                            n += (groups[idx] == g).sum()
                            if n >= k:
                                break
                        m = np.isin(groups[idx], chosen)
                        K, ev = idx[m], idx[~m]
                        base = dict(setting=setting, model=model, config=cfg, seed=seed, source=s, k=k, rep=rep)
                        rows.append(dict(base, arm="frozen", **read(P, q0, thr0, ev)))
                        Tk = Calibrator("ts").fit(Z[K], y[K])
                        rows.append(dict(base, arm="T_from_K", **read(Tk(Z), q0, thr0, ev)))
                        rows.append(dict(base, arm="q_thr_from_K", **read(P, quantile(1 - P[K, y[K]], .10), coverage_threshold(P[K].max(1), .8), ev)))
                base = dict(setting=setting, model=model, config=cfg, seed=seed, source=s, k=0, rep=0)
                for name, T in [("T_median_dev_source", np.median(Ts)), ("T_max_dev_source", max(Ts)), ("T_oracle", Calibrator("ts").fit(Z[idx], y[idx]).T)]:
                    Q = softmax(Z / T)
                    rows.append(dict(base, arm=name, T=T, **read(Q, quantile(1 - Q[cb, y[cb]], .10), coverage_threshold(Q[cb].max(1), .8), idx)))
        print("target", setting, flush=True)
        pd.DataFrame(rows).to_csv(a.out / "target.csv.gz", index=False)


def score_probe(Z, t, y, ca, setting):
    sp, src = t.split.to_numpy(), t.source.to_numpy()
    te, oo, va = sp == "test", sp == "test_ood", sp == "validation"
    P, T = ts(Z, y, ca)
    pred = Z.argmax(1)
    r = dict(val_bacc=balanced_accuracy_score(y[va], pred[va]), test_bacc=balanced_accuracy_score(y[te], pred[te]), chaksu_bacc=balanced_accuracy_score(y[oo], pred[oo]), T=T,
             ece_raw=ece(softmax(Z[te]), y[te])[0], ece_ts=ece(P[te], y[te])[0], nll_ts=nll(P[te], y[te]))
    for c in range(3):
        r[f"recall_{c}"] = float((pred[te & (y == c)] == c).mean())
    return r


def curve(a):
    t, classes = load_table(a.bench)
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    tr, va, ca = sp == "train", sp == "validation", (sp == "calibration") & (sub == "cal_A")
    rows = []
    for m in ["medsiglip", "dinov2_b14", "siglip2_l16"]:
        X = np.load(Path(a.features) / f"{m}.npy").astype(np.float32)
        X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
        for wd in [1e-4, 1e-2, 1.0]:
            for k in [4, 8, 16, 32, 64, 128, 256, 512, "full_class", "full_class_source"]:
                for seed in range(5):
                    if isinstance(k, int):
                        idx = sample_shots(t, seed, k)
                        head = train_head(X[idx], y[idx], np.ones(len(idx)), X[va], y[va], "ce", seed, epochs=300, wd=wd)
                        n = len(idx)
                    else:
                        w = balanced_weights(y[tr].tolist()) if k == "full_class" else balanced_weights(zip(src[tr], y[tr]))
                        head = train_head(X[tr], y[tr], w, X[va], y[va], "ce", seed, wd=wd)
                        n = int(tr.sum())
                    with torch.no_grad():
                        Z = head(torch.tensor(X)).numpy()
                    rows.append(dict(model=m, wd=wd, k=k, seed=seed, n=n, **score_probe(Z, t, y, ca, "curve")))
            print("curve", m, wd, flush=True)
            pd.DataFrame(rows).to_csv(a.out / "curve.csv", index=False)


def loso(a):
    t, classes = load_table(a.bench)
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    tr, va, ca = sp == "train", sp == "validation", (sp == "calibration") & (sub == "cal_A")
    models = ["biomedclip", "clip_b16", "convnext_t", "dinov2_b14", "flair", "medsiglip", "resnet50", "siglip2_b16", "siglip2_l16", "vit_b16_in21k"]
    dev = np.unique(src[tr])
    rows = []
    for m in models:
        F = np.load(Path(a.features) / f"{m}.npy").astype(np.float32)
        X0 = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-6)
        for kind in ["full", "shots"]:
            for wt in (["none", "class", "class_source"] if kind == "full" else ["none"]):
                for loss in ["ce", "ls"]:
                    fold = {}
                    for s in dev:
                        keep = tr & (src != s)
                        ival = va & (src != s)
                        held = (tr | va) & (src == s)
                        X = (F - F[keep].mean(0)) / (F[keep].std(0) + 1e-6)
                        if kind == "full":
                            idx = np.flatnonzero(keep)
                            w = {"none": np.ones(len(idx)), "class": balanced_weights(y[idx].tolist()), "class_source": balanced_weights(zip(src[idx], y[idx]))}[wt]
                            head = train_head(X[idx], y[idx], w, X[ival], y[ival], loss, 0)
                        else:
                            sub_t = t.copy()
                            sub_t.loc[~keep, "split"] = "other"
                            idx = sample_shots(sub_t, 0, 16)
                            head = train_head(X[idx], y[idx], np.ones(len(idx)), X[ival], y[ival], loss, 0, epochs=300)
                        with torch.no_grad():
                            p = head(torch.tensor(X[held])).argmax(1).numpy()
                        fold[s] = balanced_accuracy_score(y[held], p)
                    for seed in range(3):
                        if kind == "full":
                            idx = np.flatnonzero(tr)
                            w = {"none": np.ones(len(idx)), "class": balanced_weights(y[idx].tolist()), "class_source": balanced_weights(zip(src[idx], y[idx]))}[wt]
                            head = train_head(X0[idx], y[idx], w, X0[va], y[va], loss, seed)
                        else:
                            idx = sample_shots(t, seed, 16)
                            head = train_head(X0[idx], y[idx], np.ones(len(idx)), X0[va], y[va], loss, seed, epochs=300)
                        with torch.no_grad():
                            Z = head(torch.tensor(X0)).numpy()
                        rows.append(dict(model=m, kind=kind, weighting=wt, loss=loss, seed=seed, loso_bacc=float(np.mean(list(fold.values()))), loso_worst=float(min(fold.values())), **{f"fold_{k}": v for k, v in fold.items()}, **score_probe(Z, t, y, ca, "loso")))
                    pd.DataFrame(rows).to_csv(a.out / "loso.csv", index=False)
                    print("loso", m, kind, wt, loss, flush=True)


def nll_rule(a):
    t, classes = load_table(a.bench)
    sp, y, sub = t.split.to_numpy(), t.y.to_numpy(), t.cal_subset.to_numpy()
    ca = (sp == "calibration") & (sub == "cal_A")
    te = sp == "test"
    rows = []
    roots = {"hpc": Path(a.runs) / "finetune_nll"}
    if a.mac and Path(a.mac).exists():
        roots["mac"] = Path(a.mac)
    for where, root in roots.items():
        for d in sorted(root.glob("*_s?")):
            if not (d / "logits.npy").exists():
                continue
            nll_pred = None
            for rule, f in [("nll", "logits.npy"), ("bacc", "logits_bacc_rule.npy")]:
                Z = np.load(d / f)
                nll_pred = Z.argmax(1) if rule == "nll" else nll_pred
                for ph, (P, _, _) in posthoc_states(Z, y, ca).items():
                    rows.append(dict(where=where, run=d.name, rule=rule, posthoc=ph, same_checkpoint=bool((Z.argmax(1) == nll_pred).all()), bacc=balanced_accuracy_score(y[te], P[te].argmax(1)), ece=ece(P[te], y[te])[0], nll=nll(P[te], y[te]), recall_G=float((P[te & (y == 2)].argmax(1) == 2).mean())))
    pd.DataFrame(rows).to_csv(a.out / "nll_rule.csv", index=False)


def extras(a):
    t, classes, reps, files = load(a)
    sp, y, sub, dr = t.split.to_numpy(), t.y.to_numpy(), t.cal_subset.to_numpy(), t.dr.to_numpy()
    ca, cb = (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B")
    te = sp == "test"
    sel, grade = [], []

    def aurc(s, err):
        o = np.argsort(-s, kind="stable")
        return float((np.cumsum(err[o]) / np.arange(1, len(s) + 1)).mean())

    for setting, rs in files.items():
        for model, cfg, seed, f in rs:
            Z = load_logits(f, setting, t)
            P, _ = ts(Z, y, ca)
            top = np.sort(P, 1)
            scores = {"MSP": P.max(1), "entropy": (P * np.log(np.clip(P, 1e-12, 1))).sum(1), "margin": top[:, -1] - top[:, -2]}
            for name, s in scores.items():
                th = coverage_threshold(s[cb], .8)
                acc = te & (s >= th)
                pred = P.argmax(1)
                sel.append(dict(setting=setting, model=model, seed=seed, score=name, coverage=float(acc.sum() / te.sum()), risk=float((pred[acc] != y[acc]).mean()) if acc.any() else np.nan, aurc=aurc(s[te], (pred[te] != y[te]).astype(float))))
            pred = P.argmax(1)
            for g in sorted(set(dr[te & (y == 1)])):
                m = te & (y == 1) & (dr == g)
                grade.append(dict(setting=setting, model=model, seed=seed, grade=int(g), n=int(m.sum()), recall_D=float((pred[m] == 1).mean()), to_N=float((pred[m] == 0).mean())))
        print("extras", setting, flush=True)
    pd.DataFrame(sel).to_csv(a.out / "selection_scores.csv", index=False)
    pd.DataFrame(grade).to_csv(a.out / "dr_grade.csv", index=False)
    v = pd.read_csv(a.views, low_memory=False)
    v["config"] = v.config.fillna("")
    m = v[v.posthoc == "none"].groupby(["setting", "model", "config", "view"]).bacc.mean().unstack("view")[["validation", "test"]].dropna().reset_index()
    rows = [dict(setting=s, n=len(g), spearman=spearmanr(g.validation, g.test).statistic) for s, g in m.groupby("setting") if len(g) > 2 and not s.endswith("+ens")]
    mm = m[~m.setting.str.endswith("+ens")]
    rows.append(dict(setting="ALL", n=len(mm), spearman=spearmanr(mm.validation, mm.test).statistic))
    pd.DataFrame(rows).to_csv(a.out / "val_test_rank.csv", index=False)


class Medsiglip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel, AutoProcessor
        path = GATED / "medsiglip"
        self.m, self.proc = AutoModel.from_pretrained(path), AutoProcessor.from_pretrained(path)


def net_for(setting, model, ckpt):
    net, c = build_vm(model, 3) if model == "swin_t" else build(model, 3)
    bad = net.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    if bad.unexpected_keys:
        raise SystemExit(f"{ckpt}: unexpected {bad.unexpected_keys[:4]}")
    return net, c


REFERRAL = [("finetune", "swin_t", "finetune/swin_t_mbls_s{}"), ("groupdro", "convnext_t", "robust/convnext_t_groupdro_s{}"), ("erm", "dinov2_lora", "robust/dinov2_lora_erm_s{}")]


def feats(net, t, images, c, keep, dev):
    from .data import ImageTable
    dl = torch.utils.data.DataLoader(ImageTable(t.iloc[keep], images, transforms(c, False)), 64, num_workers=6)
    F, L = [], []
    head = net.get_classifier()
    with torch.no_grad():
        for x, _ in dl:
            pre = net.forward_head(net.forward_features(x.to(dev)), pre_logits=True)
            F.append(pre.float().cpu())
            L.append(head(pre).float().cpu())
    return torch.cat(F).numpy(), torch.cat(L).numpy()


def referral_scores(F, y, va):
    norm = lambda m: m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    Fn = norm(F)
    bank = Fn[va]
    knn = np.concatenate([1 - np.sort(Fn[i:i + 2048] @ bank.T, 1)[:, -50] for i in range(0, len(F), 2048)])
    mu = np.stack([F[va & (y == c)].mean(0) for c in range(3)])
    prec = LedoitWolf().fit(np.concatenate([F[va & (y == c)] - mu[c] for c in range(3)])).precision_
    maha = np.min([(((F - m) @ prec) * (F - m)).sum(1) for m in mu], 0)
    return {"KNN": -knn, "Mahalanobis": -maha}


def referral(a):
    t, classes = load_table(a.bench)
    sp, y, sub = t.split.to_numpy(), t.y.to_numpy(), t.cal_subset.to_numpy()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    keep = np.flatnonzero(np.isin(sp, ["validation", "calibration", "test", "test_ood", "test_openset"]))
    yk, spk, subk = y[keep], sp[keep], sub[keep]
    va, ca, cb = spk == "validation", (spk == "calibration") & (subk == "cal_A"), (spk == "calibration") & (subk == "cal_B")
    rows = []

    def read(setting, model, seed, Z, F):
        P, _ = ts(Z, yk, ca)
        scores = {"MSP": P.max(1), **referral_scores(F, yk, va)}
        pred = P.argmax(1)
        for name, s in scores.items():
            for cov in [.8, .9]:
                th = coverage_threshold(s[cb], cov)
                for view in ["test", "test_ood"]:
                    m = spk == view
                    ok = s[m] >= th
                    r = dict(setting=setting, model=model, seed=seed, score=name, target=cov, view=view, coverage=float(ok.mean()), risk=float((pred[m][ok] != yk[m][ok]).mean()) if ok.any() else np.nan, open_accepted=float((s[spk == "test_openset"] >= th).mean()))
                    for c, cn in [(1, "D"), (2, "G")]:
                        if (yk[m] == c).any():
                            r[f"{cn}_to_N_rate"] = float((ok & (pred[m] == 0) & (yk[m] == c)).sum() / (yk[m] == c).sum())
                    rows.append(r)

    for setting, model, pat in REFERRAL:
        for seed in range(3):
            run = pat.format(seed)
            net, c = net_for(setting, model, Path(a.ckpt) / run / "best.pt")
            F, Z = feats(net.to(dev).eval(), t, a.images, c, keep, dev)
            read(setting, model, seed, Z, F)
            print("referral", run, flush=True)
            pd.DataFrame(rows).to_csv(a.out / "referral.csv", index=False)
    F = np.load(Path(a.features) / "medsiglip.npy").astype(np.float32)[keep]
    for setting, stem in [("linear-probe", "linear_probe/logits_medsiglip_class_ls_s{}.npy"), ("16-shot-probe", "lp16/logits_medsiglip_none_ce_s{}.npy")]:
        for seed in range(3):
            read(setting, "medsiglip", seed, np.load(Path(a.runs) / stem.format(seed))[keep], F)
    pd.DataFrame(rows).to_csv(a.out / "referral.csv", index=False)


def corrupt(img, kind, level, size):
    img = img.resize((size, size), Image.BICUBIC)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(level * size / 224))
    if kind == "jpeg":
        b = io.BytesIO()
        img.save(b, "JPEG", quality=int(level))
        return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(level)
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(level)
    return img


class Corrupted(torch.utils.data.Dataset):
    def __init__(self, table, images, kind, level, size, tf):
        self.t, self.root, self.kind, self.level, self.size, self.tf = table.reset_index(drop=True), Path(images), kind, level, size, tf

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        r = self.t.iloc[i]
        return self.tf(corrupt(Image.open(self.root / r.source / f"{r.image_id}.jpg").convert("RGB"), self.kind, self.level, self.size)), i


LEVELS = [("clean", 0)] + [("blur", v) for v in (1, 2, 4)] + [("jpeg", v) for v in (50, 20, 10)] + [("brightness", v) for v in (.7, .5, .3)] + [("contrast", v) for v in (.7, .5, .3)]


def degrade(a):
    t, classes = load_table(a.bench)
    sp, y, src, sub = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy(), t.cal_subset.to_numpy()
    ca, cb = (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B")
    te = np.flatnonzero(sp == "test")
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []

    def record(setting, model, seed, kind, level, Zt, Zs):
        c = Calibrator("ts").fit(Zs[ca], y[ca])
        Pc = c(Zs)
        thr, q = coverage_threshold(Pc[cb].max(1), .8), quantile(1 - Pc[cb, y[cb]], .10)
        P = c(Zt)
        yt = y[te]
        op = operating(P, yt, thr, classes)
        rows.append(dict(setting=setting, model=model, seed=seed, kind=kind, level=level, bacc=balanced_accuracy_score(yt, P.argmax(1)), ece_raw=ece(softmax(Zt), yt)[0], ece_ts=ece(P, yt)[0], nll_ts=nll(P, yt), coverage=op["coverage"], risk=op["risk"], D_to_N_rate=op.get("D_to_N_rate"), G_to_N_rate=op.get("G_to_N_rate"), lac_cov=float(((1 - P[np.arange(len(yt)), yt]) <= q).mean()), recall_G=float((P[yt == 2].argmax(1) == 2).mean())))

    for setting, model, pat in REFERRAL:
        for seed in range(3):
            run = pat.format(seed)
            net, c = net_for(setting, model, Path(a.ckpt) / run / "best.pt")
            net = net.to(dev).eval()
            Zs = np.load(Path(a.runs) / run / "logits.npy").astype(np.float32)
            size = c["input_size"][-1]
            tf = transforms(c, False)
            for kind, level in LEVELS:
                dl = torch.utils.data.DataLoader(Corrupted(t.iloc[te], a.images, kind, level, size, tf), 64, num_workers=6)
                with torch.no_grad():
                    Zt = np.concatenate([net(x.to(dev)).float().cpu().numpy() for x, _ in dl])
                record(setting, model, seed, kind, level, Zt, Zs)
            print("degrade", run, flush=True)
            pd.DataFrame(rows).to_csv(a.out / "degrade.csv", index=False)
    # MedSigLIP: heads are retrained on clean features with the original recipe and checked against the saved logits
    sg = Medsiglip().to(dev).eval()
    size = sg.proc.image_processor.size["height"] if "height" in sg.proc.image_processor.size else 448
    F = np.load(Path(a.features) / "medsiglip.npy").astype(np.float32)
    tr, va = sp == "train", sp == "validation"
    mean, std = F[tr].mean(0), F[tr].std(0) + 1e-6
    X0 = (F - mean) / std
    heads = []
    for seed in range(3):
        idx = sample_shots(t, seed, 16)
        heads.append(("16-shot-probe", seed, train_head(X0[idx], y[idx], np.ones(len(idx)), X0[va], y[va], "ce", seed, epochs=300), Path(a.runs) / f"lp16/logits_medsiglip_none_ce_s{seed}.npy"))
        heads.append(("linear-probe", seed, train_head(X0[tr], y[tr], balanced_weights(y[tr].tolist()), X0[va], y[va], "ls", seed), Path(a.runs) / f"linear_probe/logits_medsiglip_class_ls_s{seed}.npy"))
    base = []
    for setting, seed, head, f in heads:
        with torch.no_grad():
            Zc = head(torch.tensor(X0)).numpy()
        match = float((Zc[te].argmax(1) == np.load(f)[te].argmax(1)).mean())
        base.append((setting, seed, head, Zc))
        print("medsiglip head", setting, seed, "prediction match", round(match, 4), flush=True)
    tf = lambda im: sg.proc.image_processor(images=im, return_tensors="pt")["pixel_values"][0]
    for kind, level in LEVELS:
        dl = torch.utils.data.DataLoader(Corrupted(t.iloc[te], a.images, kind, level, size, tf), 32, num_workers=6)
        with torch.no_grad():
            out = []
            for x, _ in dl:
                o = sg.m.get_image_features(pixel_values=x.to(dev))
                out.append((o if torch.is_tensor(o) else o.pooler_output).float().cpu())
        Xt = torch.tensor(((torch.cat(out).numpy() - mean) / std).astype(np.float32))
        for setting, seed, head, Zc in base:
            with torch.no_grad():
                record(setting, "medsiglip", seed, kind, level, head(Xt).numpy(), Zc)
        print("degrade medsiglip", kind, level, flush=True)
        pd.DataFrame(rows).to_csv(a.out / "degrade.csv", index=False)


def saved_logits(a, pat, seed):
    f = Path(a.runs) / pat.format(seed)
    return np.load(f / "logits.npy" if f.suffix != ".npy" else f).astype(np.float32)


def gate(a):
    t, classes = load_table(a.bench)
    sp, y, sub, src = t.split.to_numpy(), t.y.to_numpy(), t.cal_subset.to_numpy(), t.source.to_numpy()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    keep = np.flatnonzero(np.isin(sp, ["validation", "calibration", "test", "test_ood", "test_openset"]))
    yk, spk, subk, srck = y[keep], sp[keep], sub[keep], src[keep]
    va, ca, cb = spk == "validation", (spk == "calibration") & (subk == "cal_A"), (spk == "calibration") & (subk == "cal_B")
    te, op = spk == "test", spk == "test_openset"
    cache = Path(a.ckpt) / "_gate_features"
    cache.mkdir(exist_ok=True)
    rows = []
    for setting, model, pat in [*REFERRAL, ("linear-probe", "medsiglip", "linear_probe/logits_medsiglip_class_ls_s{}.npy"), ("16-shot-probe", "medsiglip", "lp16/logits_medsiglip_none_ce_s{}.npy")]:
        for seed in range(3):
            Z = saved_logits(a, pat, seed)[keep]
            if model == "medsiglip":
                F = np.load(Path(a.features) / "medsiglip.npy").astype(np.float32)[keep]
            else:
                run = pat.format(seed)
                f = cache / (run.replace("/", "_") + ".npy")
                if not f.exists():
                    net, c = net_for(setting, model, Path(a.ckpt) / run / "best.pt")
                    Fn, Zn = feats(net.to(dev).eval(), t, a.images, c, keep, dev)
                    print(run, "prediction match with saved logits", round(float((Zn.argmax(1) == Z.argmax(1)).mean()), 4), flush=True)
                    np.save(f, Fn)
                F = np.load(f)
            P, _ = ts(Z, yk, ca)
            pred = P.argmax(1)
            s = {"MSP": P.max(1), "KNN": referral_scores(F, yk, va)["KNN"]}
            th = {k: coverage_threshold(v[cb], .8) for k, v in s.items()}
            accept = {k: s[k] >= th[k] for k in s}
            # joint rule: raise both quantiles together until Cal-B coverage reaches the target
            for q in np.linspace(.8, 1, 81):
                tm, tk = coverage_threshold(s["MSP"][cb], q), coverage_threshold(s["KNN"][cb], q)
                if ((s["MSP"][cb] >= tm) & (s["KNN"][cb] >= tk)).mean() >= .8:
                    break
            accept["MSP+KNN"] = (s["MSP"] >= tm) & (s["KNN"] >= tk)
            for name, ok in accept.items():
                for source in ["all", *sorted(set(srck[te]))]:
                    m = te if source == "all" else te & (srck == source)
                    r = dict(setting=setting, model=model, seed=seed, score=name, source=source, n=int(m.sum()), coverage=float(ok[m].mean()), risk=float((pred[m & ok] != yk[m & ok]).mean()) if (m & ok).any() else np.nan)
                    r["calB_coverage"] = float(ok[cb].mean())
                    r["open_accepted"] = float(ok[op if source == "all" else op & (srck == source)].mean()) if (source == "all" or (op & (srck == source)).any()) else np.nan
                    for c, cn in [(1, "D"), (2, "G")]:
                        if (yk[m] == c).any():
                            r[f"{cn}_to_N_rate"] = float((ok[m] & (pred[m] == 0) & (yk[m] == c)).sum() / (yk[m] == c).sum())
                    rows.append(r)
            print("gate", setting, seed, flush=True)
            pd.DataFrame(rows).to_csv(a.out / "gate.csv", index=False)


def mondrian(a):
    t, classes = load_table(a.bench)
    sp, y, sub, src = t.split.to_numpy(), t.y.to_numpy(), t.cal_subset.to_numpy(), t.source.to_numpy()
    ca, cb, te = (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B"), sp == "test"
    rows = []
    for setting, model, pat in [*REFERRAL, ("linear-probe", "medsiglip", "linear_probe/logits_medsiglip_class_ls_s{}.npy"), ("16-shot-probe", "medsiglip", "lp16/logits_medsiglip_none_ce_s{}.npy")]:
        for seed in range(3):
            P, _ = ts(saved_logits(a, pat, seed), y, ca)
            q0 = quantile(1 - P[cb, y[cb]], .10)
            qc = np.array([quantile(1 - P[cb & (y == c), c], .10) for c in range(3)])
            for rule, q in [("marginal", np.full(3, q0)), ("per_class", qc)]:
                S = (1 - P) <= q
                for source in ["all", *sorted(set(src[te]))]:
                    m = te if source == "all" else te & (src == source)
                    for c in range(3):
                        mc = m & (y == c)
                        if mc.any():
                            only_n = S[mc, 0] & (S[mc].sum(1) == 1)
                            rows.append(dict(setting=setting, model=model, seed=seed, rule=rule, source=source, cls=classes[c], n=int(mc.sum()), coverage=float(S[mc, c].mean()), size=float(S[mc].sum(1).mean()), only_N=float(only_n.mean()), q=float(q[c])))
    pd.DataFrame(rows).to_csv(a.out / "mondrian.csv", index=False)


def curve_source(a, n_boot=2000):
    t, classes = load_table(a.bench)
    sp, y, src = t.split.to_numpy(), t.y.to_numpy(), t.source.to_numpy()
    tr, va, te = sp == "train", sp == "validation", sp == "test"
    sources = sorted(set(src[te]))
    rows, boots = [], []
    for m in ["medsiglip", "dinov2_b14", "siglip2_l16"]:
        X = np.load(Path(a.features) / f"{m}.npy").astype(np.float32)
        X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
        counts = {}
        for k in [16, 64, 128, 512, "full_class"]:
            C = np.zeros((5, len(sources), 3, 2))    # seed, source, class, (correct, total)
            for seed in range(5):
                if isinstance(k, int):
                    idx = sample_shots(t, seed, k)
                    head = train_head(X[idx], y[idx], np.ones(len(idx)), X[va], y[va], "ce", seed, epochs=300)
                else:
                    head = train_head(X[tr], y[tr], balanced_weights(y[tr].tolist()), X[va], y[va], "ce", seed)
                with torch.no_grad():
                    pred = head(torch.tensor(X[te])).argmax(1).numpy()
                for i, s in enumerate(sources):
                    for c in range(3):
                        mk = (src[te] == s) & (y[te] == c)
                        C[seed, i, c] = (pred[mk] == c).sum(), mk.sum()
            counts[k] = C
            for i, s in enumerate(sources):
                for c in range(3):
                    if C[0, i, c, 1]:
                        rows.append(dict(model=m, k=k, source=s, cls=classes[c], n=int(C[0, i, c, 1]), recall=float(C[:, i, c, 0].mean() / C[0, i, c, 1])))
            print("curve_source", m, k, flush=True)
        rng = np.random.default_rng(0)

        def pooled(C, seeds, ids):
            c = C[seeds][:, ids].sum(1)
            with np.errstate(invalid="ignore"):
                return np.nanmean(c[..., 0] / c[..., 1], 1).mean()

        for k in [16, 64, 128, 512]:
            d = []
            for _ in range(n_boot):
                seeds, ids = rng.integers(0, 5, 5), rng.integers(0, len(sources), len(sources))
                d.append(pooled(counts[k], seeds, ids) - pooled(counts["full_class"], seeds, ids))
            boots.append(dict(model=m, k=k, diff=pooled(counts[k], np.arange(5), np.arange(len(sources))) - pooled(counts["full_class"], np.arange(5), np.arange(len(sources))), lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5)))
        pd.DataFrame(rows).to_csv(a.out / "curve_source.csv", index=False)
        pd.DataFrame(boots).to_csv(a.out / "curve_source_boot.csv", index=False)


class SiglipTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, im):
        return self.processor(images=im, return_tensors="pt")["pixel_values"][0]


def degrade_pred(a):
    t, classes = load_table(a.bench)
    sp, y = t.split.to_numpy(), t.y.to_numpy()
    te = np.flatnonzero(sp == "test")
    tr, va = sp == "train", sp == "validation"
    yt = y[te]
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    levels = [("clean", 0), ("blur", 2), ("blur", 4), ("jpeg", 20), ("jpeg", 10)]
    out_file = a.out / "degrade_pred.csv"
    rows = pd.read_csv(out_file).to_dict("records") if out_file.exists() else []
    done = {(r["setting"], r["seed"]) for r in rows}

    def record(setting, model, seed, kind, level, Zt):
        pred = Zt.argmax(1)
        for c in range(3):
            for p in range(3):
                rows.append(dict(setting=setting, model=model, seed=seed, kind=kind, level=level, true=classes[c], pred=classes[p], rate=float((pred[yt == c] == p).mean())))

    for setting, model, pat in REFERRAL:
        for seed in range(3):
            if (setting, seed) in done:
                continue
            run = pat.format(seed)
            net, c = net_for(setting, model, Path(a.ckpt) / run / "best.pt")
            net = net.to(dev).eval()
            tf = transforms(c, False)
            for kind, level in levels:
                dl = torch.utils.data.DataLoader(Corrupted(t.iloc[te], a.images, kind, level, c["input_size"][-1], tf), 64, num_workers=4)
                with torch.no_grad():
                    record(setting, model, seed, kind, level, np.concatenate([net(x.to(dev)).float().cpu().numpy() for x, _ in dl]))
            print("degrade_pred", run, flush=True)
            pd.DataFrame(rows).to_csv(a.out / "degrade_pred.csv", index=False)
    sg = Medsiglip().to(dev).eval()
    size = sg.proc.image_processor.size["height"] if "height" in sg.proc.image_processor.size else 448
    F = np.load(Path(a.features) / "medsiglip.npy").astype(np.float32)
    mean, std = F[tr].mean(0), F[tr].std(0) + 1e-6
    X0 = (F - mean) / std
    heads = []
    for seed in range(3):
        idx = sample_shots(t, seed, 16)
        heads.append(("16-shot-probe", seed, train_head(X0[idx], y[idx], np.ones(len(idx)), X0[va], y[va], "ce", seed, epochs=300)))
        heads.append(("linear-probe", seed, train_head(X0[tr], y[tr], balanced_weights(y[tr].tolist()), X0[va], y[va], "ls", seed)))
    tf = SiglipTransform(sg.proc.image_processor)
    for kind, level in levels:
        dl = torch.utils.data.DataLoader(Corrupted(t.iloc[te], a.images, kind, level, size, tf), 16, num_workers=4)
        with torch.no_grad():
            out = []
            for x, _ in dl:
                o = sg.m.get_image_features(pixel_values=x.to(dev))
                out.append((o if torch.is_tensor(o) else o.pooler_output).float().cpu())
        Xt = torch.tensor(((torch.cat(out).numpy() - mean) / std).astype(np.float32))
        for setting, seed, head in heads:
            with torch.no_grad():
                record(setting, "medsiglip", seed, kind, level, head(Xt).numpy())
        print("degrade_pred medsiglip", kind, level, flush=True)
        pd.DataFrame(rows).to_csv(a.out / "degrade_pred.csv", index=False)


TASKS = {"prior": prior, "paired": paired, "target": target, "curve": curve, "loso": loso, "nll": nll_rule, "extras": extras, "referral": referral, "degrade": degrade, "gate": gate, "mondrian": mondrian, "curve_source": curve_source, "degrade_pred": degrade_pred}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--bench", required=True)
    ap.add_argument("--images", default="")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--features", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--views", default="")
    ap.add_argument("--mac", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.out = Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    TASKS[a.task](a)


if __name__ == "__main__":
    main()
