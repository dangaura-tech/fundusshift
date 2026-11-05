"""External cohorts (Messidor-2, G1020): frozen published-split models, evaluated once. Protocol: docs/protocol_external.md.
prep     CPU  label map (Messidor-2 ICDR 0 -> N, 1-4 -> D; G1020 0 -> N, 1 -> G), crop and resize as fsbench.process_images, table.
extract  GPU  MedSigLIP last-layer mean tokens (as rptoken + mean3), pooled features, zero-shot logits for the 3 banks,
              ConvNeXt-T CE and Group DRO logits; each step is first checked on 64 benchmark rows against the stored arrays.
evaluate CPU  refit LOTUS / initial design on the published split with the original seeds (checked against results/revision/final),
              TS on Cal-A, thresholds and quantiles on Cal-B, then every metric on each external set.
python -m experiments.external prep|extract|evaluate --root ~/fundusshift
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

K, N, D, G = 3, 0, 1, 2
BANKS = ["clip_ensemble", "cupl", "descriptive_single"]


def paths(root):
    r = Path(root).expanduser()
    return dict(r=r, bench=r / "data/bench", images=r / "data/processed", ext=r / "data/external", out=r / "results/external")


def prep(a):
    import zipfile
    import pyarrow.parquet as pq
    from PIL import Image
    from fsbench.hashing import crop_fundus
    P = paths(a.root)
    img = P["ext"] / "processed"
    rows = []

    def save(im, source, iid):
        c = crop_fundus(im.convert("RGB"))
        if max(c.size) > 1024:
            s = 1024 / max(c.size)
            c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.LANCZOS)
        (img / source).mkdir(parents=True, exist_ok=True)
        c.save(img / source / f"{iid}.jpg", "JPEG", quality=95)
    for f in sorted((P["ext"] / "messidor2_hf").glob("*.parquet")):
        for rg in range(pq.ParquetFile(f).num_row_groups):
            for x in pq.ParquetFile(f).read_row_group(rg).to_pylist():
                iid = f"m2_{len([r for r in rows if r['source'] == 'messidor2']):04d}"
                save(Image.open(io.BytesIO(x["image"]["bytes"])), "messidor2", iid)
                g = int(x["label"])
                rows.append(dict(source="messidor2", image_id=iid, orig=x["image"].get("path") or "", grade=g, y=N if g == 0 else D, region="Europe", country="France"))
    with zipfile.ZipFile(P["r"] / "data/Images_splitted.zip") as z:
        for n in sorted(z.namelist()):
            if not n.lower().endswith(".jpg"):
                continue
            part, lab, fn = n.split("/")[-3:]
            iid = "g1020_" + Path(fn).stem
            save(Image.open(io.BytesIO(z.read(n))), "g1020", iid)
            rows.append(dict(source="g1020", image_id=iid, orig=n, grade=-1, y=N if lab == "0" else G, region="Europe", country="Germany"))
    t = pd.DataFrame(rows)
    t.to_csv(P["ext"] / "external.csv", index=False)
    print(t.groupby(["source", "y"]).size().to_string(), flush=True)


def extract(a):
    import torch
    from torch import nn
    from PIL import Image
    from experiments.data import ImageTable, load_table
    from experiments.followup import Medsiglip, net_for
    from experiments.finetune import transforms
    from experiments.prompts import bank
    P = paths(a.root)
    dev = "cuda"
    t, classes = load_table(P["bench"])
    e = pd.read_csv(P["ext"] / "external.csv")
    out = P["ext"] / "features"
    out.mkdir(parents=True, exist_ok=True)
    sg = Medsiglip().to(dev).eval()
    proc = sg.proc
    pre = lambda im: proc.image_processor(images=im, return_tensors="pt")["pixel_values"][0]
    R = None

    def medsig(table, root):
        nonlocal R
        dl = torch.utils.data.DataLoader(ImageTable(table, root, pre), 32, num_workers=8)
        M, F = [], []
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(dev)
                h = sg.m.vision_model(pixel_values=x, output_hidden_states=True).hidden_states[-1]
                Gd = int(h.shape[1] ** .5)
                v = nn.functional.adaptive_avg_pool2d(h.reshape(len(x), Gd, Gd, -1).permute(0, 3, 1, 2), 14).permute(0, 2, 3, 1)
                if R is None:
                    R = torch.linalg.qr(torch.randn(v.shape[-1], 384, generator=torch.Generator().manual_seed(0)))[0].to(v)
                M.append((v @ R).float().cpu().numpy().astype(np.float16).astype(np.float32).mean((1, 2)))
                o = sg.m.get_image_features(pixel_values=x)
                F.append((o if torch.is_tensor(o) else o.pooler_output).float().cpu().numpy())
        return np.concatenate(M), np.concatenate(F)

    def text_logits(F):
        Xn = F / np.linalg.norm(F, axis=1, keepdims=True)
        scale = sg.m.logit_scale.exp().item()
        Z = {}
        for b in BANKS:
            W = []
            for c in classes:
                tok = proc.tokenizer(bank(b, c), padding="max_length", truncation=True, return_tensors="pt").to(dev)
                with torch.no_grad():
                    o = sg.m.get_text_features(**tok)
                    o = (o if torch.is_tensor(o) else o.pooler_output).float()
                o = (o / o.norm(dim=1, keepdim=True)).mean(0)
                W.append((o / o.norm()).cpu().numpy())
            Z[b] = scale * Xn @ np.stack(W).T
        return Z
    chk = t.iloc[:64]
    Mc, Fc = medsig(chk, P["images"])
    Zc = text_logits(Fc)
    rep = dict(mean3_maxabs=float(np.abs(Mc - np.load(P["r"] / "data/strata/mean3_medsiglip.npy", mmap_mode="r")[:64, 2]).max()),
               pooled_maxabs=float(np.abs(Fc - np.load(P["r"] / "runs/features/medsiglip.npy", mmap_mode="r")[:64]).max()),
               **{f"zs_{b}_maxabs": float(np.abs(Zc[b] - np.load(P["r"] / f"runs/medsiglip/logits_medsiglip_{b}.npy", mmap_mode="r")[:64]).max()) for b in BANKS})
    print("check", rep, flush=True)
    Me, Fe = medsig(e, P["ext"] / "processed")
    np.save(out / "mean3_last.npy", Me)
    np.save(out / "medsiglip_pooled.npy", Fe)
    for b, z in text_logits(Fe).items():
        np.save(out / f"logits_medsiglip_{b}.npy", z)
    del sg
    torch.cuda.empty_cache()
    for name, setting, ck in [("convnext_ce", "finetune", "runs/finetune/convnext_t_ce_s{}"), ("convnext_groupdro", "groupdro", "runs/robust/convnext_t_groupdro_s{}")]:
        for s in range(3):
            d = P["r"] / ck.format(s)
            if not (d / "best.pt").exists():
                print("missing", d, flush=True)
                continue
            net, c = net_for(setting, "convnext_t", d / "best.pt")
            net = net.to(dev).eval()

            def run(table, root):
                dl = torch.utils.data.DataLoader(ImageTable(table, root, transforms(c, False)), 64, num_workers=8)
                with torch.no_grad():
                    return np.concatenate([net(x.to(dev)).float().cpu().numpy() for x, _ in dl])
            L = np.load(d / "logits.npy", mmap_mode="r")
            if L.shape[0] == len(t):
                rep[f"{name}_s{s}_maxabs"] = float(np.abs(run(chk, P["images"]) - L[:64]).max())
            else:
                rep[f"{name}_s{s}_stored_rows"] = int(L.shape[0])
            np.save(out / f"{name}_s{s}.npy", run(e, P["ext"] / "processed"))
            print(name, s, rep, flush=True)
    json.dump(rep, open(out / "checks.json", "w"), indent=1)


def evaluate(a):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from experiments.calib import Calibrator, ece
    from experiments.data import load_table
    from experiments.full_eval import coverage_threshold, quantile
    from experiments.revision_runs import ARMS
    from experiments.scope import pipeline
    P = paths(a.root)
    t, _ = load_table(P["bench"])
    e = pd.read_csv(P["ext"] / "external.csv")
    if "duplicate" in e:  # near copies of a benchmark image are kept in the arrays but not scored
        e.loc[e.duplicate.fillna(False).astype(bool), "source"] = e.source + "_dup"
    fe = P["ext"] / "features"
    n0 = len(t)
    M = np.concatenate([np.load(P["r"] / "data/strata/mean3_medsiglip.npy")[:, 2].astype(np.float32), np.load(fe / "mean3_last.npy").astype(np.float32)])
    Tb = [np.concatenate([np.load(P["r"] / f"runs/medsiglip/logits_medsiglip_{b}.npy"), np.load(fe / f"logits_medsiglip_{b}.npy")]).astype(np.float32) for b in BANKS]
    y = np.concatenate([t.y.to_numpy(), e.y.to_numpy()])
    src = np.concatenate([t.source.to_numpy(), e.source.to_numpy()]).astype(object)
    sp = np.concatenate([t.split.to_numpy(), np.full(len(e), "external")]).astype(object)
    sub = np.concatenate([t.cal_subset.to_numpy().astype(object), np.full(len(e), "", dtype=object)])
    scope = np.concatenate([t.normal_scope.to_numpy(), np.where(e.source == "messidor2", "dr_only", "full")]).astype(object)
    tr, va = sp == "train", sp == "validation"
    X = (M - M[tr].mean(0)) / (M[tr].std(0) + 1e-6)
    Z, checks = {}, {}
    for name, arm in [("LOTUS", "no_projection"), ("initial_design", "full"), ("zero_shot", "text_only")]:
        for sd in (range(3) if arm != "text_only" else [0]):
            z = pipeline(X, y, scope, src, tr, va, Tb, "exact", 1000 * sd, **ARMS[arm]).astype(np.float32)
            f = P["r"] / f"results/revision/final/{arm}_s{sd}.npy"
            if f.exists():
                checks[f"{name}_s{sd}_maxabs_vs_final"] = float(np.abs(z[:n0] - np.load(f)).max())
            Z[(name, sd)] = z
            print(name, sd, checks, flush=True)
    for name in ["convnext_ce", "convnext_groupdro"]:
        for sd in range(3):
            f = P["r"] / ("runs/finetune" if name == "convnext_ce" else "runs/robust") / f"convnext_t_{name.split('_')[1]}_s{sd}" / "logits.npy"
            g = fe / f"{name}_s{sd}.npy"
            if f.exists() and g.exists() and np.load(f, mmap_mode="r").shape[0] == n0:
                Z[(name, sd)] = np.concatenate([np.load(f), np.load(g)]).astype(np.float32)
    ca, cb = (sp == "calibration") & (sub == "cal_A"), (sp == "calibration") & (sub == "cal_B")
    cbi = np.flatnonzero(cb)
    half = np.random.default_rng(0).integers(0, 2, len(cbi)).astype(bool)
    b1 = cbi[half]
    rows, probs = [], {}
    for (name, sd), z in Z.items():
        cal = Calibrator("ts").fit(z[ca], y[ca])
        Pp = cal(z)
        probs[(name, sd)] = Pp
        conf, pred = Pp.max(1), Pp.argmax(1)
        thr, q = coverage_threshold(conf[b1], .8), quantile(1 - Pp[b1, y[b1]], .10)
        spec_thr = {(c, s): float(np.quantile(Pp[cbi][y[cbi] != c, c], s)) for c in (D, G) for s in (.90, .95)}
        for s in ["messidor2", "g1020", "published_test"]:
            m = (src == s) if s != "published_test" else (sp == "test")
            ym, Pm, pm = y[m], Pp[m], pred[m]
            pos = D if s == "messidor2" else G
            r = dict(model=name, seed=sd, set=s, n=int(m.sum()), T=cal.T, bacc=balanced_accuracy_score(ym, pm),
                     ece_ts=ece(Pm, ym)[0], nll_ts=float(-np.log(Pm[np.arange(len(ym)), ym].clip(1e-12)).mean()))
            for c, cn in enumerate("NDG"):
                r[f"recall_{cn}"] = float((pm[ym == c] == c).mean()) if (ym == c).any() else np.nan
                r[f"pred_rate_{cn}"] = float((pm == c).mean())
            ok = conf[m] >= thr
            r.update(cov80=float(ok.mean()), risk80=float((pm[ok] != ym[ok]).mean()) if ok.any() else np.nan,
                     lac90=float(((1 - Pm[np.arange(len(ym)), ym]) <= q).mean()))
            if s != "published_test":
                r["auroc_pos_vs_N"] = float(roc_auc_score(ym == pos, 1 - Pm[:, N]))
                r["auroc_pos_score"] = float(roc_auc_score(ym == pos, Pm[:, pos]))
                for sv in (.90, .95):
                    th = spec_thr[(pos, sv)]
                    r[f"spec@calB{int(sv * 100)}"] = float((Pm[ym != pos, pos] < th).mean())
                    r[f"sens@calB{int(sv * 100)}"] = float((Pm[ym == pos, pos] >= th).mean())
                    r[f"spec@calB{int(sv * 100)}_calB"] = float((Pp[cbi][y[cbi] != pos, pos] < th).mean())
                if s == "messidor2":
                    gr = e.grade.to_numpy()[m[n0:]]
                    r["auroc_referable"] = float(roc_auc_score(gr >= 2, 1 - Pm[:, N]))
                    for g in range(5):
                        r[f"recallD_grade{g}" if g else "recallN_grade0"] = float((pm[gr == g] == (D if g else N)).mean())
            rows.append(r)
    out = P["out"]
    out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "external_runs.csv", index=False)
    json.dump(checks, open(out / "refit_checks.json", "w"), indent=1)
    s = d.groupby(["set", "model"]).agg(["mean", "std"]).drop(columns=["seed"], level=0)
    s.to_csv(out / "external_summary.csv")
    # paired bootstrap of bAcc: LOTUS (mean probability over seeds) minus each comparator, resampled within each external set
    rng = np.random.default_rng(0)
    avg = {n: np.mean([p for (k, _), p in probs.items() if k == n], 0) for n in {k for k, _ in probs}}
    bt = []
    for s in ["messidor2", "g1020"]:
        idx = np.flatnonzero(src == s)
        for n in avg:
            if n == "LOTUS":
                continue
            diffs = []
            for _ in range(2000):
                b = rng.choice(idx, len(idx))
                diffs.append(balanced_accuracy_score(y[b], avg["LOTUS"][b].argmax(1)) - balanced_accuracy_score(y[b], avg[n][b].argmax(1)))
            pt = balanced_accuracy_score(y[idx], avg["LOTUS"][idx].argmax(1)) - balanced_accuracy_score(y[idx], avg[n][idx].argmax(1))
            bt.append(dict(set=s, comparator=n, diff=pt, lo=float(np.quantile(diffs, .025)), hi=float(np.quantile(diffs, .975)), p=float(2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0)))))
    pd.DataFrame(bt).to_csv(out / "external_bootstrap.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["n", "bacc", "recall_N", "recall_D", "recall_G", "pred_rate_G", "pred_rate_D", "ece_ts", "auroc_pos_vs_N", "auroc_referable", "cov80", "risk80", "lac90", "spec@calB90", "sens@calB90", "spec@calB95", "sens@calB95"]
    print(d.groupby(["set", "model"])[[c for c in cols if c in d]].mean().round(3).to_string())
    print(pd.DataFrame(bt).round(3).to_string())
    print(checks)


def dedup(a):
    """Nearest benchmark image of every external image by pooled MedSigLIP cosine; candidates above 0.95 are checked with pHash."""
    from PIL import Image
    from fsbench.hashing import phash
    P = paths(a.root)
    from experiments.data import load_table
    t, _ = load_table(P["bench"])  # the rows the feature arrays are aligned to
    e = pd.read_csv(P["ext"] / "external.csv")
    B = np.load(P["r"] / "runs/features/medsiglip.npy").astype(np.float32)
    E = np.load(P["ext"] / "features/medsiglip_pooled.npy").astype(np.float32)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ B.T
    j = S.argmax(1)
    e["nn_cos"], e["nn_source"], e["nn_image"] = S.max(1), t.source.to_numpy()[j], t.image_id.to_numpy()[j]
    ham = []
    for i, r in e.iterrows():
        if r.nn_cos < .95:
            ham.append(np.nan)
            continue
        a_ = phash(Image.open(P["ext"] / "processed" / r.source / f"{r.image_id}.jpg"))
        b_ = phash(Image.open(P["images"] / r.nn_source / f"{r.nn_image}.jpg"))
        ham.append(float(bin(int(a_) ^ int(b_)).count("1")))
    e["nn_phash_dist"] = ham
    e["duplicate"] = e.nn_phash_dist <= 6
    e.to_csv(P["ext"] / "external.csv", index=False)
    print(e.groupby("source")[["nn_cos"]].describe().round(3).to_string(), "\nduplicates:", e.groupby("source").duplicate.sum().to_dict(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "extract", "dedup", "evaluate"])
    ap.add_argument("--root", default="~/fundusshift")
    a = ap.parse_args()
    dict(prep=prep, extract=extract, dedup=dedup, evaluate=evaluate)[a.mode](a)
