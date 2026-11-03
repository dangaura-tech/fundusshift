"""Blinded packs for human review, and the agreement script to run on the returned sheets.

make  : stratified image sample + confident-learning flags + repeats for intra-rater agreement; near-threshold duplicate pairs
score : Cohen's kappa rater vs rater and rater vs benchmark label, overall and per source
python -m experiments.label_review make --bench data/bench --images data/processed --flags results/data_validation/v3_flagged_dev.csv --out results/label_review
python -m experiments.label_review score --key results/label_review/key_DO_NOT_SHARE.csv --sheets rater_a.csv rater_b.csv
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score

from .data import load_table

GRADES = "N, D, G, other_disease, ungradable"


def make(a):
    out, rng = Path(a.out), np.random.default_rng(0)
    (out / "images").mkdir(parents=True, exist_ok=True)
    t, _ = load_table(a.bench)
    t = t[t.y >= 0]
    picks = [g.sample(min(a.per_cell, len(g)), random_state=0).assign(stratum="random") for _, g in t.groupby(["source", "label"])]
    random = pd.concat(picks)
    flags = pd.read_csv(a.flags, dtype={"image_id": str}).sort_values(["votes", "p_given"], ascending=[False, True])
    flags = flags[~flags.image_id.isin(random.image_id)].head(a.flagged)
    flagged = t[t.image_id.isin(flags.image_id)].assign(stratum="confident_learning_flag")
    both = pd.concat([random, flagged])
    repeats = both.sample(a.repeats, random_state=1).assign(stratum="repeat")
    d = pd.concat([both, repeats]).sample(frac=1, random_state=2).reset_index(drop=True)
    d["review_id"] = [f"R{i:04d}" for i in range(len(d))]
    for r in d.itertuples():
        shutil.copy(Path(a.images) / r.source / f"{r.image_id}.jpg", out / "images" / f"{r.review_id}.jpg")
    d[["review_id", "image_id", "source", "split", "label", "native_label", "stratum"]].to_csv(out / "key_DO_NOT_SHARE.csv", index=False)
    pd.DataFrame({"review_id": d.review_id, "file": "images/" + d.review_id + ".jpg", "grade": "", "confidence_1_to_3": "", "comment": ""}).to_csv(out / "sheet_blank.csv", index=False)
    (out / "INSTRUCTIONS.txt").write_text(f"For each image choose one grade: {GRADES}.\nGrade independently and do not see another rater's sheet.\nN = no visible disease. D = diabetic retinopathy. G = glaucomatous optic neuropathy.\nReturn the sheet as rater_<name>.csv.\n")
    pairs = pd.read_csv(Path(a.bench) / "dup_candidates.csv", dtype={"image_id_a": str, "image_id_b": str})
    p = pd.concat([pairs[(pairs.ncc >= lo) & (pairs.ncc < hi)].sample(min(a.pairs // 3, ((pairs.ncc >= lo) & (pairs.ncc < hi)).sum()), random_state=0) for lo, hi in [(.6, .7), (.7, .8), (.8, .95)]]).reset_index(drop=True)
    have = {(r.source, r.image_id) for r in t.itertuples()}
    (out / "duplicate_pairs").mkdir(exist_ok=True)
    rows = []
    for i, r in enumerate(p.itertuples()):
        if (r.source_a, r.image_id_a) not in have or (r.source_b, r.image_id_b) not in have:
            continue
        ims = [np.asarray(Image.open(Path(a.images) / s / f"{m}.jpg").convert("RGB").resize((512, 512))) for s, m in [(r.source_a, r.image_id_a), (r.source_b, r.image_id_b)]]
        Image.fromarray(np.concatenate(ims, 1)).save(out / "duplicate_pairs" / f"P{i:03d}.jpg", quality=90)
        rows.append({"pair_id": f"P{i:03d}", "same_photo_or_same_eye": "", "ncc_hidden": r.ncc, "linked_hidden": r.linked})
    pr = pd.DataFrame(rows)
    pr.to_csv(out / "duplicate_key_DO_NOT_SHARE.csv", index=False)
    pr[["pair_id", "same_photo_or_same_eye"]].to_csv(out / "duplicate_sheet_blank.csv", index=False)
    print(d.stratum.value_counts().to_dict(), "pairs", len(pr))


def score(a):
    key = pd.read_csv(a.key, dtype=str)
    sheets = {Path(f).stem: pd.read_csv(f, dtype=str).set_index("review_id").grade for f in a.sheets}
    d = key.set_index("review_id")
    for k, v in sheets.items():
        d[k] = v
    names = list(sheets)
    main = d[d.stratum == "random"]
    rows = []
    for vw, g in [("all", main), *main.groupby("source")]:
        for n in names:
            m = g[n].notna()
            rows.append({"view": vw, "pair": f"{n} vs benchmark", "n": int(m.sum()), "kappa": cohen_kappa_score(g.label[m], g[n][m]), "agreement": float((g.label[m] == g[n][m]).mean())})
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                m = g[x].notna() & g[y].notna()
                rows.append({"view": vw, "pair": f"{x} vs {y}", "n": int(m.sum()), "kappa": cohen_kappa_score(g[x][m], g[y][m]), "agreement": float((g[x][m] == g[y][m]).mean())})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    for k in ["bench", "images", "flags", "out"]:
        m.add_argument(f"--{k}", required=True)
    m.add_argument("--per-cell", type=int, default=20)
    m.add_argument("--flagged", type=int, default=100)
    m.add_argument("--repeats", type=int, default=30)
    m.add_argument("--pairs", type=int, default=60)
    s = sub.add_parser("score")
    s.add_argument("--key", required=True)
    s.add_argument("--sheets", nargs="+", required=True)
    a = ap.parse_args(argv)
    make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    main()
