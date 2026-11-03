"""Step 2: link images that must never be separated across splits.

Edges
  identical SHA-256                                             -> same picture
  vessel-signature cosine >= --cand (or top-k most similar image of another source)
      AND shift-tolerant NCC >= --verify (--verify-cross for different sources)  -> same picture (near copy)
      -> dup_group_id
  + same (source, patient_id)                                   -> group_id (leakage group)

Thresholds were calibrated on 1,500 real images with synthetic copies (rescale 0.3-0.9x, JPEG q60-95, <=1% crop,
gamma 0.8-1.25): 95% of copies reach NCC >= 0.70, while visually confirmed different eyes peaked at 0.62.

python -m fsbench.dedup --raw data/raw --bench data/bench [--cand 0.93 --verify 0.70]
Outputs: groups.csv, near_duplicates.csv (verified pairs), dup_candidates.csv (all scored candidates),
         cross_source_reuse.csv, vessel_signature_cache.npz
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .vessel import fine_map, pairs_above, signature, verify

Image.MAX_IMAGE_PIXELS = None


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _sig(path):
    try:
        with Image.open(path) as im:
            return signature(im)
    except Exception:
        return None


def _verify_star(args):
    """args: (path_i, [(j, path_j), ...]) -> [(j, ncc)]; the anchor map is computed once."""
    pi, others = args
    with Image.open(pi) as im:
        fi = fine_map(im)
    out = []
    for j, pj in others:
        with Image.open(pj) as im:
            out.append((j, verify(fi, fine_map(im))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--cand", type=float, default=0.93)
    ap.add_argument("--verify", type=float, default=0.55)
    ap.add_argument("--verify-cross", type=float, default=0.55,
                    help="cross-source pairs; 0.60-0.66 pairs between collections were all different eyes on visual review")
    ap.add_argument("--cross-topk", type=int, default=3,
                    help="also verify each image against its k most similar images from other sources")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--reuse-candidates", action="store_true",
                    help="reuse NCC scores from an earlier dup_candidates.csv (same manifest) instead of recomputing")
    a = ap.parse_args(argv)
    bench = Path(a.bench)
    m = pd.read_csv(bench / "manifest.csv", dtype=str, keep_default_na=False)
    n = len(m)
    paths = [str(Path(a.raw) / s / p) for s, p in zip(m.source, m.rel_path)]
    ok = (m.sha256 != "").to_numpy()
    uf = UnionFind(n)
    edges = {"sha256": 0, "vessel": 0, "patient": 0}

    for _, idx in m[ok].groupby("sha256").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            uf.union(idx[0], j)
            edges["sha256"] += 1

    # signatures are cached by file SHA-256, so adding a collection only computes the new images
    cache_file = bench / "vessel_signature_cache.npz"
    cache = {}
    if cache_file.exists():
        z = np.load(cache_file)
        cache = dict(zip(z["sha"].tolist(), z["sig"]))
    todo = [i for i in np.flatnonzero(ok) if m.sha256.iloc[i] not in cache]
    if todo:
        print(f"vessel signatures: {len(todo)} new, {ok.sum() - len(todo)} cached")
        with ProcessPoolExecutor(a.workers) as ex:
            for i, sg in zip(todo, ex.map(_sig, [paths[i] for i in todo], chunksize=32)):
                if sg is not None:
                    cache[m.sha256.iloc[i]] = sg
        np.savez(cache_file, sha=np.array(list(cache)), sig=np.stack(list(cache.values())))
    S = np.zeros((n, 1024), np.float16)
    for i in np.flatnonzero(ok):
        if m.sha256.iloc[i] in cache:
            S[i] = cache[m.sha256.iloc[i]]

    cand = {(i, j): c for i, j, c in pairs_above(S, a.cand) if m.sha256.iloc[i] != m.sha256.iloc[j]}
    src = m.source.to_numpy()
    Sf = S.astype(np.float32)
    for i0 in range(0, n, 2048):
        sim = Sf[i0:i0 + 2048] @ Sf.T
        sim[src[i0:i0 + 2048, None] == src[None, :]] = -1
        top = np.argsort(-sim, axis=1)[:, :a.cross_topk]
        for r, js in enumerate(top):
            i = i0 + r
            for j in js:
                if ok[i] and ok[j] and m.sha256.iloc[i] != m.sha256.iloc[j]:
                    cand.setdefault((min(i, j), max(i, j)), float(sim[r, j]))
    cand = [(i, j, c) for (i, j), c in cand.items()]
    print(f"candidates (cos >= {a.cand} or cross-source top-{a.cross_topk}): {len(cand)}", flush=True)
    if a.reuse_candidates:
        old = pd.read_csv(bench / "dup_candidates.csv", dtype={"image_id_a": str, "image_id_b": str})
        pos = dict(zip(m.image_id, range(n)))
        prev = {(min(pos[x], pos[y]), max(pos[x], pos[y])): v for x, y, v in zip(old.image_id_a, old.image_id_b, old.ncc)}
        missing = [(i, j, c) for i, j, c in cand if (i, j) not in prev]
        assert not missing, f"{len(missing)} candidates have no stored score; rerun without --reuse-candidates"
        ncc = prev
    else:
        star = {}
        for i, j, _ in cand:
            star.setdefault(i, []).append((j, paths[j]))
        ncc = {}
        with ProcessPoolExecutor(a.workers) as ex:
            for k, (i, r) in enumerate(zip(star, ex.map(_verify_star, [(paths[i], v) for i, v in star.items()], chunksize=4))):
                ncc.update({(i, j): v for j, v in r})
                if k % 2000 == 0:
                    print(f"  verified anchors {k}/{len(star)}", flush=True)
    # images the source marks as ungradable carry no eye identity (dark / featureless): on visual review every
    # cross-collection link through them was a different eye, so they are linked by identical bytes only
    no_vessel = m.native_label.str.contains("ungradable", case=False).to_numpy()

    rows = []
    for i, j, c in cand:
        v = ncc[(i, j)]
        thr = a.verify if src[i] == src[j] else a.verify_cross
        link = v >= thr and not (no_vessel[i] or no_vessel[j])
        rows.append([m.image_id.iloc[i], m.image_id.iloc[j], m.source.iloc[i], m.source.iloc[j], round(c, 4),
                     round(v, 4), m.label.iloc[i], m.label.iloc[j], link])
        if link:
            uf.union(i, j)
            edges["vessel"] += 1
    allc = pd.DataFrame(rows, columns=["image_id_a", "image_id_b", "source_a", "source_b", "sig_cos", "ncc",
                                       "label_a", "label_b", "linked"])
    allc.to_csv(bench / "dup_candidates.csv", index=False)
    # exact copies are near-duplicates too; list them so the audit checks them as pairs
    exact = []
    for _, idx in m[ok].groupby("sha256").groups.items():
        idx = list(idx)
        exact += [[m.image_id.iloc[idx[0]], m.image_id.iloc[j], m.source.iloc[idx[0]], m.source.iloc[j], 1.0, 1.0,
                   m.label.iloc[idx[0]], m.label.iloc[j], True] for j in idx[1:]]
    pairs = pd.concat([allc[allc.linked], pd.DataFrame(exact, columns=allc.columns)], ignore_index=True)
    pairs["exact"] = pairs.ncc >= 1.0
    pairs.to_csv(bench / "near_duplicates.csv", index=False)
    m["dup_group_id"] = ["d" + m.image_id.iloc[uf.find(i)] for i in range(n)]

    for _, idx in m[m.patient_id != ""].groupby(["source", "patient_id"]).groups.items():
        idx = list(idx)
        for j in idx[1:]:
            uf.union(idx[0], j)
            edges["patient"] += 1
    m["group_id"] = ["g" + m.image_id.iloc[uf.find(i)] for i in range(n)]
    m[["image_id", "dup_group_id", "group_id"]].to_csv(bench / "groups.csv", index=False)

    cross = pairs[pairs.source_a != pairs.source_b]
    rep = (cross.assign(a=[min(x, y) for x, y in zip(cross.source_a, cross.source_b)],
                        b=[max(x, y) for x, y in zip(cross.source_a, cross.source_b)],
                        disagree=cross.label_a != cross.label_b)
           .groupby(["a", "b"]).agg(pairs=("exact", "size"), exact=("exact", "sum"), label_disagree=("disagree", "sum"))
           .reset_index()) if len(cross) else pd.DataFrame(columns=["a", "b", "pairs", "exact", "label_disagree"])
    rep.to_csv(bench / "cross_source_reuse.csv", index=False)

    within = pairs[pairs.source_a == pairs.source_b].groupby("source_a").size().rename("within_source_pairs")
    dsz, gsz = m.groupby("dup_group_id").size(), m.groupby("group_id").size()
    print(f"edges: {edges}")
    print(f"duplicate groups >1: {(dsz > 1).sum()} (largest {dsz.max()}) | leakage groups: {len(gsz)} (largest {gsz.max()})")
    print(f"verified near-duplicate pairs: {len(pairs)} (cross-source {len(cross)})")
    print(within.to_string())
    if len(rep):
        print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
