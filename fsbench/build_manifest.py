"""Step 1: adapters -> harmonized labels -> fingerprints -> manifest.csv

python -m fsbench.build_manifest --raw data/raw --out data/bench [--sources configs/sources.yaml] [--workers 8]
Expected raw layout: <raw>/<source_id>/... exactly as released (see adapter config of each source).
"""
import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .adapters import run_adapter
from .common import USED_ROLES, harmonize, load_sources, project_root
from .hashing import fingerprint

FP_COLS = ["sha256", "phash", "dhash", "width", "height", "read_error"]


def _fingerprints(paths, cache_file, workers):
    cache = pd.read_csv(cache_file, dtype=str).fillna("") if cache_file.exists() else pd.DataFrame()
    known = {}
    if len(cache):
        for r in cache.to_dict("records"):
            known[(r["path"], r["size"], r["mtime"])] = r
    todo, out = [], {}
    for p in paths:
        st = os.stat(p)
        key = (p, str(st.st_size), str(int(st.st_mtime)))
        if key in known:
            out[p] = known[key]
        else:
            todo.append((p, key))
    if todo:
        print(f"  fingerprinting {len(todo)} new files ({len(out)} cached) ...")
        with ProcessPoolExecutor(workers) as ex:
            for (p, key), fp in zip(todo, ex.map(fingerprint, [t[0] for t in todo], chunksize=64)):
                out[p] = {"path": key[0], "size": key[1], "mtime": key[2], **fp}
        pd.DataFrame(out.values()).to_csv(cache_file, index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(project_root() / "configs/sources.yaml"))
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for s in load_sources(a.sources):
        if s.get("role") not in USED_ROLES:
            continue
        if not (raw / s["id"]).is_dir():
            print(f"[skip] {s['id']:<16} not downloaded ({raw / s['id']})")
            continue
        df = run_adapter(s, raw)
        df["file_missing"] = [not (raw / s["id"] / p).is_file() for p in df.rel_path]
        lab = [harmonize(nl.split("|") if nl else [], s) for nl in df.native_label]
        df["label"] = [l or "" for l, _ in lab]
        df["label_reason"] = [r for _, r in lab]
        for k in ("family", "region", "role", "country", "device"):
            df[k] = s.get(k, "")
        df["normal_scope"] = s.get("normal_scope", "full")
        df["source"] = s["id"]
        frames.append(df)
        print(f"[ok]   {s['id']:<16} rows={len(df):>7} missing_files={int(df.file_missing.sum()):>5} "
              f"labels={df[df.label != ''].label.value_counts().to_dict()}")
    if not frames:
        raise SystemExit("no sources found under --raw")
    m = pd.concat(frames, ignore_index=True)
    dup_rows = m.duplicated(["source", "rel_path"], keep="first")
    if dup_rows.any():
        print(f"  dropping {int(dup_rows.sum())} rows listed twice in the same source annotation")
        m = m[~dup_rows].copy()
    m["image_id"] = [hashlib.sha1(f"{s}/{p}".encode()).hexdigest()[:16] for s, p in zip(m.source, m.rel_path)]

    present = m[~m.file_missing]
    fps = _fingerprints([str(raw / s / p) for s, p in zip(present.source, present.rel_path)],
                        out / "fingerprint_cache.csv", a.workers)
    for c in FP_COLS:
        m[c] = [fps.get(str(raw / s / p), {}).get(c, "") if not miss else ""
                for s, p, miss in zip(m.source, m.rel_path, m.file_missing)]

    cols = ["image_id", "source", "family", "region", "role", "country", "device", "rel_path", "native_label", "label",
            "label_reason", "normal_scope", "dr_grade", "patient_id", "eye", *FP_COLS, "file_missing"]
    m[cols].to_csv(out / "manifest.csv", index=False)
    print(f"manifest: {len(m)} rows -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
