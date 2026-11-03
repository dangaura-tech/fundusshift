"""Source adapters: read each source's native annotation format into one row per image.

Every adapter returns a DataFrame with columns
    rel_path      path relative to <raw_root>/<source_id>
    native_label  native tokens joined by '|' (kept verbatim next to the harmonized label)
    patient_id    patient / exam identifier when the source provides one, else ''
    eye           left / right / ''
    dr_grade      ICDR grade 0-4 as string when available, else ''
"""
import re
from pathlib import Path

import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
COLS = ["rel_path", "native_label", "patient_id", "eye", "dr_grade"]


def _norm(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _first(row, cols):
    for c in [cols] if isinstance(cols, str) else cols:
        v = _norm(row.get(c))
        if v:
            return v
    return ""


def _need(df, cols, where):
    cols = [c for c in cols if c]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: missing columns {missing}; found {list(df.columns)[:30]}")


def _read_table(path, sep=",", header=True, names=None):
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, sep=sep, header=0 if header else None, names=names, dtype=str,
                         skipinitialspace=True, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _flatten(x):
    if not x:
        return []
    return [x] if isinstance(x, str) else list(x)


def adapter_csv(src_dir, cfg):
    rows = []
    for t in cfg["tables"]:
        df = _read_table(src_dir / t["file"], t.get("sep", ","), t.get("header", True), t.get("names"))
        label_cols = _flatten(cfg.get("label_col"))
        mh = cfg.get("multihot_cols")
        if mh is None and cfg.get("multihot_exclude_cols") is not None:
            mh = [c for c in df.columns if c not in cfg["multihot_exclude_cols"]]
        _need(df, [cfg["image_col"], cfg.get("patient_col"), cfg.get("eye_col"), *label_cols,
                   *_flatten(cfg.get("grade_col")), *(mh or []), *(cfg.get("keep_if") or {})], t["file"])
        for _, r in df.iterrows():
            if any(_norm(r[c]) != _norm(v) for c, v in (cfg.get("keep_if") or {}).items()):
                continue
            img = _norm(r[cfg["image_col"]])
            if not img:
                continue
            if label_cols:
                tokens = [_first(r, label_cols)]
            else:
                tokens = [c for c in mh if _norm(r[c]) == "1"]
                if not tokens and cfg.get("normal_if_all_zero"):
                    tokens = ["NORMAL"]
            patient = _norm(r[cfg["patient_col"]]) if cfg.get("patient_col") else ""
            if not patient and cfg.get("patient_regex"):
                m = re.search(cfg["patient_regex"], img)
                patient = m.group(1) if m else ""
            eye = _norm(r[cfg["eye_col"]]) if cfg.get("eye_col") else ""
            if not eye and cfg.get("eye_regex"):
                m = re.search(cfg["eye_regex"], img)
                eye = m.group(1) if m else ""
            rel = Path(t["image_dir"]) / cfg.get("path_template", "{image}").format(image=img, patient=patient)
            grade = _first(r, cfg["grade_col"]) if cfg.get("grade_col") else ""
            rows.append([str(rel), "|".join(tokens), patient, eye.lower(), grade])
    return pd.DataFrame(rows, columns=COLS)


def _images_under(base, pattern="**/*"):
    return sorted(p for p in base.glob(pattern) if p.is_file() and p.suffix.lower() in IMG_EXT)


def adapter_folders(src_dir, cfg):
    """Class = name of the image's parent folder (e.g. train/glaucoma/x.png -> 'glaucoma')."""
    base = src_dir / cfg.get("subdir", "")
    rows = [[str(p.relative_to(src_dir)), p.parent.name, "", "", ""] for p in _images_under(base)]
    return pd.DataFrame(rows, columns=COLS)


def adapter_filename(src_dir, cfg):
    """Class from regex rules on the file name; first matching rule wins, unmatched files get no label."""
    rules = [(re.compile(p), tok) for p, tok in cfg["rules"]]
    rows = []
    for p in _images_under(src_dir, cfg.get("glob", "**/*")):
        tok = next((tok for rx, tok in rules if rx.search(p.name)), "")
        rows.append([str(p.relative_to(src_dir)), tok, "", "", ""])
    return pd.DataFrame(rows, columns=COLS)


def adapter_odir(src_dir, cfg):
    """ODIR-5K: one row per patient, per-eye diagnostic keywords separated by Chinese or ASCII commas."""
    df = _read_table(src_dir / cfg["file"])
    rows = []
    for side in ("Left", "Right"):
        fcol, kcol = f"{side}-Fundus", f"{side}-Diagnostic Keywords"
        _need(df, ["ID", fcol, kcol], cfg["file"])
        for _, r in df.iterrows():
            kws = [k.strip() for k in re.split(r"[，,]", _norm(r[kcol])) if k.strip()]
            rows.append([str(Path(cfg["image_dir"]) / _norm(r[fcol])), "|".join(kws), _norm(r["ID"]), side.lower(), ""])
    return pd.DataFrame(rows, columns=COLS)


def adapter_labels_csv(src_dir, cfg):
    """Manually converted sources: <source>/labels.csv with image,label[,patient,eye,grade]; label tokens '|'-separated."""
    df = _read_table(src_dir / cfg.get("file", "labels.csv"))
    _need(df, ["image", "label"], "labels.csv")
    get = lambda r, c: _norm(r[c]) if c in df.columns else ""
    rows = [[get(r, "image"), get(r, "label"), get(r, "patient"), get(r, "eye").lower(), get(r, "grade")]
            for _, r in df.iterrows()]
    return pd.DataFrame(rows, columns=COLS)


ADAPTERS = {
    "csv": adapter_csv,
    "folders": adapter_folders,
    "filename": adapter_filename,
    "odir": adapter_odir,
    "labels_csv": adapter_labels_csv,
}


def run_adapter(src, raw_root):
    cfg = src["adapter"]
    return ADAPTERS[cfg["type"]](Path(raw_root) / src["id"], cfg)
