"""Shared helpers: source registry loading and native -> harmonized label mapping."""
import re
from pathlib import Path

import yaml

CLASSES = ["N", "D", "G", "A", "M"]   # canonical order; a release uses those present in train
USED_ROLES = {"dev", "test", "ood_test", "inventory"}   # inventory: deduplicated, never released
_MAP_VALUES = set(CLASSES) | {"IGNORE", "EXCLUDE", "OPENSET"}   # OPENSET: outside the core classes, kept as probe


def load_sources(path):
    with open(path) as f:
        sources = yaml.safe_load(f)["sources"]
    ids = [s["id"] for s in sources]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate source ids: {dupes}")
    for s in sources:
        s["label_map"] = {str(k).strip().lower(): str(v) for k, v in (s.get("label_map") or {}).items()}
        s["label_map_regex"] = [(re.compile(p, re.I), str(v)) for p, v in (s.get("label_map_regex") or [])]
        bad = {v for v in s["label_map"].values()} | {v for _, v in s["label_map_regex"]}
        bad -= _MAP_VALUES
        if bad:
            raise ValueError(f"{s['id']}: unknown map values {bad}")
    _check_family_sides(sources)
    return sources


def _check_family_sides(sources):
    """A family (acquisition origin) may not appear on both the development and the test side."""
    side = {}
    for s in sources:
        if s.get("role") not in USED_ROLES:
            continue
        sd = "dev" if s["role"] in ("dev", "inventory") else "test"
        fam = s["family"]
        if side.setdefault(fam, sd) != sd:
            raise ValueError(f"family {fam} is used on both dev and test side")


def map_token(token, src):
    t = str(token).strip().lower()
    if t in src["label_map"]:
        return src["label_map"][t]
    for pat, v in src["label_map_regex"]:
        if pat.search(str(token).strip()):
            return v
    return "OTHER"


def harmonize(tokens, src):
    """Map native tokens to one harmonized class.

    Returns (label or None, reason). A label is assigned only when every non-ignored token
    supports the same class; any unmapped/excluded token or a mix of classes removes the image.
    """
    tokens = [t for t in tokens if str(t).strip() != ""]
    if not tokens:
        return None, "no_native_label"
    mapped = {map_token(t, src) for t in tokens} - {"IGNORE"}
    if not mapped:
        return None, "only_ignored_tokens"
    if "EXCLUDE" in mapped:
        return None, "excluded_native_label"
    if "OTHER" in mapped:
        return None, "unmapped_condition"
    if len(mapped) > 1:
        return None, "multiple_classes:" + "+".join(sorted(mapped))
    return mapped.pop(), "ok"


def project_root():
    return Path(__file__).resolve().parents[1]
