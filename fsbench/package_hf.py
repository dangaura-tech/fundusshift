"""Step 6: package the benchmark as a Hugging Face dataset and push it.

python -m fsbench.package_hf --bench data/bench --processed data/processed \
    --sources configs/fundusshift_open.yaml --card hf/README.md --repo le-dang1haha/FundusShift [--public]
Image splits go to parquet shards via `datasets`; metadata (manifest incl. unused rows, duplicate pairs, audit,
EDA, configs, pipeline code) is uploaded under metadata/ and code/.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import yaml
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi, hf_hub_download

from .common import CLASSES
SPLIT_NAMES = ["train", "validation", "calibration", "test", "test_ood", "test_openset"]
STR_COLS = ["image_id", "label_name", "source", "family", "region", "country", "device", "licence", "native_label",
            "normal_scope", "patient_id", "eye", "group_id", "cal_subset", "orig_sha256", "quality_flags"]
FLOAT_COLS = ["fov_fraction", "brightness", "contrast", "sharpness"]


def features(core):
    f = {"image": Image(), "label": ClassLabel(names=core)}
    f.update({c: Value("string") for c in STR_COLS})
    f.update({"dr_grade": Value("int8"), "orig_width": Value("int32"), "orig_height": Value("int32")})
    f.update({c: Value("float32") for c in FLOAT_COLS})
    return Features(f)


def build(bench, processed, sources):
    meta = {s["id"]: s for s in sources}
    d = pd.read_csv(bench / "splits_final.csv", dtype=str, keep_default_na=False)
    d = d[d.split.isin(SPLIT_NAMES)].copy()
    core = [c for c in CLASSES if c in set(d.label[d.split == "train"])]
    d["image"] = [str(processed / s / f"{i}.jpg") for s, i in zip(d.source, d.image_id)]
    d["label_name"] = d.label
    d["label"] = [core.index(x) if x in core else -1 for x in d.label]
    d["device"] = [str(meta[s].get("device", "")) for s in d.source]
    d["licence"] = [str(meta[s].get("licence", "")) for s in d.source]
    d["orig_sha256"] = d.sha256
    d["dr_grade"] = pd.to_numeric(d.dr_grade, errors="coerce").fillna(-1).astype(int)
    for c in ["orig_width", "orig_height"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(float).astype(int)
    for c in FLOAT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    cols = list(features(core))
    dd = {}
    for sp in SPLIT_NAMES:
        sub = d[d.split == sp].sort_values(["source", "image_id"])
        if len(sub):
            dd[sp] = Dataset.from_pandas(sub[cols].reset_index(drop=True), features=features(core), preserve_index=False)
    return DatasetDict(dd)


def merge_card(repo, card_path, api):
    """Keep the auto-generated dataset_info/configs YAML from push_to_hub, add our front matter and body."""
    auto = Path(hf_hub_download(repo, "README.md", repo_type="dataset")).read_text()
    auto_yaml = yaml.safe_load(auto.split("---")[1]) if auto.startswith("---") else {}
    text = Path(card_path).read_text()
    _, ours_yaml, body = text.split("---", 2)
    front = {**yaml.safe_load(ours_yaml), **{k: v for k, v in auto_yaml.items() if k in ("dataset_info", "configs")}}
    out = "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---" + body
    api.upload_file(path_or_fileobj=out.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                    commit_message="Dataset card")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--skip-images", action="store_true", help="only refresh card + metadata")
    a = ap.parse_args(argv)
    bench, processed = Path(a.bench), Path(a.processed)
    sources = yaml.safe_load(open(a.sources))["sources"]
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=not a.public, exist_ok=True)

    if not a.skip_images:
        ds = build(bench, processed, sources)
        print(ds)
        ds.push_to_hub(a.repo, private=not a.public, max_shard_size="500MB",
                       commit_message="FundusShift image splits")

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "metadata/eda").mkdir(parents=True)
        for f in ["manifest.csv", "near_duplicates.csv", "dup_candidates.csv", "cross_source_reuse.csv",
                  "audit_report.md", "splits_final.csv", "quality.csv"]:
            if (bench / f).exists():
                shutil.copy(bench / f, t / "metadata" / f)
        for f in (bench / "eda").glob("*"):
            shutil.copy(f, t / "metadata/eda" / f.name)
        shutil.copytree(root / "configs", t / "code/configs")
        shutil.copytree(root / "fsbench", t / "code/fsbench", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(root / "scripts", t / "code/scripts")
        shutil.copytree(root / "baselines", t / "code/baselines", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(root / "tests", t / "code/tests", ignore=shutil.ignore_patterns("__pycache__"))
        for f in ["requirements.txt", "run_all.sh"]:
            shutil.copy(root / f, t / "code" / f)
        api.upload_folder(folder_path=str(t), repo_id=a.repo, repo_type="dataset",
                          commit_message="Metadata, audit, EDA and pipeline code")
    merge_card(a.repo, a.card, api)
    print(f"https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
