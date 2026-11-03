# FundusShift and LANTERN

[![DOI](https://zenodo.org/badge/1444002583.svg)](https://doi.org/10.5281/zenodo.23911966)

Code and supplementary material for:

> **FundusShift: Benchmarking Accuracy and Reliability of Fundus Disease Classification at Unseen Sources, and a Low-Capacity Remedy.**
> Dang Truong Minh Le, Mohamed bin Zayed University of Artificial Intelligence.

FundusShift is a source-held-out benchmark of normal (N), diabetic retinopathy (D) and glaucoma (G) colour fundus photographs from fifteen public collections. It evaluates accuracy and reliability (calibration, selective prediction, conformal sets) of fine-tuned, frozen, prompt-based and domain-generalization models under one protocol, with 50 random source rotations. LANTERN is a low-capacity method on a frozen MedSigLIP encoder: between-site nuisance projection, a bag of linear heads that each leave out one source, and fusion with zero-shot text scores.

**Supplementary material (PDF, 24 pages):** [view online](https://github.com/dangaura-tech/fundusshift/blob/main/paper/supplementary.pdf) · [download](https://github.com/dangaura-tech/fundusshift/raw/main/paper/supplementary.pdf)

## Repository layout

| Path | Content |
|---|---|
| `fsbench/` | Benchmark construction: label mapping, manifest, duplicate and near-duplicate detection (vessel signature + NCC), leakage-grouped splits, audit, image processing |
| `configs/` | Source catalogue and the released benchmark configuration |
| `scripts/` | Download and extraction helpers for the public sources |
| `baselines/`, `experiments/` | Feature extraction, all baselines, calibration, evaluation, statistics and LANTERN |
| `results/` | Saved result tables behind every number in the paper (CSV; large tables gzipped) |
| `paper/make_assets.py`, `paper/make_supp.py` | Rebuild every table and figure of the paper and of the supplementary material from `results/` |
| `docs/` | Evaluation protocol, datasheet, frozen LANTERN configuration, pretraining-overlap and ethics checks |

## Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU runs used Linux with one NVIDIA RTX 5000 Ada; LANTERN, the probes and all post-hoc analyses run on CPU from saved features. MedSigLIP and RETFound weights are gated: request access from their providers and point `FS_GATED` at the folder that holds them.

## Data

Images are not redistributed here. Every source is public and is obtained from its original publisher under its own licence (see `configs/sources.yaml` and `docs/ethics_by_source.md`); `scripts/` and `fsbench/` rebuild the benchmark from those downloads. The split tables used in the paper are available from the author on request. Note that OIA-DDR is CC BY-NC-SA 4.0, so derived data is non-commercial.

Build order: `scripts/download_open.sh`, `scripts/extract.sh`, then `fsbench.prepare_labels`, `build_manifest`, `dedup`, `make_splits`, `audit`, `process_images`.

## Reproducing the paper

Set `DATA=<folder with bench/, processed/, runs/>`.

**1. Features and zero-shot scores**
```
python -m experiments.extract_features --bench $DATA/bench --images $DATA/processed --out $DATA/runs/features
python -m experiments.medsiglip_zeroshot --bench $DATA/bench --images $DATA/processed --features $DATA/runs/features --out $DATA/runs/medsiglip
python -m experiments.rptoken extract --bench $DATA/bench --images $DATA/processed --out $DATA/strata
```

**2. Benchmark** (fine-tuning, probes, prompt methods, DG methods, MCQ), then one evaluator for raw, TS and FTS
```
python -m experiments.finetune ...        # see --help of each module
python -m experiments.probe_calib ...
python -m experiments.coop ... ; python -m experiments.tipadapter ...
python -m experiments.robust_train ... ; python -m experiments.wiseft ...
python -m experiments.full_eval --bench $DATA/bench --runs $DATA/runs --features $DATA/runs/features --out results/full_eval
python -m experiments.paired_tests --bench $DATA/bench --runs $DATA/runs --views results/full_eval/views.csv --out results/paired_tests
```

**3. LANTERN** (configuration in `docs/proposed_frozen_config.json`)
```
python -m experiments.top_dev_final fit --bench $DATA/bench --features $DATA/runs/features --zeroshot $DATA/runs/medsiglip --strata $DATA/strata --out $DATA/runs/top_dev
python -m experiments.top_dev_final eval --bench $DATA/bench --runs $DATA/runs --out results/top_dev
```

**4. Source rotations, ablation and promises**
```
python -m experiments.supplement rotation --bench $DATA/bench --runs $DATA/runs --features $DATA/runs/features --out results/supplement
python -m experiments.lantern_rotation --bench $DATA/bench --features $DATA/runs/features --zeroshot $DATA/runs/medsiglip --strata $DATA/strata --out results/lantern_rotation.csv
python -m experiments.ablation_rotation --bench $DATA/bench --strata $DATA/strata --zeroshot $DATA/runs/medsiglip --out results/ablation_rotation
python -m experiments.certify --bench $DATA/bench --strata $DATA/strata --zeroshot $DATA/runs/medsiglip --out results/certify
python -m experiments.scope dev --bench $DATA/bench --strata $DATA/strata --zeroshot $DATA/runs/medsiglip --out results/scope
```
`ablation_rotation` and `certify` accept `--worker i --workers k` to split the rotations across processes.

**5. Tables and figures**
```
FS_DATA=$DATA python paper/make_assets.py   # main paper
FS_DATA=$DATA python paper/make_supp.py     # supplementary material
```

## Where each result comes from

| Paper item | Script | Saved table |
|---|---|---|
| Benchmark results, reliability (Tables 5, 8; Figs. 2, 4) | `experiments/full_eval.py` | `results/full_eval/` |
| Paired tests | `experiments/proposed_paired.py`, `experiments/paired_tests.py` | `results/last_reg/`, `results/paired_tests/` |
| Rotations (Table 6, Fig. 3) | `experiments/supplement.py`, `experiments/lantern_rotation.py` | `results/supplement/rotation.csv`, `results/lantern_rotation.csv` |
| Rotation ablation (Table 7) | `experiments/ablation_rotation.py` | `results/ablation_rotation/` |
| Shift-aware reference (Table 9) | `experiments/certify.py` | `results/certify/` |
| Development ablation and design search | `experiments/strata_gate*.py`, `experiments/rptoken.py` | `results/strata/` |
| Label-scope analysis | `experiments/scope.py` | `results/scope/` |
| Seeds over rotations, projection analysis, glaucoma-source analysis, ECE intervals | `experiments/ablation_rotation.py --seed`, `experiments/review_checks.py` | `results/ablation_rotation/`, `results/review/` |
| Patch-level evidence maps | `experiments/evidence_maps.py` | figure in the supplementary material |
| Site and disease probes, t-SNE of fine-tuned and frozen features | `experiments/site_probe.py` | `results/review/site_probe.csv` |


## Licence

Code: MIT (see `LICENSE`). Data: each source keeps its own licence.
