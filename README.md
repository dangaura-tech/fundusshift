# FundusShift and LOTUS

[![DOI](https://zenodo.org/badge/1444002583.svg)](https://doi.org/10.5281/zenodo.23911966)

Code, results and supplementary material for:

> **When Fundus Classifiers Meet a New Source: The FundusShift Benchmark and a Low-Capacity Baseline.**
> Dang Truong Minh Le, Mohamed bin Zayed University of Artificial Intelligence.

**FundusShift** is a source-held-out benchmark of normal (N), diabetic retinopathy (D) and glaucoma (G) colour fundus photographs from fifteen public collections. It evaluates the accuracy and reliability (calibration, selective prediction, conformal sets) of 818 fine-tuned, frozen, prompt-based and domain-generalization configurations and four multimodal LLMs under one protocol, with 50 random source rotations.

**LOTUS** (Leave-One-source-out Token heads Unified with zero-shot Scores) is a low-capacity method on a frozen MedSigLIP encoder: a bag of linear heads on mean patch tokens, each trained without one development source, fused with zero-shot text scores (about 8,000 trained parameters). Over the 50 rotations it reaches a balanced accuracy of 0.789 against 0.746 for the strongest probe, with an expected calibration error of 0.034 against 0.084. The *initial design* (LOTUS plus a between-site projection) is reported alongside it.

**Paper (PDF, 15 pages):** [view online](https://github.com/dangaura-tech/fundusshift/blob/main/paper/main.pdf) · [download](https://github.com/dangaura-tech/fundusshift/raw/main/paper/main.pdf)

**Supplementary material (PDF, 37 pages):** [view online](https://github.com/dangaura-tech/fundusshift/blob/main/paper/supplementary.pdf) · [download](https://github.com/dangaura-tech/fundusshift/raw/main/paper/supplementary.pdf)

## Main results

- Over 50 random source rotations (three seeds), LOTUS reached balanced accuracy 0.789 against 0.746 for the strongest probe (higher on 43 of 50) and ECE 0.034 against 0.084 (lower on 47 of 50).
- Choosing the configuration in each rotation from development data only (leave-one-source-out) still gave 0.780 against 0.746 (39 of 50) and ECE 0.035 (47 of 50).
- Validation barely ranks fine-tuned models at unseen sources (Spearman 0.05); development-tuned risk, conformal and specificity targets break at new sources; glaucoma is the weakest class.
- Leave-one-source-out thresholds kept a 10% risk target on 42 of 50 rotations instead of 19.

## Repository layout

| Path | Content |
|---|---|
| `fsbench/` | Benchmark construction: label mapping, manifest, duplicate and near-duplicate detection (vessel signature + NCC), leakage-grouped splits, audit, image processing |
| `configs/` | Source catalogue and the released benchmark configuration |
| `scripts/` | Download and extraction helpers for the public sources; `revision.sbatch` and `extra.sbatch` for the LOTUS runs on a SLURM cluster |
| `baselines/`, `experiments/` | Feature extraction, all baselines, calibration, evaluation, statistics, LOTUS and its ablations |
| `results/` | Saved result tables behind every number in the paper (CSV; large tables gzipped); LOTUS in `results/revision/`, `results/lotus/` and `results/extra/` |
| `paper/main.pdf`, `paper/supplementary.pdf` | The paper and the supplementary material |
| `paper/latex/` | LaTeX sources (`main.tex`, `supplementary.tex`, `refs.bib`), generated tables and figures, and the scripts that build them: `make_assets.py` (main paper), `make_supp.py` and `make_supp_extra.py` (supplement) |
| `docs/` | Evaluation protocol, datasheet, frozen configuration of the initial design, pretraining-overlap and ethics checks |

## Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU runs used Linux with one NVIDIA RTX 5000 Ada; LOTUS, the probes and all post-hoc analyses run on CPU from saved features. MedSigLIP and RETFound weights are gated: request access from their providers and point `FS_GATED` at the folder that holds them.

## Data

The benchmark is public on Hugging Face: https://huggingface.co/datasets/le-dang1haha/FundusShift (images under their source licences, `metadata/splits_final.csv` = the exact split used in the paper, leakage groups and duplicate lists). Every source keeps its own licence (see `configs/sources.yaml` and `docs/ethics_by_source.md`); `scripts/` and `fsbench/` rebuild the benchmark from the original downloads. Note that OIA-DDR is CC BY-NC-SA 4.0, so derived data is non-commercial.

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

**3. Initial design** (configuration in `docs/proposed_frozen_config.json`)
```
python -m experiments.top_dev_final fit --bench $DATA/bench --features $DATA/runs/features --zeroshot $DATA/runs/medsiglip --strata $DATA/strata --out $DATA/runs/top_dev
python -m experiments.top_dev_final eval --bench $DATA/bench --runs $DATA/runs --out results/top_dev
```

**4. Source rotations of the baselines, label scope and the development search**
```
python -m experiments.supplement rotation --bench $DATA/bench --runs $DATA/runs --features $DATA/runs/features --out results/supplement
python -m experiments.scope dev --bench $DATA/bench --strata $DATA/strata --zeroshot $DATA/runs/medsiglip --out results/scope
```

**5. LOTUS: factorial ablation over the published split and the 50 rotations (three seeds), leave-one-region-out, and the leave-one-source-out reference.** On a SLURM cluster, `bash scripts/revision.sbatch submit` runs all three; otherwise:
```
python -m experiments.revision_runs rotation --bench $DATA/bench --strata $DATA/strata --zeroshot $DATA/runs/medsiglip --out results/revision
python -m experiments.revision_runs region --bench $DATA/bench --strata $DATA/strata --zeroshot $DATA/runs/medsiglip --features $DATA/runs/features --out results/revision
python -m experiments.revision_runs certify --bench $DATA/bench --strata $DATA/strata --zeroshot $DATA/runs/medsiglip --out results/revision
```
Each accepts `--worker i --workers k`. Then copy the published-split logits into the runs folder and evaluate them:
```
mkdir -p $DATA/runs/lotus $DATA/runs/lantern
for s in 0 1 2; do cp results/revision/final/no_projection_s$s.npy $DATA/runs/lotus/logits_s$s.npy; cp results/revision/final/full_s$s.npy $DATA/runs/lantern/logits_s$s.npy; done
python -m experiments.extra_runs nested --bench $DATA/bench --strata $DATA/strata --zeroshot $DATA/runs/medsiglip --out results/extra
python -m experiments.extra_runs recipe --bench $DATA/bench --strata $DATA/strata --zeroshot $DATA/runs/medsiglip --features $DATA/runs/features --out results/extra
python -m experiments.extra_runs prior --bench $DATA/bench --strata $DATA/strata --zeroshot $DATA/runs/medsiglip --revision results/revision --out results/extra
python -m experiments.extra_runs screen --bench $DATA/bench --strata $DATA/strata --zeroshot $DATA/runs/medsiglip --revision results/revision --out results/extra
python -m experiments.lotus_eval --bench $DATA/bench --runs $DATA/runs --out results/lotus
python -m experiments.lotus_rotation --bench $DATA/bench --logits results/revision/rot_logits --out results/lotus
python -m experiments.review_tables --bench $DATA/bench --images $DATA/processed --runs $DATA/runs --out results/lotus/review --method-dir lotus --method-name LOTUS --fig-dir paper/latex/figures
```

**6. Tables, figures and PDFs**
```
FS_DATA=$DATA FS_RUNS=$DATA/runs python paper/latex/make_assets.py
FS_DATA=$DATA python paper/latex/make_supp.py
FS_DATA=$DATA FS_RUNS=$DATA/runs python paper/latex/make_supp_extra.py
cd paper/latex && latexmk -pdf main.tex supplementary.tex
```

## Where each result comes from

| Paper item | Script | Saved table |
|---|---|---|
| Benchmark results and baseline reliability (Tables 2, 7, 8; Fig. 2) | `experiments/full_eval.py` | `results/full_eval/` |
| LOTUS on the published split, paired tests (Tables 7, 8) | `experiments/lotus_eval.py` | `results/lotus/` |
| Rotations (Table 5, Fig. 3) | `experiments/supplement.py`, `experiments/revision_runs.py`, `experiments/lotus_rotation.py` | `results/supplement/rotation.csv`, `results/lotus/rotation_lotus.csv` |
| Rotation ablation, seeds, glaucoma-source and region analyses (Table 6; Suppl. S6) | `experiments/revision_runs.py` | `results/revision/` |
| Selection without held-out data, input versus recipe, prior correction over rotations, screening thresholds (Sec. 5; Suppl. S6) | `experiments/extra_runs.py` | `results/extra/` |
| Shift-aware reference (Table 9) | `experiments/revision_runs.py certify` | `results/revision/certify_no_projection/` |
| Per-source, disease-level and scope tables, risk-coverage curves (Suppl. S4, S5) | `experiments/review_tables.py` | `results/lotus/review/` |
| Development ablation and design search (Table 4; Suppl. S8) | `experiments/strata_gate*.py`, `experiments/rptoken.py` | `results/strata/` |
| Label-scope analysis | `experiments/scope.py` | `results/scope/` |
| Projection analysis, site and disease probes | `experiments/review_checks.py`, `experiments/site_probe.py` | `results/review/` |
| Patch-level evidence maps (initial design) | `experiments/evidence_maps.py` | figure in the supplementary material |

## Licence

Code: MIT (see `LICENSE`). Data: each source keeps its own licence.
