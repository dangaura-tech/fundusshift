#!/bin/bash
# Revision runs (CPU only, from saved features). Submit everything with:   bash scripts/revision.sbatch submit
# Edit the paths below if your layout differs. Results go to $OUT; zip it afterwards with:   bash scripts/revision.sbatch pack
#SBATCH --job-name=fs-revision
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/revision_%x_%A_%a.out

ROOT=${ROOT:-$HOME/fundusshift}
CODE=${CODE:-$ROOT/code}
BENCH=${BENCH:-$ROOT/data/bench}
STRATA=${STRATA:-$ROOT/data/strata}
ZEROSHOT=${ZEROSHOT:-$ROOT/runs/medsiglip}
FEATURES=${FEATURES:-$ROOT/runs/features}
OUT=${OUT:-$ROOT/results/revision}
ENV=${ENV:-$ROOT/conda_env}
NROT=${NROT:-24}     # parallel workers for the rotation ablation
NCERT=${NCERT:-12}   # parallel workers for the LOSO reference

set -euo pipefail

if [ "${1:-}" = "submit" ]; then
  for f in "$BENCH/splits_final.csv" "$STRATA/mean3_medsiglip.npy" "$FEATURES/medsiglip.npy" "$ZEROSHOT/logits_medsiglip_clip_ensemble.npy" "$ZEROSHOT/logits_medsiglip_cupl.npy" "$ZEROSHOT/logits_medsiglip_descriptive_single.npy"; do
    [ -f "$f" ] || { echo "missing: $f  (set the path variables at the top)"; exit 1; }
  done
  "$ENV/bin/python" -c "import torch, sklearn, scipy, pandas" 2>/dev/null || { echo "the env at $ENV cannot import torch/sklearn/scipy/pandas: set ENV=/path/to/env"; exit 1; }
  [ -f "$CODE/experiments/revision_runs.py" ] || { echo "missing: $CODE/experiments/revision_runs.py (copy it there first)"; exit 1; }
  mkdir -p "$ROOT/logs" "$OUT"
  cd "$ROOT"
  sbatch --array=0-$((NROT - 1)) --export=ALL,MODE=rotation,NW=$NROT "$CODE/scripts/revision.sbatch"
  sbatch --array=0-$((NCERT - 1)) --export=ALL,MODE=certify,NW=$NCERT "$CODE/scripts/revision.sbatch"
  sbatch --array=0 --export=ALL,MODE=region,NW=1 "$CODE/scripts/revision.sbatch"
  exit 0
fi

if [ "${1:-}" = "pack" ]; then
  cd "$(dirname "$OUT")" && zip -qr "$ROOT/fundusshift_revision.zip" "$(basename "$OUT")" && ls -lh "$ROOT/fundusshift_revision.zip"
  exit 0
fi

PY="$ENV/bin/python"; [ -x "$PY" ] || PY=python   # the conda env's own python, no activation needed
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
cd "$CODE"
"$PY" -m experiments.revision_runs "$MODE" --bench "$BENCH" --strata "$STRATA" --zeroshot "$ZEROSHOT" --features "$FEATURES" --out "$OUT" --worker "${SLURM_ARRAY_TASK_ID:-0}" --workers "$NW"
