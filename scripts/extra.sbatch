#!/bin/bash
# Follow-up runs (CPU only, from saved features and results/revision). Submit with:   bash scripts/extra.sbatch submit
# After nested and recipe finish:                                                      bash scripts/extra.sbatch after
# Under a per-user job limit (e.g. two jobs on ws-ia): two big jobs, each running many workers at once:
#                                                                                      CPUS=32 MEM=120G bash scripts/extra.sbatch submit2
# Zip the results:                                                                     bash scripts/extra.sbatch pack
#SBATCH --job-name=fs-extra
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/extra_%x_%A_%a.out

ROOT=${ROOT:-$HOME/fundusshift}
CODE=${CODE:-$ROOT/code}
BENCH=${BENCH:-$ROOT/data/bench}
STRATA=${STRATA:-$ROOT/data/strata}
ZEROSHOT=${ZEROSHOT:-$ROOT/runs/medsiglip}
FEATURES=${FEATURES:-$ROOT/runs/features}
REVISION=${REVISION:-$ROOT/results/revision}
OUT=${OUT:-$ROOT/results/extra}
ENV=${ENV:-$ROOT/conda_env}
NNEST=${NNEST:-50}   # workers for nested selection (the heaviest mode)
NREC=${NREC:-20}     # workers for input versus recipe
PARTITION=${PARTITION:-ws-ia}   # one or several, comma-separated (e.g. ws-ia,cpu): each job starts on whichever is free first

set -euo pipefail
PY="$ENV/bin/python"; [ -x "$PY" ] || PY=python

if [ "${1:-}" = "submit" ]; then
  for f in "$BENCH/splits_final.csv" "$STRATA/mean3_medsiglip.npy" "$FEATURES/medsiglip.npy" "$REVISION/rot_logits/no_projection_k1.npy" "$REVISION/final/no_projection_s0.npy"; do
    [ -f "$f" ] || { echo "missing: $f  (set the path variables at the top)"; exit 1; }
  done
  [ -f "$CODE/experiments/extra_runs.py" ] || { echo "missing: $CODE/experiments/extra_runs.py (copy it there first)"; exit 1; }
  mkdir -p "$ROOT/logs" "$OUT"
  cd "$ROOT"
  sbatch -p "$PARTITION" --array=0-$((NNEST - 1)) --export=ALL,MODE=nested,NW=$NNEST "$CODE/scripts/extra.sbatch"
  sbatch -p "$PARTITION" --array=0-$((NREC - 1)) --export=ALL,MODE=recipe,NW=$NREC "$CODE/scripts/extra.sbatch"
  sbatch -p "$PARTITION" --array=0 --time=02:00:00 --export=ALL,MODE=prior,NW=1 "$CODE/scripts/extra.sbatch"
  exit 0
fi

if [ "${1:-}" = "submit2" ]; then   # NJOB jobs; the worker list (nested, recipe, prior) is dealt out between them
  CPUS=${CPUS:-32}; MEM=${MEM:-120G}; NJOB=${NJOB:-2}
  [ -f "$CODE/experiments/extra_runs.py" ] || { echo "missing: $CODE/experiments/extra_runs.py (copy it there first)"; exit 1; }
  mkdir -p "$ROOT/logs" "$OUT"; cd "$ROOT"
  for j in $(seq 0 $((NJOB - 1))); do
    sbatch -p "$PARTITION" --job-name=fs-extra-$j --cpus-per-task=$CPUS --mem=$MEM --time=${TIME:-24:00:00} --output=logs/extra_pack_${j}_%j.out \
      --export=ALL,MODE=pack,JOB=$j,NJOB=$NJOB,CPUS=$CPUS "$CODE/scripts/extra.sbatch"
  done
  exit 0
fi

if [ "${MODE:-}" = "pack" ]; then   # inside one big job: CPUS/2 workers at a time, two threads each
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 SLURM_CPUS_PER_TASK=2
  cd "$CODE"; mkdir -p "$ROOT/logs/extra_workers"
  { for w in $(seq 0 $((NNEST - 1))); do echo "nested $w $NNEST"; done
    for w in $(seq 0 $((NREC - 1))); do echo "recipe $w $NREC"; done
    echo "prior 0 1"; } | awk -v j="$JOB" -v n="$NJOB" '(NR - 1) % n == j' | \
  xargs -P $((CPUS / 2)) -L 1 bash -c '"$0" -m experiments.extra_runs "$1" --bench "'"$BENCH"'" --strata "'"$STRATA"'" --zeroshot "'"$ZEROSHOT"'" --features "'"$FEATURES"'" --revision "'"$REVISION"'" --out "'"$OUT"'" --worker "$2" --workers "$3" > "'"$ROOT"'/logs/extra_workers/$1_$2.log" 2>&1 && echo "done $1 $2" || echo "FAILED $1 $2"' "$PY"
  exit 0
fi

if [ "${1:-}" = "after" ]; then   # screening needs the probe logits written by the recipe jobs
  mkdir -p "$ROOT/logs"; cd "$ROOT"
  sbatch -p "$PARTITION" --array=0 --time=02:00:00 --export=ALL,MODE=screen,NW=1 "$CODE/scripts/extra.sbatch"
  exit 0
fi

if [ "${1:-}" = "pack" ]; then
  cd "$(dirname "$OUT")" && zip -qr "$ROOT/fundusshift_extra.zip" "$(basename "$OUT")" -x "*/lp_logits/*" && ls -lh "$ROOT/fundusshift_extra.zip"
  exit 0
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
cd "$CODE"
"$PY" -m experiments.extra_runs "$MODE" --bench "$BENCH" --strata "$STRATA" --zeroshot "$ZEROSHOT" --features "$FEATURES" --revision "$REVISION" --out "$OUT" --worker "${SLURM_ARRAY_TASK_ID:-0}" --workers "$NW"
