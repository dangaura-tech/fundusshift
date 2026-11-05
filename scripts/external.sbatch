#!/bin/bash
#SBATCH --job-name=fs-ext
#SBATCH --time=06:00:00
#SBATCH --output=logs/external_%x_%j.out
set -euo pipefail
cd $HOME/fundusshift/repo_main
for m in $MODES; do echo == $m; [ -n "${NOGPU:-}" ] && export CUDA_VISIBLE_DEVICES=""; $HOME/fundusshift/conda_env/bin/python -m experiments.external $m --root $HOME/fundusshift; done
