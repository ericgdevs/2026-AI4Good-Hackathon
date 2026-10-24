#!/bin/bash
# Usage (from ~/tpa-work/repo on UNF HPC):
#   sbatch scripts/slurm/predict.sbatch                  # all mapped videos
#   sbatch scripts/slurm/predict.sbatch --video-id v03   # selected videos
#   sbatch scripts/slurm/predict.sbatch --kind promo     # test run on one kind (do not list it as official)
# Weights must already be in ~/hf-cache; nothing is downloaded. Logs are redacted; slurm-*.out is gitignored.
#SBATCH --job-name=tpa-predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:6KSE:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

export HF_HOME="$HOME/hf-cache"
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
uv sync --frozen
uv run tpa-index predict "$@"
