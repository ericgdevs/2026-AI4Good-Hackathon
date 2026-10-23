#!/bin/bash
# Usage: sbatch scripts/slurm/job.sbatch tpa-index check
#        sbatch scripts/slurm/job.sbatch tpa-index segment
# Submit from the repo root. Logs are redacted by the app; slurm-*.out is gitignored.
#SBATCH --job-name=tpa-index
#SBATCH --partition=gpu
#SBATCH --gres=gpu:6KSE:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
uv sync --frozen
uv run "$@"
