#!/bin/bash
# 004 per-segment signals. Submit two jobs in parallel from ~/tpa-work/repo on UNF HPC, one GPU each:
#   sbatch scripts/slurm/signals.sbatch --stages transcript,audio      # job A: shared audio extraction
#   sbatch scripts/slurm/signals.sbatch --stages people,expression     # job B: shared frame decode
# Extra arguments pass through (e.g. --video-id v03, --kind promo for a smoke test, --force).
# Timed out? Resubmit the same line: finished (video, stage) pairs are skipped. One video failing is logged
# and the job continues with the next. Weights must already be in ~/hf-cache (scripts/download_weights.sh on
# the login node); nothing is downloaded here. Logs are redacted; slurm-*.out is gitignored.
#SBATCH --job-name=tpa-signals
#SBATCH --partition=gpu
#SBATCH --gres=gpu:6KSE:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

export HF_HOME="$HOME/hf-cache"
export HF_HUB_OFFLINE=1
export YOLO_OFFLINE=1
export TPA_WEIGHTS_DIR="$HOME/hf-cache/tpa-weights"

cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
uv sync --frozen
# faster-whisper (CTranslate2) needs CUDA 12 cuBLAS and cuDNN 9; the node has CUDA 13. Use the pip-installed libraries.
CUDA12_LIBS="$(uv run --frozen python -c 'import nvidia.cublas.lib as b, nvidia.cudnn.lib as d; print(b.__path__[0] + ":" + d.__path__[0])')"
export LD_LIBRARY_PATH="$CUDA12_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run --frozen tpa-index signals "$@"
