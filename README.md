# TPA video index

Turns long program videos into a searchable, staff-verified timeline of performance and participation moments, plus program-level indicators. Everything runs locally or on UNF HPC; footage never leaves those machines. The foundation (this package, `tpa_index`) fixes the data contract, configs, mock data, segmenter, and run tracking that later features build on.

## Install (Mac or UNF HPC)

```bash
uv sync    # Linux pulls CUDA 13.0 torch wheels; macOS pulls default (MPS) wheels
```

On HPC, keep uv's cache off a small home quota: `export UV_CACHE_DIR=/path/to/scratch/uv-cache`.

## Configure

Create `.env` in the repo root (gitignored):

```bash
DATA_DIR=/absolute/path/outside/repo/videos
OUTPUT_DIR=/absolute/path/outside/repo/outputs
```

Both must be outside the repo or every command exits with code 2. Real video metadata goes in `config/videos.local.yaml` (gitignored), using the format of `config/videos.example.yaml`. Labels and pipeline settings live in `config/labels.yaml` and `config/pipeline.yaml`.

## Run

```bash
uv run tpa-index check                        # guard, configs, device, hardware
uv run tpa-index mock --seed 0                # full fake dataset, no footage needed
uv run tpa-index segment                      # all videos in videos.local.yaml
uv run tpa-index segment --video-id v01
uv run pytest                                 # mock data and synthetic clips only
```

Outputs: `{OUTPUT_DIR}/runs/{run_id}/{table}.parquet` (immutable per run) and `{OUTPUT_DIR}/reviews.duckdb` (append-only). Read them with `tpa_index.schema.read_table(name)` and `read_reviews()`.

## UNF HPC

```bash
sbatch scripts/slurm/job.sbatch tpa-index check
sbatch scripts/slurm/job.sbatch tpa-index segment
```

The job requests one GPU on the `gpu` partition with a time limit; logs go to `slurm-tpa-index-<jobid>.out` (gitignored, redacted).

## Data rules

See `.specify/memory/constitution.md`. In short: no footage, filenames, or metadata in git or in AI assistant context; no identity (faces, tracks, embeddings); no judgments about people; every AI label is a suggestion that staff review. Assistant read-deny rules are in `.claude/settings.json`.
