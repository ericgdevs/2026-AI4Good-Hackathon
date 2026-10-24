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

Outputs: `{OUTPUT_DIR}/runs/{run_id}/{table}.parquet` (immutable per run) and `{OUTPUT_DIR}/reviews.duckdb` (append-only). Result-facing code reads with `tpa_index.schema.read_official(name)` (official runs only); `read_table(name)` reads every run and is for tools and tests.

## Visual scoring (002)

```bash
uv run tpa-index predict                      # every mapped video, one run per video (siglip2-base on the Mac)
uv run tpa-index predict --kind promo         # one kind only, e.g. a test run
uv run tpa-index runs                         # run IDs, stage, video IDs, hardware, counts, runtime, peak memory
```

Every visual label gets a 0–1 score per segment. `uncertain_mixed` (per segment) and `needs_review` (per moment) are rules, never model scores. A segment is idle only when the idle prompts match **and** the picture barely changes. Weights load from the local Hugging Face cache only.

## Official runs and calibration

List the runs that count in `config/official_runs.yaml` (`predict:`, optionally `moments:`). Then:

```bash
uv run tpa-index calibrate --dry-run          # official dev-split data only; never eval
uv run tpa-index calibrate                    # writes config/thresholds.local.yaml (gitignored, data-derived)
uv run tpa-index moments                      # rebuild idle flags + moments with the new thresholds, no model
```

Label thresholds are the higher of a dev-score percentile (`calibrate.top_pct`) and a floor (`calibrate.min_threshold`); calibrate lists labels that hit the floor. The idle threshold uses its own percentile (`idle.calibrate_top_pct`); `idle.change_threshold` stays hand-set. List the new `moments` runs under `moments:` to use them.

## UNF HPC

```bash
sbatch scripts/slurm/job.sbatch tpa-index check
sbatch scripts/slurm/job.sbatch tpa-index segment
```

The job requests one GPU on the `gpu` partition with a time limit; logs go to `slurm-tpa-index-<jobid>.out` (gitignored, redacted).

```bash
scripts/sync_code_to_hpc.sh --dry-run && scripts/sync_code_to_hpc.sh   # Mac -> ~/tpa-work/repo (code only)
sbatch scripts/slurm/predict.sbatch                                    # on HPC: predict on one GPU (HF_HOME=~/hf-cache)
scripts/sync_outputs_from_hpc.sh --dry-run && scripts/sync_outputs_from_hpc.sh   # HPC run folders -> local OUTPUT_DIR
```

The sync scripts read `HPC_USER`, `HPC_HOST` and `HPC_OUTPUT_DIR` from `.env` (see `.env.example`). Code sync never sends `.env`, `config/*.local.yaml`, data, weights or outputs (`scripts/sync.exclude`). Output sync copies only run-folder Parquet files, never overwrites a local run, never copies `reviews.duckdb`, and merges HPC's calibrated thresholds per model. HPC keeps its own `.env` and `config/videos.local.yaml`; they are never synced.

## Data rules

See `.specify/memory/constitution.md`. In short: no footage, filenames, or metadata in git or in AI assistant context; no identity (faces, tracks, embeddings); no judgments about people; every AI label is a suggestion that staff review. Assistant read-deny rules are in `.claude/settings.json`.
