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

## Review app (003)

```bash
uv run tpa-index mock --seed 0 --clips    # mock tables plus synthetic videos (no footage)
# config/official_runs.yaml lists the official runs, e.g. predict: [mock-seed0]
uv run tpa-index app                      # http://127.0.0.1:7860 (port: pipeline.yaml app.port)
```

Enter a reviewer (first name or role, never a participant's name), then pick a tab.

- **Blind labeling** shows no model output. `1`–`9`/`0` pick the primary activity, letter keys toggle the other labels (shown next to each), `U` marks uncertain/mixed, `Enter` saves and moves on, `Backspace` goes back, `Space` replays. Every save is stored, so closing and reopening resumes where you stopped.
  - **Main sample**: a seeded random sample of eval segments (`app.sample_size`), saved once and never reshuffled.
  - **Overlap set**: the first `app.overlap_size` main segments, for a second labeler (agreement).
  - **Top-up**: for labels still rare after the main sample, the eval segments with the highest official model score, shuffled and still shown blind. It appears once official predict runs cover the eval videos.
- **Review** shows moments per video, filterable by label, needs-review, status and a show-idle toggle. Eval videos stay hidden until every main segment has a blind save and every overlap segment has saves from two reviewers. Selecting a moment plays it with `app.padding_s` seconds of context. Every AI label is marked "AI suggestion, unvalidated" until reviewed. Labels are listed primary first, then by strength (score / threshold). Accept, edit, reject, defer, add a label or add an observation; each action is one append-only row in `reviews.duckdb`, shown in the moment's history. Notes: no names, no judgments about a person.

The app listens on 127.0.0.1 only, never creates a share link and sends no telemetry. Clips are cut on demand into a temporary folder outside the repo and deleted when the next clip is shown and when the app exits.

## Annotation export (006)

```bash
uv run tpa-index export                   # official moments + review log -> OUTPUT_DIR/exports/
uv run tpa-index export --verified-only   # leave out moments whose primary label was rejected
uv run tpa-index export --sample          # Devpost sample from mock data only -> docs/samples/
```

One record per official moment, in the case's 13 fields (`specs/006-export-demo/contracts/export-format.md`), as JSON and CSV. Each label shows its latest review; `review_decision` follows the primary label. Each run writes `annotations-{time}-{hex}-g{guideline}.json/.csv` (never overwritten) and refreshes `annotations-latest.json/.csv`. No reviewer names, filenames or transcript text.

## Mac benchmark (006)

```bash
uv run tpa-index benchmark --video-id <id>   # type the id at run time; never commit it
```

Runs segment, predict, transcript, audio, people, expression and moments for one video on this machine (Mac model sizes from `pipeline.yaml`). Use a promo that already has every official stage, because the signal stages and moments work on the official segments. Each stage writes its usual runs row (`tpa-index runs`). The summary (`OUTPUT_DIR/benchmarks/benchmark-*.md/.json`) compares the Mac with the official HPC runs per hour of video, per stage and end to end. Processing time excludes model loading, which is shown once per stage. Benchmark runs never go into `config/official_runs.yaml`.

## App tabs (006 redesign)

The review app (`uv run tpa-index app`, 127.0.0.1 only) has five tabs under a deep-purple top bar. IBM Plex fonts come from Gradio's own bundled files, so nothing is fetched from the internet.

- **Scorecard** (opens first): reads the newest complete `OUTPUT_DIR/results/insights-*` folder (run `tpa-index insights` first). Compare two phases (default early vs showcase). Shows the TPA Expression Index gauge (four measures, equal weights), the four headline measures with every tag exactly as 005 wrote it, and facts from the newest `evaluate-*` folder and the time breakdown. **Download report** writes the plain-English Word report.
- **Search moments**: a local filter over moments (label names, review status, "needs review", solo or group). It never searches transcripts. The same eval lock applies, and staff-led moments are hidden unless you ask for them. Selecting a result opens it in the Review queue.
- **Review queue**: the queue for one video, the clip with what the AI noticed (suggestions, unvalidated), and your review. Accept, Edit, Reject and Defer (keys A, E, R and D; ↓ moves to the next item) save the same records as before. Ticking or unticking activities rejects AI labels or adds staff labels. The five observation buttons save a staff observation (`expression_*`, guideline 0.6.0), and the note is saved as `reviewer_note`. The AI's suggestion is shown as a ratio to its cut-off, never as a percentage.
- **Blind labels**: unchanged.
- **Export**: writes the annotation export (JSON and CSV) and offers it for download.

Screenshots for the deck come from mock data only (`tpa-index mock --clips` in a temporary `OUTPUT_DIR`).

## Data rules

See `.specify/memory/constitution.md`. In short: no footage, filenames, or metadata in git or in AI assistant context; no identity (faces, tracks, embeddings); no judgments about people; every AI label is a suggestion that staff review. Assistant read-deny rules are in `.claude/settings.json`.
