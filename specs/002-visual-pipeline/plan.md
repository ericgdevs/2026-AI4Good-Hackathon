# Implementation Plan: Visual Pipeline

**Branch**: `002-visual-pipeline` | **Date**: 2026-10-23 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-visual-pipeline/spec.md`

## Summary

Add a `predict` stage. For each video it:
- reuses the 001 segmenter (letterboxed frames, no audio);
- embeds the frames and each label's ensembled prompts with SigLIP2;
- scores every visual label with SigLIP's own sigmoid;
- marks segments idle when the idle prompt matches **and** the frames barely change;
- writes a rule-based `uncertain_mixed` row;
- merges moments with each video's effective window.

Each video is its own tracked run. An optional X-CLIP model adds motion-label rows under its own model name.

Results are selected through `config/official_runs.yaml` and `read_official`. A `runs` command lists runs, and there are a Slurm wrapper and rsync scripts for HPC.

Thresholds are per label and per model, with the global value as fallback. `tpa-index calibrate` proposes them from official **dev** predictions only and writes a gitignored local file. `tpa-index moments` rebuilds moments and `uncertain_mixed` rows with the new thresholds, without re-running the model.

The status group is rules-only: `uncertain_mixed` per segment, and `needs_review` per moment (low confidence or `uncertain_mixed`).

## Technical Context

**Language/Version**: Python 3.12, uv, package `tpa_index` (from 001)

**Primary Dependencies**: transformers 5.20 (`AutoModel` and `AutoProcessor`, weights loaded from local files only), torch 2.14 (cu130 on Linux, MPS on Mac), and the 001 stack (PyAV, pyarrow, DuckDB, PyYAML). No new dependencies.

**Storage**: the existing 001 Parquet run folders and reviews DuckDB, with no schema changes. Adds `config/official_runs.yaml`.

**Testing**: pytest. A `FakeEmbedder` is used for unit tests, plus one `slow` test with real `siglip2-base` that is skipped unless the weights are cached.

**Target Platform**: Mac with MPS (benchmark, siglip2-base) and UNF HPC with one RTX PRO 6000 Blackwell (official runs, siglip2-so400m)

**Project Type**: library + CLI (extends 001)

**Performance Goals**:
- HPC: 1 hour of video in under 15 minutes (SC-007, to be confirmed on the first run).
- Mac: measured and reported, with no target.

**Constraints**:
- Weights are local only, and nothing calls the network during runs.
- No embeddings are stored.
- One run per video.
- Thresholds are config and get tuned on the dev split.

**Scale/Scope**:
- Several videos of up to about 2 hours each, about 1,200 windows per video.
- About 25 visual labels. That gives roughly 30,000 prediction rows per video.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| # | Principle | Status | How |
|---|---|---|---|
| I | Data local/HPC | PASS | Runs happen on the Mac or HPC. Sync scripts move code up and run folders down, and exclude every media type, `.env`, `videos.local.yaml` and the reviews log. Calibrated thresholds stay in a gitignored local file. |
| II | Open models, efficient, measured | PASS | Open SigLIP2 and X-CLIP weights, loaded from local files only with `HF_HUB_OFFLINE=1`. Sparse frames (4 per 6 s) and batching. Every run records runtime, peak memory and hardware. The Mac path uses the small model, for the 16 GB laptop. |
| III | Assistants never see data | PASS | Redaction from 001 applies. `runs` prints opaque IDs only. The assistant never runs predict or `runs` on real outputs. |
| IV | No identity | PASS | Embeddings are per window, in memory, and never written. No per-person or face rows. |
| V | No judgments | PASS | Visual labels are observable activities and settings. Status changes only add a rule flag. |
| VI | Humans verify | PASS | Every row has a score and model. `uncertain_mixed` and `needs_review` surface low confidence for staff. |
| VII | Contracts first | PASS | No table changes. API, CLI and config are in `contracts/`, and the mock data follows the new vocabulary. |
| VIII | Measure everything shown | PASS | `official_runs.yaml` plus `read_official` make sure reported numbers come from known runs. `calibrate` refuses eval data, so thresholds never see the eval split. Metrics come in 005. |
| IX | Simplest thing | PASS | Zero-shot only, no training. One new interface (`Embedder`), there for testability. |
| X | Presentation-ready | N/A | No UI. |
| XI | Config over code | PASS | Models, batch sizes, thresholds, idle prompts, the margin, motion labels and official runs are all YAML. |
| XII | One app, two modes | PASS | Predict output uses the same tables and readers as mock data. |
| XIII | Deliverables | PASS | Runs feed the demo view. Model IDs and the score definition feed the model card. |

**Post-design re-check**: PASS, with no violations.

**Risk handled** (research §3, §15–16): SigLIP's raw sigmoid scores are low, so `label_threshold: 0.5` may flag almost nothing. Per-label thresholds come from `calibrate` on dev-split predictions, followed by `tpa-index moments`. Real calibration against staff labels comes in 005.

**Deviation** (research §15): calibrated thresholds go to the gitignored `config/thresholds.local.yaml`, not `pipeline.yaml`. They are derived from footage (Constitution I), and rewriting `pipeline.yaml` would strip its comments.

## Project Structure

### Documentation (this feature)

```text
specs/002-visual-pipeline/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── interfaces.md
│   └── config.md
└── tasks.md        # /speckit-tasks
```

### Source Code (repository root)

```text
config/
├── labels.yaml            # status group -> uncertain_mixed + needs_review, both rules; guideline 0.2.0
├── pipeline.yaml          # + visual_groups, uncertain_margin, label_thresholds, models, idle; - needs_review_below, conflict_margin
├── official_runs.yaml     # new, committed, run IDs only (predict, moments, segment)
└── thresholds.local.yaml  # gitignored, written by calibrate
.env.example               # new: DATA_DIR, OUTPUT_DIR, HPC_USER, HPC_HOST, HPC_OUTPUT_DIR
src/tpa_index/
├── embed.py             # new: Embedder protocol + SiglipEmbedder
├── score.py             # new: ensembling, sigmoid scores, idle, uncertain_mixed, predict_video
├── xclip.py             # new, optional: XClipScorer (imported only when enabled)
├── segment.py           # with_audio flag
├── moments.py           # window_s + model params (shortcut removed); rules-based needs_review
├── calibrate.py         # new: propose_thresholds (dev only), write_local_thresholds
├── config.py            # source: rules; threshold(); thresholds.local.yaml merge; new keys validated
├── schema.py            # load_official, read_official, list_runs
├── mock.py              # skip rules labels; rules rows; per-video window for moments
└── cli.py               # predict, runs, calibrate, moments
scripts/
├── slurm/predict.sbatch
├── sync.exclude
├── sync_code_to_hpc.sh
└── sync_outputs_from_hpc.sh
tests/
├── fakes.py             # FakeEmbedder
├── test_score.py        # ensembling, sigmoid, idle (both signals), uncertain_mixed
├── test_predict.py      # predict_video + CLI on synthetic clips with FakeEmbedder; one bad video
├── test_official.py     # read_official (incl. moments precedence), bad IDs, duplicate video per stage, list_runs
├── test_calibrate.py    # percentile per label, dev-only (eval rows ignored, --split eval rejected), local file write, moments rebuild
├── test_scripts.py      # sbatch directives + HF_HOME; sync dry-run excludes; missing keys
└── test_siglip_slow.py  # @slow, skipped without cached weights
```

**Structure Decision**: new modules only where 002 adds a job: `embed.py` holds the model, `score.py` turns embeddings into rows, and `xclip.py` stays optional so its import cost and risk only apply when it's on. Everything else extends the 001 modules in place.

## Complexity Tracking

No violations.
