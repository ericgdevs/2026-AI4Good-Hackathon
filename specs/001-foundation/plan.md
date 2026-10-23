# Implementation Plan: Shared Data Foundation

**Branch**: `001-foundation` | **Date**: 2026-10-23 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-foundation/spec.md`

## Summary

Build the contract layer every later feature imports. It has six parts:

- **Tables**: six schemas defined in pyarrow. Pipeline outputs are written as per-run immutable Parquet files and reviews go to an append-only DuckDB table.
- **Config**: YAML files for labels and pipeline settings.
- **Mock data**: a seeded generator that runs the real merge and validation code.
- **Segmenter**: PyAV-based. It yields letterboxed frames and 16 kHz mono audio per window, in memory.
- **Run tracking**: a context manager that records runtime, peak memory and hardware.
- **Data guard**: a `.env` path guard plus global redaction.

The same `pyproject.toml` runs on the Mac (MPS) and on UNF HPC (CUDA 13.0 wheels), with a Slurm template. There is no model inference.

## Technical Context

**Language/Version**: Python 3.12, managed with uv; package `tpa_index` (src layout)

**Primary Dependencies**:
- Already locked: pyarrow, duckdb, PyAV (`av`), PyYAML, python-dotenv, numpy (transitive), torch (device and memory only).
- New: pytest, in the dev group.

**Storage**: Parquet at `{OUTPUT_DIR}/runs/{run_id}/{table}.parquet` and DuckDB at `{OUTPUT_DIR}/reviews.duckdb`

**Testing**: pytest with mock data and synthetic clips (landscape, portrait, no-audio) generated in `tmp_path`

**Target Platform**: macOS arm64 (MPS/CPU, 16 GB) and UNF HPC Linux x86_64 (RTX PRO 6000 Blackwell, driver 13.0, Slurm)

**Project Type**: Python library + CLI (`tpa-index`)

**Performance Goals**:
- Mock dataset generated in under 60 s.
- Guard failure in under 1 s, because torch is imported lazily.
- Segmenting decodes only 4 frames per window, never every frame.

**Constraints**:
- No data, paths or filenames in the repo, logs or errors.
- No identity columns.
- Same segment rows on both machines.

**Scale/Scope**: several videos of up to about 2 h each, a few thousand segments, 30–40 labels

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| # | Principle | Status | How |
|---|---|---|---|
| I | Data local / HPC only | PASS | Paths come only from `.env`. The guard rejects paths inside the repo. Nothing is uploaded. Retention and deletion are not touched. |
| II | Open models, efficient, measured | PASS | No models in this feature. Sparse frame sampling. `track_run` records runtime and peak memory on CUDA, MPS and CPU. |
| III | Assistants never see data | PASS | Global redaction covers logs and excepthook. Deny-rule setup is in the quickstart. The local metadata file is gitignored. |
| IV | No identity | PASS | Exact-column validation makes identity columns unwritable. Person and face values are counts per segment. |
| V | No judgments | PASS | The label vocabulary is checked against a banned-term list in the tests. Expression labels are limited to the five allowed. |
| VI | Humans verify | PASS | The reviews schema carries source, mode, reviewer, guideline_version and created_at, and is append-only. The `status` group includes uncertain, mixed and needs_review. |
| VII | Contracts first | PASS | This feature is the contract. Mock data covers all six tables. Only `mock*` IDs and names are used. |
| VIII | Measure everything shown | PASS (partial) | The runs table records runtime and hardware. `split` is set per video. Metrics are out of scope for this feature. |
| IX | Simplest thing | PASS | Plain functions and existing deps. The only new dep is pytest. No validation library. |
| X | Presentation-ready | N/A | No UI. Label colors are chosen to show on white. |
| XI | Config over code | PASS | Labels, windows, frames, thresholds, phases, merge rules and per-video overrides are all in YAML. Real metadata is local only. |
| XII | One app, two modes | PASS | Mock output uses the same layout and readers as real runs. |
| XIII | Deliverables | PASS | Mock data feeds the public samples and the structured exports later. |

**Post-design re-check (after Phase 1)**: PASS. No violations, so the Complexity Tracking table is empty.

**Deviations from the tech input** (recorded in [research.md](research.md)):

1. **CUDA 13.0 (cu130) wheels instead of CUDA 12.8 (cu128).** The cu128 index has no torch 2.14.1, and cu130 has every locked version.
2. **PyAV's built-in resampler for audio, not an `ffmpeg` binary.** PyAV ships ffmpeg's libraries, so this produces the same output without a PATH dependency on HPC.
3. **Stdlib `ru_maxrss` instead of psutil RSS.** It gives the true peak rather than current memory, and needs no extra dependency.

## Project Structure

### Documentation (this feature)

```text
specs/001-foundation/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── tables.md
│   ├── config.md
│   └── interfaces.md
└── tasks.md             # /speckit-tasks
```

### Source Code (repository root)

```text
pyproject.toml           # name tpa-index; script tpa-index; cu130 index for Linux; pytest dev group
.gitignore               # + slurm-*.out
config/
├── labels.yaml
├── pipeline.yaml
└── videos.example.yaml  # videos.local.yaml is gitignored
src/tpa_index/           # renamed from src/2026_ai4good_hackathon/
├── __init__.py
├── paths.py             # .env, guard, redact, install_redaction
├── config.py            # load + validate YAML, effective_settings, config_hash
├── schema.py            # pyarrow SCHEMAS, validate, write/read Parquet, reviews DuckDB
├── segment.py           # discover, probe, windows, frames (letterbox), audio
├── runs.py              # track_run, hardware name, peak memory
├── moments.py           # merge_moments
├── mock.py              # seeded generator for all six tables
└── cli.py               # check | mock | segment
scripts/slurm/job.sbatch
tests/
├── conftest.py          # env in tmp_path, synthetic clips
├── test_paths.py        # guard, redaction (incl. third-party exception)
├── test_config.py
├── test_schema.py       # one malformed case per rule, immutability, reviews append-only
├── test_segment.py      # windows math, landscape/portrait letterbox, no-audio
├── test_runs.py
├── test_moments.py
└── test_mock.py         # full dataset validates, reproducible by seed
```

**Structure Decision**: a single src-layout package with one module per spec area and flat tests, since it's 3 people and 48 hours. The CLI calls module functions directly.

## Complexity Tracking

No violations.
