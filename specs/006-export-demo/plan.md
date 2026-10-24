# Implementation Plan: Official Export, Judge Demo and Mac Benchmark

**Branch**: `006-export-demo` | **Date**: 2026-10-24 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/006-export-demo/spec.md`

**Gate**: implementation starts only after 005-insights-eval is merged into `main` and this branch is rebased onto it. Phases A–C don't read any 005 output and can be built first.

## Summary

Five additive pieces on top of 001–004, all reading through the existing official read paths:

1. **Export**: `tpa-index export` builds one record per official moment from `review.latest_state` with one pure function. It writes a timestamped JSON/CSV pair plus `annotations-latest.*` under `OUTPUT_DIR/exports/`. `--verified-only` drops rejected moments. `--sample` writes the checked-in Devpost sample from a temporary mock dataset with scripted reviews.
2. **Mac benchmark**: `tpa-index benchmark --video-id X` runs the existing seven stages in-process. Each already records runtime and peak memory. It then writes a Mac vs HPC per-video-hour summary (per stage plus end to end) under `OUTPUT_DIR/benchmarks/`. It never touches `official_runs.yaml`.
3. **Theme**: a new `config/theme.yaml` holds the TPA palette and credit line. It is validated, including a WCAG AA contrast check, and the Gradio theme and the charts read it. A new `config/guideline_history.yaml`, plus `check_guideline`, refuses a vocabulary change without a guideline bump in `check` and at app launch.
4. **Timeline tab**: per-video moments plus per-segment signals (people, speech yes/no, audio cues, expression marked "AI suggestion, unvalidated", led by). It uses an allow-list view model with no transcript field. Staff-led segments follow 005's rule and are hidden by default.
5. **Insights tab**: reads the newest complete `results/insights-*` folder (005 contract). It draws four early-vs-showcase charts with matplotlib on white and shows every tag verbatim.

## Technical Context

**Language/Version**: Python 3.12, uv, package `tpa_index` (src layout)

**Primary Dependencies**: all already locked. Gradio 6.30 (app, theme `.set()` variables), matplotlib (charts, already locked and also used by 005), pyarrow/DuckDB (existing reads), the standard library `csv`/`json`/`hashlib`. **No new dependencies.**

**Storage**: no schema change. New files:
- outside the repo: `OUTPUT_DIR/exports/*`, `OUTPUT_DIR/benchmarks/*`;
- tracked: `config/theme.yaml`, `config/guideline_history.yaml`, `docs/samples/annotation_export_sample.{json,csv}`, `tests/fixtures/results/insights-…/` (mock 005 folder).

**Testing**: pytest with the existing `conftest` isolation (temporary DATA_DIR, OUTPUT_DIR, official runs, no local videos or thresholds). Mock data, synthetic clips and fake signal models only. No test starts a server; the UI is tested through the controllers and `build_app` config, as in 003.

**Target Platform**: macOS (MPS/CPU) for the app, export and benchmark. The export also runs anywhere the reviews DB and official runs exist.

**Project Type**: single Python package with a CLI and a local Gradio app.

**Performance Goals**:
- export of all official moments (about 12 videos) in under 10 s;
- Timeline tab opens a video in under 3 s;
- Insights tab loads in under 2 s (SC-008).

**Constraints**:
- 127.0.0.1 only;
- light theme;
- no transcript text, filenames or identity on any screen or in any file;
- assistants never read real data;
- changes to `cli.py` and `pipeline.yaml` stay small and additive (`pipeline.yaml` is not changed at all).

**Scale/Scope**: about 12 videos, a few hundred moments, about 205 blind-labeled segments. The benchmark covers one promo.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Pre | Post |
|---|---|---|---|
| I Data local | Exports and benchmark summaries go to OUTPUT_DIR (guarded outside the repo). Only the mock sample is tracked. | ✅ | ✅ |
| II Open models, efficient local | The benchmark records runtime and peak memory per stage on the Mac and compares them with HPC: the 16 GB-laptop proof. | ✅ | ✅ |
| III Assistants never see data | Tests use mock data and temporary dirs only. The quickstart separates assistant-safe steps (1–5) from the team-only step (6). Fixtures live in `tests/fixtures/`, not `tests/data/`. | ✅ | ✅ |
| IV No identity | Segment-level aggregates only. No face crops. Reviewer names are left out of the export. | ✅ | ✅ |
| V No judgments | Expressions are shown only as "AI suggestion, unvalidated". No clinical wording. Indicators are program-level, from 005. | ✅ | ✅ |
| VI Humans verify | The export carries `source`, `review_decision`, `guideline_version` and `review_date`. The timeline shows confidence and review status. | ✅ | ✅ |
| VII Contracts first | The export format, CLI and UI contracts are in `contracts/`. 005 is consumed through its `results.md` contract only. Mock data drives every test. | ✅ | ✅ |
| VIII Measure everything shown | Insights numbers come from 005 with tags (and intervals for metrics). Benchmark numbers carry hardware and model. The timeline shows raw signals, not metrics. | ✅ | ✅ |
| IX Simplest thing | Reuses `latest_state`, `read_official`, `read_signals`, `track_run`, the existing commands and matplotlib. No training. | ✅ | ✅ |
| X Presentation-ready | The forced-light theme is kept. Charts on white with legends. The palette passes AA (SC-010). | ✅ | ✅ |
| XI Config over code | The palette is in `theme.yaml`, observation types in `labels.yaml`. Guideline history is config. The three staff-led rule label names are constants checked against `labels.yaml` at startup (see Complexity Tracking). | ✅ | ✅ |
| XII One app, two modes | Timeline and Insights are tabs in the same app, using the same read paths as Review. There is no separate mock UI. | ✅ | ✅ |
| XIII Deliverables | The structured export (JSON/CSV) and a public sample from mock data are produced by the repo. | ✅ | ✅ |

No violations. One deliberate shortcut is recorded under Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/006-export-demo/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── export-format.md
│   ├── cli.md
│   └── ui.md
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks (not created here)
```

### Source Code: every file added or changed

```text
config/
├── theme.yaml                    # NEW  palette + credit (FR-031a-d)
├── guideline_history.yaml        # NEW  {"0.5.0": "<fingerprint>"} (FR-026)
├── labels.yaml                   # unchanged (optional `display:` keys only if wanted later; not in fingerprint)
└── pipeline.yaml                 # unchanged (keeps the 005 merge clean)

src/tpa_index/
├── export.py                     # NEW  build_records (pure), write_export, sample_reviews, write_sample
├── benchmark.py                  # NEW  run_benchmark (in-process cli.main per stage), summarize, write_summary
├── demo.py                       # NEW  display_label, video_name, staff_led, DemoController, InsightsController,
│                                 #      headline figures (matplotlib)
├── config.py                     # CHANGE  + load_theme (with contrast check), guideline_fingerprint,
│                                 #         check_guideline; labels check allows optional `display`
├── cli.py                        # CHANGE  + `export`, `benchmark` subparsers (one block before parse_args)
│                                 #         + handlers; `_check` calls check_guideline
└── app.py                        # CHANGE  theme() from theme.yaml; header/footer credit; Timeline and Insights
                                  #         tabs; launch() calls check_guideline; review panel readable names

docs/samples/
├── annotation_export_sample.json # NEW  mock only (~20 records)
└── annotation_export_sample.csv  # NEW

tests/
├── test_export.py                # NEW  fields/order, decision table, verified-only, files never overwritten,
│                                 #      CSV escaping, sample regenerates identically + no filenames/sentinel
├── test_benchmark.py             # NEW  fake stages: run IDs collected, failure continues, per-video-hour math,
│                                 #      "no HPC run", official_runs.yaml untouched
├── test_theme.py                 # NEW  palette loads, AA contrast, hex values only in theme.yaml, credit shown
├── test_guideline.py             # NEW  fingerprint stable; new obs type without bump refused; with bump ok;
│                                 #      blind checklist identical; obs picker shows new type
├── test_demo.py                  # NEW  timeline signals vs mock, "not available", staff-led table (005 cases),
│                                 #      hidden by default, no transcript sentinel/count, no filenames, video names
├── test_insights.py              # NEW  newest complete folder, ignores no-manifest, schema_version guard, tags
│                                 #      verbatim, mid only if present, status rows, empty state
├── fixtures/results/insights-20261024T213040-3c4d/   # NEW  mock 005 folder: manifest.json, headline.csv,
│                                                     #      indicators.csv, time_breakdown.csv
└── test_app.py                   # CHANGE  tab list includes Timeline/Insights; light theme + credit in config

CLAUDE.md                         # CHANGE  via update-agent-context.sh (006 entry)
README.md                         # CHANGE  commands: export, export --sample, benchmark
```

**Structure Decision**: Keep the existing single package. The new logic lives in three new modules (`export`, `benchmark`, `demo`) so that `cli.py` and `app.py` only gain thin wiring. That keeps the 005 merge to neighbouring lines in `cli.main()`.

## Build Order

The parts that don't depend on 005 come first. Each phase is independently testable and ends with the full test suite green.

| Phase | Scope | Depends on 005? | Spec |
|---|---|---|---|
| **A. Export + Devpost sample** | `export.py`, `cli export`, the sample files, `test_export.py` | no | US1, FR-001–014, SC-001–003 |
| **B. Mac benchmark** | `benchmark.py`, `cli benchmark`, `test_benchmark.py` (fakes). The team runs it on a real promo afterwards. | no | US5, FR-027–030, SC-007 |
| **C. Theme + guideline check** | `theme.yaml`, `load_theme`, `app.theme()`, credit, `guideline_history.yaml`, `check_guideline`, `test_theme.py`, `test_guideline.py` | no | US4, US6, FR-024–026, FR-031–032, SC-006, SC-010 |
| **D. Timeline tab** | `demo.py` (DemoController, staff_led, names), the Timeline tab, `test_demo.py` | the staff-led rule must match 005's final rule | US2, FR-015–019, SC-004–005, SC-009 |
| **E. Insights tab** | InsightsController, figures, the Insights tab, fixtures, `test_insights.py` | yes (contract `results.md`, schema_version 1) | US3, FR-020–023, SC-008 |

Final step: README commands, the agent context, and a full `uv run pytest`.

## Flags Before Tasks

1. **The benchmark needs the promo in the official runs.** The 004 stages and `moments` use the official segments and predict output for the video (`signals.official_segments`, `read_official`). If the chosen promo has no official segment/predict run synced to the Mac, those stages fail. The summary marks them, but the proof is incomplete. Pick a promo that is already official.
2. **The Mac per-hour figure is pessimistic on a short promo.** Model load time is amortised over a few minutes of video. The summary says so. Optionally, also report HPC for the same video's official runs.
3. **The staff-led rule is restated, not shared.** 006 can't import 005 code, and 005's files are aggregates. `demo.staff_led` copies the rule (Clarification 2). Its tests mirror 005's cases, and after the merge it should switch to 005's function if one exists. If 005's Q1 settles differently, D changes.
4. **The eval lock carries over.** The Timeline tab uses `review_videos()`, so eval videos stay hidden until blind labeling is finished, exactly like Review. For the judge demo, pick a dev video or finish blind labeling first.
5. **Guideline check placement.** `check_guideline` runs in `tpa-index check` and at app launch, not inside `load_labels()`, because three existing tests load renamed vocabularies without a bump. A new observation type with a new version is allowed but unrecorded. `check` prints the line to add to `config/guideline_history.yaml`, and staff must add it to protect that version.
6. **Readable label names are automatic** ("Lyricism spoken word", "Actor staff facilitation"). Nicer wording ("Lyricism / spoken word", "Staff-led") needs optional `display:` keys in `labels.yaml`. These are left out of the guideline fingerprint, but they are still a `labels.yaml` edit. Decide in tasks whether to add them.
7. **005 headline names.** The insights page expects `solo_moments_count` (and optional `solo_moments_minutes`) in `headline.csv`, from `results.md`'s indicator list. If 005's headline uses a different solo indicator ID, only the constant in `demo.py` changes.
8. **Review tab readable names.** Using display names in Review's AI-labels panel touches 003 UI tests. If that's too risky before the deadline, limit readable names to Timeline and Insights. Blind labeling stays unchanged either way.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected because |
|---|---|---|
| The staff-led rule's three label names are constants in `demo.py` (checked against `labels.yaml` at startup), not config | keeps `pipeline.yaml` untouched until 005's `insights:` block lands; then it can read them from there | putting them in `pipeline.yaml` now creates a merge conflict with 005 in the same file |
| The staff-led rule is duplicated from 005 | 005 is file-coupled, and its outputs have no per-segment staff-led flag | importing 005 code breaks the "files only" rule and the build order (D before 005 merge) |
