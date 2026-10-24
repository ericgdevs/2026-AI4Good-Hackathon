# Implementation Plan: Evaluation and Program Indicators

**Branch**: `005-insights-eval` | **Date**: 2026-10-24 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/005-insights-eval/spec.md`

## Summary

Two read-only commands on the Mac:
- **`tpa-index evaluate`** compares official model output with blind labels on eval videos. It reports:
  - per-label F1, macro-F1, primary accuracy, confusion and kappa;
  - accuracy on confident vs flagged segments;
  - block-bootstrap 95% CIs and per-video rows;
  - a zero-shot vs zero-shot + signal-rules ablation;
  - a "signal-based indicators" table and a failure gallery.
- **`tpa-index insights`** computes youth-only indicators per video and phase, each tagged staff-verified, model estimate or mixed, with an early-vs-showcase headline.

Both build one per-segment view from `read_official`, `read_signals` and the review log. Visual yes/no decisions use dev-calibrated thresholds through a strict lookup. Audio conditions use a fixed AST probability (`insights.audio_threshold`, 0.5). Every rule is in one `insights:` config block. Each run writes a new result folder (CSV, JSON, charts, `summary.md`), and `manifest.json` written last marks it complete for 006. A missing official stage or threshold stops the command with exit 3 and a list of what is missing.

## Technical Context

**Language/Version**: Python 3.12, uv, package `tpa_index`

**Primary Dependencies**: all already locked: pandas 3.1, DuckDB 2.0 (the review log), scikit-learn 1.9 (`cohen_kappa_score`), matplotlib 3.11 (charts, `Agg`) and numpy. **No new dependency.**

**Storage**: reads official Parquet runs and `reviews.duckdb` under `OUTPUT_DIR`. Writes `OUTPUT_DIR/results/{evaluate|insights}-{ts}-{hex}/` (outside the repo, never overwritten).

**Testing**: pytest with mock runs and mock blind rows (`review.blind_rows`, `append_reviews`) under the `_isolated_env` fixture. No model, no data.

**Target Platform**: the Mac (also runs anywhere with the synced outputs).

**Project Type**: CLI and library (single project).

**Performance Goals**: each command under 30 s for about 12 videos and about 205 labeled segments, 1,000 bootstrap resamples (SC-007). The bootstrap reuses precomputed per-segment arrays.

**Constraints**: aggregates only; no filenames, transcript text, frames or per-person fields; fail loudly on missing inputs; dev-only thresholds.

**Scale/Scope**: about 12 videos, a few thousand segments, about 205 labeled eval segments.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Check | Result |
|---|---|---|
| I. Data local | Reads local synced outputs. Results go to `OUTPUT_DIR` (outside the repo, gitignored). Nothing is uploaded. | ✅ |
| II. Open models, local | No model loaded. Runtime is printed, and a runtime line is added to the manifest. | ✅ |
| III. Assistants never see data | Built and tested on mocks only. Outputs are aggregate numbers and opaque IDs, which Principle III allows. | ✅ |
| IV. No identity | Unit is the segment. No person or face fields. SC-005 test. | ✅ |
| V. No judgments about people | Program-level indicators only. Expressions are "observable, unvalidated". There is no ranking of participants. The Expression Index is labeled "example, unvalidated" and is program-level only. | ✅ |
| VI. Humans verify | Every indicator is tagged staff-verified / model estimate / mixed. Verified labels take precedence. | ✅ |
| VII. Contracts first | Results contract for 006 in `contracts/results.md`. Config and interface contracts are written. | ✅ |
| VIII. Measure everything shown | Eval split only, per-label P/R/F1, macro-F1, support, block-bootstrap CIs, single-labeler limitation. `summary.md` refuses a metric without its interval (SC-008). | ✅ |
| IX. Simplest thing | No classifier (cut). Small rule grammar, no expression engine. Reuses `read_official`, `read_signals`, `latest_state` and `primary_label`. | ✅ |
| X. Presentation-ready | Charts on a white background with legends, TPA palette, tag in title or footnote. | ✅ |
| XI. Config over code | Fusion rules, indicator rules, support cut, bootstrap and gallery size are all in `insights:`. Thresholds come from calibration. | ✅ |
| XII. One app, two modes | App unchanged. 006 consumes the result files. | ✅ |
| XIII. Deliverables | Summary and charts feed the deck and Devpost; the CSV and JSON feed 006. | ✅ |

There are no violations, so no Complexity Tracking table is needed.

## Confirmed defaults from clarify

1. **Category kappa** skips segments with 2 or more human categories. They are counted and reported (`category_kappa_excluded` in `metrics.json`). No ticked category counts as "none".
2. **Applause time** counts each second once, even when the 10 s after-window overlaps the next performance segment (interval union, research R9).

## Project Structure

### Documentation (this feature)

```text
specs/005-insights-eval/
├── plan.md          research.md          data-model.md          quickstart.md
├── contracts/
│   ├── results.md      # result folder layout, files, columns, tags, completion marker (for 006)
│   ├── config.md       # insights: block, rule grammar, validation, strict thresholds
│   └── interfaces.md   # Python API and CLI
└── checklists/requirements.md
```

### Source code: every file added or changed

```text
config/pipeline.yaml          CHANGED  + insights: block (min_support, bootstrap, failure_gallery, applause_after_s, audio_threshold,
                                         primary_like, fusion, indicator_rules)
src/tpa_index/config.py       CHANGED  validate insights: (keys, labels, condition grammar, rule order);
                                         required_threshold() (no fallback)
src/tpa_index/view.py         NEW      check_inputs, segment_view (slices, model calls, flags, signals), human_labels,
                                         verified_labels
src/tpa_index/rules.py        NEW      condition grammar, apply_fusion, indicator_estimates (missing signal = None)
src/tpa_index/learned.py      NEW      features, block_ids (shared with the bootstrap), grouped-CV predictions, final model
src/tpa_index/evaluate.py     NEW      label/primary/category/uncertainty metrics, block_bootstrap, ablation,
                                         signal-indicator table, top-up precision, failure gallery, run_evaluate
src/tpa_index/indicators.py   NEW      staff_led, time classes, youth indicators, tags, headline, optional index,
                                         run_insights
src/tpa_index/report.py       NEW      result folder, CSV/JSON writers, charts (palette, tags), summary.md,
                                         manifest.json last, latest_result
src/tpa_index/cli.py          CHANGED  `evaluate` and `insights` subcommands (--dry-run)

tests/insights_fixtures.py    NEW      mock official runs (segment/predict/moments/audio/people/expression/transcript)
                                         on dev + eval videos with planted cases; mock blind rows; thresholds file
tests/test_config.py          CHANGED  insights: validation; required_threshold raises with the calibrate hint
tests/test_view.py            NEW      slices partition the duration; moment membership; no_call -> uncertain_mixed;
                                         needs_review inherited; human latest state; skip-only excluded;
                                         check_inputs lists every missing stage/threshold
tests/test_rules.py           NEW      each condition type; missing signal -> None; fusion drops and re-picks primary;
                                         indicator rules follow config
tests/test_evaluate.py        NEW      eval-only; F1 matching; uncertain_mixed handling; category kappa skip;
                                         block bootstrap; ablation equal segments; top-up separate; gallery ranking
tests/test_indicators.py      NEW      staff-led exclusion (verified and model); idle separate; no double counting;
                                         applause window counts once; positive argmax on face segments; tags; headline
tests/test_report.py          NEW      folder layout + manifest last; incomplete folder ignored; no filenames/text in
                                         outputs; summary lines carry CIs/tags; chart files exist; dry-run writes nothing
tests/test_cli.py             CHANGED  evaluate/insights exit codes and dry-run output
```

**Structure Decision**: the same single project. The shared per-segment view (`view.py`) is used by both commands, so evaluation and indicators read inputs one way. The rule grammar is its own module because fusion and indicator estimates share it. `report.py` owns every file write, so the leak test has one place to cover.

## Success-criterion tests

| SC | test (file::name) |
|---|---|
| SC-001 eval only | `test_evaluate.py::test_dev_labels_change_nothing` |
| SC-002 staff-only excluded | `test_indicators.py::test_staff_led_stretch_not_in_youth_numbers` (verified and model-rule variants) |
| SC-003 no double counting | `test_view.py::test_slices_partition_duration`, `test_indicators.py::test_time_totals_match_duration` |
| SC-004 every number tagged | `test_indicators.py::test_every_indicator_row_has_tag` |
| SC-005 no filenames/text | `test_report.py::test_no_filename_or_transcript_sentinel_in_any_output` |
| SC-006 reproducible CIs | `test_evaluate.py::test_bootstrap_same_seed_same_numbers` |
| SC-007 under 30 s | `test_cli.py::test_evaluate_and_insights_fast_on_mock_scale` (about 12 videos × 300 segments, 205 labels) |
| SC-008 CI on every summary metric | `test_report.py::test_summary_metric_lines_have_intervals` |
| SC-009 equal segment sets | `test_evaluate.py::test_ablation_rows_same_segments` |
| SC-010 no-call and needs_review | `test_view.py::test_no_moment_is_uncertain_and_needs_review_inherited` |
| SC-011 whole blocks | `test_evaluate.py::test_block_bootstrap_whole_blocks_per_video` |
| SC-012 rules follow config | `test_rules.py::test_person_count_condition_from_config` |
| fail loudly | `test_view.py::test_check_inputs_lists_every_missing_item`, `test_config.py::test_required_threshold_no_fallback` |

## Design notes for tasks

1. **Order**: config block + `required_threshold` → fixtures → `view.py` (`check_inputs`, then segment view) → `rules.py` → `evaluate.py` + `report.py` (evaluate folder) → CLI `evaluate` → `indicators.py` → `insights` folder and CLI → optional index.
2. **Main model**: the most common non-rules model in the official predictions (as `cli._main_model`). AST and EmotiEffLib names come from their official stage runs' `runs.model`.
3. **Top-up**: segments only in the top-up queue are reported precision-only in `topup.csv`, never pooled.
4. **The bootstrap** reuses per-segment arrays (match flags, primaries) so 1,000 resamples take well under a second.

## Learned system (added 2026-10-24)

Real results showed the top-10% thresholds miscalibrated for this footage. Examples: performance at precision 100% / recall 15%, spoken word at 73% / 12%, while dance and modeling_runway over-fire.
- **New module `learned.py`:** standardized logistic regressions on every official per-segment signal (scikit-learn, already locked). Primary is multinomial; activity, category and actor are one-vs-rest with balanced class weights.
- **Third ablation row:** the models are evaluated by grouped 5-fold cross-validation over the bootstrap blocks, so no block is in both train and test, and reported with the same metrics, CIs and signal-based indicators.
- **Final model:** fit on all labels; it predicts every segment and is written to `learned_predictions.csv`.
- **Insights:** refits the same final model (deterministic) when `insights.labels_source: learned`, the default, instead of reading a previous folder. That keeps insights independent of result-folder order.
- **New tests:** no fold shares a block between train and test; same seed gives the same output; no filename or text in outputs.

## Decisions on the flags (2026-10-24)

- **Strict thresholds stay, for visual labels too.** Audio and expression are calibrated on HPC and synced to the Mac.
- **Audio uses a fixed probability, not calibration.** In the `insights:` rules, AST audio labels use `insights.audio_threshold` (default 0.5) instead of the calibrated top-10% value, which put `music` and `speech` near 0.95 and would veto real singing. Expression and visual labels keep calibrated thresholds; the positive expression rate stays argmax.
- **Accepted as limitations in `summary.md`:** weak staff exclusion on unreviewed clips, per-moment review decisions, and wide intervals.

## Flags before /speckit-tasks (original)

1. **Calibrated audio thresholds don't exist on the Mac yet.** Until `calibrate --stage audio` runs on HPC and syncs, both commands stop with exit 3 by design. That's the intended behavior, but it means no real-data run before then.
2. **The visual thresholds must be calibrated too.** The strict lookup also covers the main model's activity, category and actor labels. If any visual label was never calibrated (for example, it had too few dev scores), the commands stop until it is. The fix is a calibrate run, not a fallback.
3. **Actor and category scores are weak** (noted as a limitation). Model-only staff exclusion may remove almost nothing; `time_breakdown.csv` will show that honestly.
4. **Review-mode decisions are moment-level.** They count as verified for every segment in the moment. A reviewer's "accept" on a 2-minute moment therefore verifies about 20 segments at once, which is a reasonable reading but worth stating in the summary.
5. **The block bootstrap with about 205 labels on two videos** gives about 21 blocks, mostly from v04, so the intervals will be wide. That's honest and expected.
