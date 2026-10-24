# Tasks: Evaluation and Program Indicators (005)

**Input**: `/specs/005-insights-eval/` (plan.md, spec.md, research.md, data-model.md, contracts/results.md, contracts/config.md, contracts/interfaces.md, quickstart.md)

**Tests**: required. Every success criterion (SC-001 to SC-012) has its own test task.

**Phase order** follows the requested checkpoints:
- Setup and Foundational first;
- **Checkpoint 1**: `tpa-index evaluate` end to end, covering US1 evaluation and US2 ablation and signal-based indicators;
- **Checkpoint 2**: `tpa-index insights` (US3);
- **Checkpoint 3**: report and chart polish (US4);
- then the optional Expression Index (US5).

**Rules for every task**:
- Never read DATA_DIR, `~/tpa-data`, `/Volumes/T7`, the real review log, `config/videos.local.yaml` or any `config/*.local.yaml`. Tests rely on the autouse `_isolated_env` fixture in `tests/conftest.py`, which already points OUTPUT_DIR, the thresholds file and the official-runs file at temp paths.
- Mock runs and mock blind rows only. No model loaded, never start the app, no ssh or rsync.
- No edits inside `.venv`. No commits and no `git add`.
- Outputs contain aggregates and opaque IDs only. No filenames, no transcript text, no per-person fields.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Add the `insights:` block to `config/pipeline.yaml` exactly as in `specs/005-insights-eval/contracts/config.md`: `min_support: 5`, `bootstrap: {resamples: 1000, block: 10, seed: 2026}`, `failure_gallery: 20`, `applause_after_s: 10`, `audio_threshold: 0.5`, `primary_like`, `fusion`, `indicator_rules`. Keep every existing key unchanged.
- [X] T002 In `src/tpa_index/config.py`, add `_check_insights(d, labels)`, called from `_check_pipeline_labels`. It implements the validation list in `contracts/config.md`:
  - integer and range checks;
  - fusion `label`, `clears` and `primary_in` entries are model-scored visual labels (groups in `visual_groups`);
  - `audio` entries are audio-group labels;
  - `indicator_rules` keys are indicator-group labels;
  - `primary_in` is either the string `primary_like` or a list of activity labels;
  - unknown condition keys are rejected, naming the key;
  - `{indicator: x}` must refer to an earlier rule.

  Each error is a `ConfigError` naming `pipeline.insights.<path>`.
- [X] T003 In `src/tpa_index/config.py`, add `required_threshold(pipeline, model, label, stage_hint="predict") -> float`. It returns `pipeline["label_thresholds"][model][label]`, or raises `ConfigError(f"no calibrated threshold for {label} (model {model}): run tpa-index calibrate --stage {stage_hint} --split dev on HPC, then sync")`. There is never a fallback to `label_threshold`.
- [X] T004 [P] In `tests/test_config.py`, add tests:
  - the shipped `insights:` block loads;
  - each validation rule raises with its path (unknown condition key `{volume: 3}`, `audio: dance`, `clears: music`, `indicator_rules.nope`, `{indicator: voice_projection}` used before it is defined, `audio_threshold: 1.5`, `min_support: 0`);
  - `test_required_threshold_no_fallback`: a missing label raises with the label, the model and "calibrate --stage", even though `label_threshold` is set.
- [X] T005 Create `tests/insights_fixtures.py` with `build_world(tmp_path, monkeypatch, **overrides) -> dict`. It writes mock official runs through `tpa_index.schema.write_table` and lists them in a tmp `official_runs.yaml` (monkeypatching `schema.OFFICIAL_RUNS`):
  - **Videos**: `d01` (dev, phase early), `e01` and `e02` (eval, phases early and showcase), and `x01` (dev, phase unknown). Each is 120 s, with 8 s windows and 2 s overlap, so 20 segments and IDs `{vid}_{start_ms}`. Columns follow `SCHEMAS`.
  - **Per video**, all with stage names matching their folder prefix:
    - a `predict` run: videos, segments with `is_idle`, main-model predictions under model `mock-siglip` for every visual label, and rules rows `uncertain_mixed`;
    - a `moments` run, built with `tpa_index.moments.merge_moments`;
    - `audio` predictions (model `mock-ast`) for the 5 audio labels;
    - `people` segments with `person_count`;
    - `expression` segments with `faces_detected` and expression predictions (model `mock-enet`) for 4 labels;
    - `transcript` segments with `has_speech` and `transcript_chars`.
  - A **thresholds file** at `config.THRESHOLDS_LOCAL` with `label_thresholds: {mock-siglip: {<every visual label>: 0.5}}`.
  - **Planted cases** (described in its docstring), controlled by `overrides`:
    - a staff-led stretch in `e02` (`actor_staff_facilitation` 0.9, `actor_participant` 0.1);
    - an idle stretch in `e01`;
    - a segment in `e01` the model calls `singing` while AST `singing_voice` and `music` are 0.1 (fusion should drop it);
    - one solo performance segment (`person_count` 1, `performance` 0.9);
    - applause right after a performance;
    - faces with `positive` as argmax in some segments;
    - a planted filename `mock/secret_name.mp4` in the videos rows;
    - a `transcript.json` containing the sentinel `ZQXJ-005-SENTINEL` in the transcript run folder.
  - **Blind labels**: written directly (not through `build_main_queues`). Write a `main` blind queue with `tpa_index.schema.append_blind_queue` over 30 eval segments plus 3 dev segments, and blind rows with `tpa_index.review.blind_rows` (reviewer `mock-staff-a`), including:
    - one segment with 2 categories;
    - one with none;
    - one Skip (`review.skip_row`);
    - one label later rejected;
    - 3 segments in a `topup` queue.
  - Returns IDs and the planted segment IDs for assertions.

---

## Phase 2: Foundational (blocks every story)

- [X] T006 Create `src/tpa_index/view.py` with `check_inputs(kind, pipeline, labels, root=None)`.
  - **Scope**: eval videos (from `read_official("videos")`) for `evaluate`, all official videos for `insights`.
  - **Required stages**: `predict` (or `moments`), `audio` and `people` always; `transcript` and `expression` for insights. Each must cover every video in scope, using `schema._official_by_video(load_official(), stage, root)`.
  - **Required thresholds**: for every visual label in the activity, category and actor groups plus every `clears` label in the rules, `required_threshold` must succeed for the main model. The main model is the most common non-rules model in `read_official("predictions")`; reuse `cli._main_model`, or move it to `view.py` and import it in `cli.py`.
  - Collect every problem and raise one `ConfigError` listing them, e.g. `official audio runs missing for: e02; no calibrated threshold for dance (model mock-siglip): run …`.
  - With no eval blind labels, evaluate raises `ConfigError("no blind labels on eval videos")`.
- [X] T007 In `src/tpa_index/view.py`, add `segment_view(pipeline, labels, root=None) -> pandas.DataFrame`, one row per official segment, with the columns in `contracts/interfaces.md`:
  - **slices**: `slice_s = min(end_s, next.start_s) - start_s` per video sorted by `start_s`; the last segment uses `end_s - start_s`. `filename` is never selected.
  - **model calls**: `model_cleared` (dict group → set, from main-model scores ≥ `required_threshold`); `model_primary` and `model_strength` via `tpa_index.score.primary_label` and `strengths` over activity labels.
  - **moments**: moment membership by time containment within the video; `no_call` = in no moment, and then `model_primary = "uncertain_mixed"` and `model_strength = 0`; `needs_review` from the moment.
  - **flags**: `uncertain_flag` = the rules row `uncertain_mixed` ≥ 0.5.
  - **signals**: `person_count`, `faces_detected` and `has_speech` from `read_signals()`.
  - **audio**: `audio_ok` = dict audio label → `score >= pipeline["insights"]["audio_threshold"]`, or None per label when the video has no official audio run.
  - **expression**: `expr_top` = argmax label over the expression predictions, or None when `faces_detected` is 0 or null.
  - **video columns**: `split` and `phase` from `read_official("videos")`.
- [X] T008 In `src/tpa_index/view.py`, add `human_labels(labels, pipeline, root=None) -> pandas.DataFrame`. It uses `review.latest_state(read_reviews(root), mode="blind")` and keeps labels whose latest action is `add`. Per segment:
  - `primary`: the label with role `primary`, or `uncertain_mixed` if that label is present;
  - `activities`, `categories`, `actors`, `indicators`: sets by group;
  - `queue`: `main` if in the main queue (`read_blind_queue("main")`), `topup` if only in the top-up queue;
  - `skipped`: True when only the skip observation is present.
- [X] T009 In `src/tpa_index/view.py`, add `verified_labels(view, labels, pipeline, root=None)`. It returns per segment a dict group → verified set, or None. A blind label (main or top-up) comes first. Otherwise the latest review-mode decision on the containing moment applies:
  - `accept` → the moment's AI label for its group;
  - `edit` → `new_label`;
  - `add` → `new_label`;
  - `reject` → removes the label.

  Groups the labeler didn't fill stay None.
- [X] T010 [P] Create `tests/test_view.py` with tests using `build_world`:
  - `test_slices_partition_duration` (**SC-003 part 1**): the per-video `slice_s` sums equal the duration, and no slice is negative;
  - `test_no_moment_is_uncertain_and_needs_review_inherited` (**SC-010**): a segment outside every moment has `no_call` True and `model_primary == "uncertain_mixed"`; a segment inside a `needs_review` moment has `needs_review` True;
  - `test_check_inputs_lists_every_missing_item`: remove the audio run for `e02` and one visual threshold; one `ConfigError` names both;
  - human-label latest state (a rejected label is gone), the 2-category segment, the Skip segment and top-up queue membership;
  - audio uses the fixed 0.5: an AST score of 0.6 is True even if `thresholds.local.yaml` has a 0.95 audio threshold.
- [X] T011 Create `src/tpa_index/rules.py` with:
  - `evaluate_condition(cond, seg, ctx) -> bool | None` for every condition type in `contracts/config.md`. A missing signal returns None; `all` with any False is False, otherwise None if any is None; `any` with any True is True, otherwise None if any is None, else False.
  - `indicator_estimates(view, rules, ctx) -> DataFrame` with columns `segment_id, indicator, fired` (True, False or None), evaluating `indicator_rules` in file order so `{indicator: x}` can see earlier results.
  - `apply_fusion(view, fusion_rules, ctx) -> (DataFrame, dict)`. For each rule, if the target label is in `model_cleared["activity"]` and `require_any` is False, the label is removed. If it is None, nothing changes and the case counts as "not applicable". The primary is then recomputed from the remaining cleared activities, or `uncertain_mixed` if none is left. Fire counts per rule are `{"fired", "not_applicable"}`.

  `ctx` carries the thresholds, the main model, `audio_threshold` and `primary_like`.
- [X] T012 [P] Create `tests/test_rules.py`:
  - each condition type, True, False and None;
  - the `all`/`any` truth table with None;
  - fusion drops the planted `singing` segment and re-picks its primary;
  - fusion leaves a segment unchanged when audio is missing (counted not applicable);
  - `test_person_count_condition_from_config` (**SC-012**): changing `solo_on_stage`'s `person_count` to `{eq: 2}` in a copied pipeline changes which segments fire, with no code change.

**Checkpoint F**: `uv run pytest tests/test_config.py tests/test_view.py tests/test_rules.py` passes.

---

## Phase 3: `tpa-index evaluate` (US1 + US2), Checkpoint 1

**Goal**: one command writes a complete `evaluate-*` result folder: Part 1 metrics, ablation, signal-based indicators, failure gallery and manifest.
**Independent test**: T013–T026 pass on `build_world`.

- [X] T013 [US1] In `src/tpa_index/evaluate.py`, add `eval_frame(view, human)`. It joins main-sample, non-skipped human labels to eval-split segments only, and counts `skipped`, `not_scored` (labeled but missing in the view) and `dev_labels_ignored`.
- [X] T014 [US1] In `src/tpa_index/evaluate.py`, add `label_metrics(human_sets, model_sets, labels, min_support)`. For each label: support, predicted, tp, fp, fn, precision, recall and f1, with `status="too few to score"` when support is below `min_support` (value None). Add `macro_f1` over the `ok` labels, plus the excluded list. Use it for activity (human activities vs model cleared activities), category and actor.
- [X] T015 [US1] In `src/tpa_index/evaluate.py`, add:
  - `primary_metrics(df)`: accuracy; `accuracy_real_activity` (human primary ≠ `uncertain_mixed`); a confusion dict over activities plus `uncertain_mixed`; `kappa` via `sklearn.metrics.cohen_kappa_score`;
  - `category_kappa(df)`: the human single category, "none" if empty, excluding segments with 2 or more categories and returning their count as `excluded`; the model's top cleared category by strength, or "none";
  - `kappa_sentence(k)` using the bands in research R6.
- [X] T016 [US1] In `src/tpa_index/evaluate.py`, add `uncertainty_metrics(df)`: rates of `uncertain_flag`, `needs_review` and `no_call`; `flagged` = any of the three; accuracy on unflagged and flagged segments; coverage (the unflagged share).
- [X] T017 [US1] In `src/tpa_index/evaluate.py`, add `block_bootstrap(df, stat_fn, block, resamples, seed) -> (low, high, n_valid)`:
  - sort each video's rows by `start_s` and cut them into blocks of `block`;
  - draw blocks with replacement using `numpy.random.default_rng(seed)` until the size reaches the original, then trim;
  - skip resamples where `stat_fn` returns None or NaN;
  - return the 2.5 and 97.5 percentiles.
- [X] T018 [US2] In `src/tpa_index/evaluate.py`, add `run_systems(view, human, pipeline, labels)`. It computes the full metric set for `zero-shot` (the view as is) and for `zero-shot + signal rules` (`rules.apply_fusion`) on the same segment IDs. It returns rows for `metrics.csv` (columns in `contracts/results.md`, `tag="metric"`), CIs from `block_bootstrap` for primary accuracy, the macro-F1s and both kappas, the per-video scope rows, and the fusion fire counts.
- [X] T019 [US2] In `src/tpa_index/evaluate.py`, add `signal_indicator_table(view, human, pipeline, labels)`. It compares `rules.indicator_estimates` with human `indicators` on eval main-sample segments. Per indicator: support, tp, fp, fn, precision, recall, f1, CI (block bootstrap) and `not_available` (fired None), with status "too few to score" below `min_support`.
- [X] T020 [US1] In `src/tpa_index/evaluate.py`, add:
  - `topup_precision(view, human)`: top-up-only segments, per label: predicted_positive, tp, precision;
  - `failure_gallery(df, n)`: wrong primary calls sorted by `model_strength` descending, top n, with columns `rank, segment_id, video_id, model_primary, human_primary, model_strength`;
  - `run_evaluate(pipeline, labels, root=None)`: calls `check_inputs("evaluate", …)`, builds the views and returns a payload dict with every table, the headline numbers, `category_kappa_excluded`, the counts from T013 and the limitations list (FR-013 and the plan decisions).
- [X] T021 [US1] Create `src/tpa_index/report.py`:
  - `new_result_dir(kind, root=None)` creates `OUTPUT_DIR/results/{kind}-{%Y%m%dT%H%M%S}-{secrets.token_hex(2)}`;
  - `write_result(kind, payload, root=None)` writes the CSV and JSON files for that kind with the exact names and columns in `contracts/results.md` (header-only CSV when a table is empty), a plain `summary.md` (the charts come in Phase 5), and `manifest.json` **last** via a temp file + `os.replace`, with sha256 per file, `schema_version: 1`, `git_commit` (`runs.git_commit()`), `config_hash`, guideline versions, official run IDs and limitations;
  - `latest_result(kind, root=None)` returns the newest folder that has `manifest.json`.
- [X] T022 [US1] In `src/tpa_index/cli.py`, add an `evaluate` subcommand with `--dry-run`. It calls `install_redaction`, `check_guard`, then `evaluate.run_evaluate`, and prints:
  - eval segments per video;
  - primary accuracy with CI;
  - macro-F1 per group with CI;
  - both kappas with the sentence;
  - the ablation rows;
  - the result folder path, relative to OUTPUT_DIR and shown as `<OUTPUT_DIR>/results/…`, or "dry run: nothing written".

  A `ConfigError` exits 3.
- [X] T023 [P] [US1] **SC-001 test**: in `tests/test_evaluate.py`, add `test_dev_labels_change_nothing`. Run `run_evaluate` on `build_world`, then add blind labels to dev segments (and change them), run again, and assert every metric value is equal.
- [X] T024 [P] [US1] **SC-006 test**: in `tests/test_evaluate.py`, add `test_bootstrap_same_seed_same_numbers`. Two runs give identical CIs, and a different seed gives different CIs.
- [X] T025 [P] [US1] **SC-011 test**: in `tests/test_evaluate.py`, add `test_block_bootstrap_whole_blocks_per_video`. Using a `stat_fn` that records the segment IDs it receives, every resample is a concatenation of whole blocks (allowing the final trim) of 10 consecutive labeled segments from a single video.
- [X] T026 [P] [US2] **SC-009 test**: in `tests/test_evaluate.py`, add `test_ablation_rows_same_segments`. Both systems have the same `n_segments` and segment ID set, and fusion changes the planted `singing` segment's primary only in the fusion row.
- [X] T027 [P] [US1] In `tests/test_evaluate.py`, add metric-rule tests:
  - per-activity F1 uses primary + secondary human activities;
  - `uncertain_mixed` is absent from per-activity F1 but present in the confusion matrix;
  - category kappa excludes the 2-category segment and reports `excluded == 1`;
  - "too few to score" appears below support 5 and the label is left out of macro-F1;
  - the top-up segments appear only in `topup`;
  - the failure gallery is sorted by strength and holds IDs and labels only;
  - the signal-based indicators table counts `not_available` when `person_count` is missing.
- [X] T028 [P] [US1] In `tests/test_cli.py`, add:
  - `evaluate --dry-run` prints accuracy and kappa lines and creates no `results/` folder;
  - `evaluate` creates exactly one complete `evaluate-*` folder with every file listed in `contracts/results.md` and `manifest.json`;
  - a missing audio run exits 3 with the video ID in stderr.
- [X] T029 **Checkpoint 1**: run `uv run pytest`, which must pass. Then, in a scratch OUTPUT_DIR outside the repo (with `TPA_OFFICIAL_RUNS`, `TPA_THRESHOLDS_FILE` and `TPA_LOCAL_VIDEOS` pointing at scratch files), build the fixture world with a short script and run `uv run tpa-index evaluate`. Confirm the folder contents against `contracts/results.md`.

---

## Phase 3b: Learned system (US2, added after real results)

- [X] T029a [US2] In `src/tpa_index/view.py`, add the raw per-segment scores to `segment_view`: `visual_scores` (every main-model label, null included), `audio_scores` (or None without an audio run), `expr_scores` (or None without faces) and `transcript_chars`.
- [X] T029b [US2] Create `src/tpa_index/learned.py` with:
  - `features`, `block_ids` (moved out of `block_bootstrap`, so both share one block definition);
  - `cross_validated`: grouped K-fold (`GroupKFold`, `min(folds, n_blocks)` splits) with `StandardScaler` + `LogisticRegression` (multinomial primary; balanced one-vs-rest for activity, category and actor), seed from `insights.learned.seed`;
  - `fit_final` and `learned_view`.

  Add `insights.learned` and `insights.labels_source` to `config/pipeline.yaml`, with validation in `config.py`.
- [X] T029c [US2] In `src/tpa_index/evaluate.py` and `report.py`:
  - add the third system "learned (cross-validated on our labels)" to `run_evaluate` (metrics, CIs, kappa, ablation row);
  - add a `system` column at the end of `signal_indicators.csv`, with rows for zero-shot and learned;
  - write `learned_predictions.csv` (final model, every official segment, tag `model estimate (learned)`);
  - add the learned line to the limitations and `summary.md`.
- [X] T029d [P] [US2] Create `tests/test_learned.py`:
  - `test_no_block_in_train_and_test` (every fold's train and test block sets are disjoint, and each test row's block is absent from its training rows);
  - `test_learned_deterministic` (same seed gives identical predictions and metrics);
  - `test_learned_predictions_cover_all_segments_without_text` (every official segment once; no filename or sentinel in `learned_predictions.csv`);
  - the third ablation row has the same `n_segments` as the others.

## Phase 4: `tpa-index insights` (US3), Checkpoint 2

**Goal**: youth-only indicators per video and phase, every number tagged, plus the early-vs-showcase headline.
**Independent test**: T030–T038 pass.

- [ ] T030 [US3] Create `src/tpa_index/indicators.py` with `time_classes(view, verified, pipeline)`, which gives each segment one class: `idle` (`is_idle`), `staff_verified`, `staff_model` or `youth`. The rules are in `data-model.md`; idle wins. It returns the class plus a `verified` flag per segment, True when the label groups the indicator uses are verified.
- [ ] T031 [US3] (Model fallback per `insights.labels_source`: `learned` by default, which refits `learned.fit_final` and uses `learned_view`; or `zero-shot`.) In `src/tpa_index/indicators.py`, add `tag(verified_share)`, returning the exact strings from `contracts/results.md`, and `youth_indicators(view, verified, estimates, pipeline, scope_rows)`. It computes every FR-022 indicator per video, per phase (`early`, `mid` if present, `showcase`) and for `all`, using `slice_s` time over youth segments only. Each row has the `indicators.csv` columns, `verified_share` (verified youth time / contributing time) and its tag.
  - **verified first**: a segment's verified labels replace the model's for the groups that are verified;
  - **solo moments**: contiguous runs of `solo_on_stage` segments (staff tag, else the rule estimate);
  - **audience response**: the union of performance intervals each extended by `applause_after_s`, intersected with applause slices, each second once;
  - **positive expression rate**: over segments with `faces_detected ≥ 1`, using `expr_top == "positive"`, always tagged `model estimate (unvalidated)`;
  - **original work**: staff only, with status "not tagged yet" if none;
  - **missing signal**: status "signal not available";
  - **diversity**: activities with ≥ 60 s and normalized entropy.
- [ ] T032 [US3] In `src/tpa_index/indicators.py`, add:
  - `time_breakdown(view, classes)`: per scope, `youth_s`, `staff_led_verified_s`, `staff_led_model_s`, `idle_s` and `total_s`;
  - `headline(rows, videos)`: early vs showcase, plus `mid` if present, for the four headline indicators, with `change_vs_early`. `unknown` videos are listed under `excluded_videos`, and a missing phase gives status "no videos in this phase";
  - `run_insights(pipeline, labels, root=None)`: calls `check_inputs("insights", …)` and builds the payload with the limitations list.
- [ ] T033 [US3] In `src/tpa_index/report.py`, extend `write_result` for kind `insights`: `indicators.csv`, `indicators.json`, `headline.csv`, `time_breakdown.csv`, a plain `summary.md` and `manifest.json` last. In `src/tpa_index/cli.py`, add an `insights` subcommand with `--dry-run`, printing the headline with tags and the time breakdown for `all`.
- [ ] T034 [P] [US3] **SC-002 test**: in `tests/test_indicators.py`, add `test_staff_led_stretch_not_in_youth_numbers`, in two variants:
  - the planted model staff stretch in `e02` (no verified label);
  - a verified blind label marking an `e01` segment staff-led.

  Each adds 0 s to every youth numerator and denominator and appears in `staff_led_model_s` / `staff_led_verified_s`.
- [ ] T035 [P] [US3] **SC-003 test (part 2)**: in `tests/test_indicators.py`, add `test_time_totals_match_duration`. Per video, `youth_s + staff_led_verified_s + staff_led_model_s + idle_s == total_s == duration` (within 1e-6), and no share exceeds 1.
- [ ] T036 [P] [US3] **SC-004 test**: in `tests/test_indicators.py`, add `test_every_indicator_row_has_tag`. Every row of `indicators.csv` and `headline.csv` has a non-empty tag from the allowed set, and `verified_share` is in [0, 1]. Mixed tags match `mixed \(\d{1,2}% staff-verified\)`.
- [ ] T037 [P] [US3] In `tests/test_indicators.py`, add rule tests:
  - applause counts each second once when the 10 s window overlaps the next performance segment;
  - positive expression uses only face segments and the argmax, and is always tagged model estimate;
  - original work reads "not tagged yet" without staff tags;
  - `unknown` video `x01` is excluded from the headline and listed;
  - youth voice time includes `has_speech` and excludes staff-led time;
  - solo moments counts contiguous runs.
- [ ] T038 [P] [US3] In `tests/test_cli.py`, add:
  - `insights --dry-run` writes nothing;
  - `insights` writes a complete `insights-*` folder;
  - a missing expression run exits 3.
- [ ] T039 **Checkpoint 2**: run `uv run pytest`, which must pass. Then, in the T029 scratch setup, run `uv run tpa-index insights` and check the folder against `contracts/results.md`.

---

## Phase 5: Report and chart polish (US4), Checkpoint 3

**Goal**: slide-ready charts and summary, and a proven-clean output.

- [ ] T040 [US4] In `src/tpa_index/report.py`, add charts with matplotlib (`matplotlib.use("Agg")`): white background, text color `#201C25`, series colors in order `#7851B9`, `#0099B3`, `#FF4255`, legends on, 150 dpi PNG.
  - **evaluate**: `ablation.png` (primary accuracy and macro-F1 per system, with CI error bars), `per_label_f1.png` (scorable activity labels), `confusion.png` (white → `#7851B9` single hue, counts annotated) and `coverage.png` (accuracy on confident vs flagged, with coverage).
  - **insights**: `headline.png` (grouped bars by phase) and `time_breakdown.png` (stacked youth, staff-led and idle per phase).

  Every chart's title or footnote carries its tag: `metric · 95% CI · n=…` for evaluate, and the indicator's tag for insights. Add the chart files to the manifest.
- [ ] T041 [US4] In `src/tpa_index/report.py`, make `summary.md` slide-ready:
  - **evaluate**: eval-set size per video, the headline metrics, each as `value (95% CI low–high, n=…)`, the kappa sentence, the ablation table, the signal-based indicators table, and a "Limitations" section listing single labeler, the eval size and v04 share, weak staff exclusion, the fixed 0.5 audio probability, per-moment review decisions and wide intervals;
  - **insights**: the headline table with tags, the time breakdown, "observable, unvalidated" on expression, and the same limitations.
- [ ] T042 [P] [US4] **SC-005 test**: in `tests/test_report.py`, add `test_no_filename_or_transcript_sentinel_in_any_output`. Run both commands on `build_world`, then read every file in both result folders as bytes. `secret_name` and `ZQXJ-005-SENTINEL` appear 0 times. No CSV or JSON column or key matches `(?i)(filename|path|person_id|face_id|track|embedding|transcript_text)`.
- [ ] T043 [P] [US4] **SC-008 test**: in `tests/test_report.py`, add `test_summary_metric_lines_have_intervals`. Every line in the evaluate `summary.md` that reports an accuracy, F1 or kappa value contains `95% CI` and `n=`, and every number line in the insights `summary.md` contains a tag string.
- [ ] T044 [P] [US4] In `tests/test_report.py`, add:
  - `manifest.json` is the newest file in the folder, and its sha256 values match the files;
  - a folder without `manifest.json` is ignored by `latest_result`;
  - chart PNGs exist and are non-empty.
- [ ] T045 [P] [US4] **SC-007 test**: in `tests/test_cli.py`, add `test_evaluate_and_insights_fast_on_mock_scale`. Use `build_world` scaled to 12 videos of 300 segments with 205 eval labels (an `overrides` option); `evaluate` and `insights` each finish in under 30 s.
- [ ] T046 **Checkpoint 3**: run `uv run pytest`, which must pass. In the scratch setup, open the generated PNGs and `summary.md` and check the palette, the tags and the CIs by eye.

---

## Phase 6: Optional TPA Expression Index (US5), after Checkpoint 3 only

- [ ] T047 [US5] In `src/tpa_index/indicators.py`, add `expression_index(headline_rows)`. Each of the four headline indicators is scaled to 0–100: shares × 100; solo moments and youth voice as a share of youth time × 100. They are averaged with equal weights per phase, and the result is written as an `indicators.csv` row with `indicator=expression_index`, `unit=index_0_100`, `tag="model estimate (unvalidated)"`, `status="example"`, with its four components in `indicators.json`. Program level only, no per-video ranking.
- [ ] T048 [P] [US5] In `tests/test_indicators.py`, add `test_expression_index_reproducible_from_components`. The index equals the mean of its listed components, the row is tagged as an example, and it never appears as a video-level ranking.

---

## Phase 7: Polish

- [ ] T049 [P] Update `specs/005-insights-eval/contracts/interfaces.md` and `contracts/results.md` with any signature or column changes made during implementation, and bump `schema_version` if a column was removed or renamed.
- [ ] T050 Final hygiene:
  - run `uv run pytest`;
  - `git status --short` shows no data, `*.parquet`, `*.png`, `*.duckdb` or `*.local.yaml` files in the repo (the result folders live under OUTPUT_DIR);
  - grep the new modules for clinical or feelings wording;
  - confirm nothing under `.venv` changed.

---

## Dependencies and execution order

- **Phase 1**: T001 → T002 → T003. T004 follows T002 and T003. T005 can run alongside T002–T004.
- **Phase 2**: T006 → T007 → T008 → T009. T011 follows T007. T010 follows T009, and T012 follows T011.
- **Phase 3**: T013 → T014 → T015 → T016 → T017 → T018 → T019 → T020 → T021 → T022. The tests T023–T028 follow T022; T029 is last.
- **Phase 4** needs Checkpoint 1 (it shares `view.py` and `report.py`): T030 → T031 → T032 → T033, then T034–T038, then T039.
- **Phase 5**: T040 → T041, then T042–T045, then T046.
- **Phase 6** comes after Checkpoint 3 and is optional.
- **Phase 7** is last.

## Parallel opportunities

- T004 and T005 alongside T002 and T003.
- T010 and T012 alongside each other once their modules exist.
- T023–T028 together (separate test functions or files).
- T034–T038 together.
- T042–T045 together.

## Implementation strategy

1. **MVP is Checkpoint 1**: `evaluate` produces the required Part 1 deliverable with CIs and the ablation. Ship that first.
2. **Checkpoint 2** adds the TPA indicators on the same view.
3. **Checkpoint 3** makes the outputs slide-ready and proves them clean.
4. The Expression Index only if time remains.
