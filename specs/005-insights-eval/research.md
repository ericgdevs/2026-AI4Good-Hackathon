# Research: Evaluation and Program Indicators (005)

Checked against the code on `005-insights-eval` and the locked packages. No data was read.

## R1. Dependencies

- **Decision**: no new dependency. Everything needed is already locked: pandas 3.1, DuckDB 2.0, scikit-learn 1.9 (`cohen_kappa_score` only), matplotlib 3.11 (charts, `Agg` backend) and numpy.
- **Rejected**: seaborn or plotly for charts (new dependency); statsmodels for intervals (a bootstrap is a few lines of numpy).

## R2. Inputs and where they come from

| input | read with | used by |
|---|---|---|
| videos (split, phase, duration) | `read_official("videos")` | both |
| segments (times, `is_idle`) + 004 columns | `read_signals()` | both |
| visual predictions + rules rows (`uncertain_mixed`) | `read_official("predictions")` | both |
| moments (`needs_review`, membership) | `read_official("moments")` | both |
| audio predictions | `read_official("predictions", stage="audio")` | fusion, voice, applause, voice_projection |
| expression predictions | `read_official("predictions", stage="expression")` | positive expression rate |
| blind labels | `read_reviews()` with `mode == "blind"`, `target_type == "segment"` | evaluate, verified labels |
| review-mode decisions | `read_reviews()` with `mode == "review"`, `target_type == "moment"` | verified labels (insights) |
| blind queues | `read_blind_queue("main")`, `read_blind_queue("topup")` | which labels are main sample vs top-up |
| thresholds | `load_pipeline()` (merges `thresholds.local.yaml`) | every yes/no decision |

The review database path comes from `OUTPUT_DIR` (`schema._reviews_db`), which is the existing override. Tests point it at a temp folder through the `_isolated_env` fixture.

## R3. Fail loudly on missing inputs (no silent fallback)

- **Today**: `config.threshold()` falls back to the global `label_threshold` (0.5) when a label has no calibrated value. That is fine for moments, but wrong for 005: an AST or EmotiEffLib label at 0.5 would silently mean something different.
- **Decision**: add `required_threshold(pipeline, model, label)`. It raises `ConfigError("no calibrated threshold for <label> (model <m>): run tpa-index calibrate --stage <stage> --split dev on HPC, then sync")`. Every 005 yes/no decision uses it.
- **Labels that need a calibrated threshold**: visual activity, category and actor labels, under the main model.
- **Audio labels use a fixed probability instead**: `insights.audio_threshold`, default 0.5 (decision 2026-10-24). AST outputs real probabilities, and the calibrated top-10% cut put `music` and `speech` near 0.95, which would veto real singing in fusion.
- **Expression labels need none**, because the positive rate uses argmax (clarify Q10).
- **Required official stages**: `predict` (or `moments`), `audio` and `people` are always required. `transcript` and `expression` are required for insights only. Each must cover every video in scope (eval videos for evaluate, all mapped videos for insights). If not, raise `ConfigError("official <stage> runs missing for: v01, v03")`, exit 3, before any output is written.
- All checks run in one pass and list every missing item together, so a single run tells the whole story.

## R4. Model calls per segment

- **Cleared labels**: visual labels with `score >= required_threshold` under the main model (the most common non-rules model, as in `cli._main_model`).
- **Model primary**: the cleared activity with the highest strength (`score.primary_label`).
  - If the segment is in **no moment**, it is "no call" and the primary is `uncertain_mixed` (clarify).
  - If **nothing clears**, the primary is also `uncertain_mixed`. Such segments are never in a moment anyway.
- **Flags**:
  - `uncertain_flag` = the rules row `uncertain_mixed` ≥ 0.5, which also fires when the top two activities are close;
  - `needs_review` = the containing moment's flag;
  - `flagged` = `uncertain_flag` OR `needs_review` OR no call.

  A close call keeps its top activity as the primary but counts as flagged. That is exactly the "accuracy on confident vs flagged" split.
- **Moment membership**: same video and `moment.start_s <= seg.start_s` and `seg.end_s <= moment.end_s`. Moments of one video never overlap, because merge rules close a moment before opening the next.

## R5. Human labels per segment

- **Latest state**: per (segment, label) over blind rows (`review.latest_state`, which already exists). A label counts when its latest action is `add`.
- **The parts**:
  - **primary** = the label with role `primary`, or `uncertain_mixed` if that label is present;
  - **activities** = every activity-group label (primary plus secondary);
  - **categories**, **actors** and **indicators** = their groups' labels.
- **Main sample vs top-up**: by queue membership (`blind_queue`). Segments only in the top-up queue go to the separate precision-only report.
- **Skip-only segments** (only `clip_unavailable`): excluded and counted as "skipped".
- **Pooling**: guideline versions 0.4.0 and 0.5.0 are pooled (identical checklist, tested in 004).

## R6. Matching and metrics (clarify Q2)

- **Per-label F1** (activity, category, actor): human set vs model cleared set, per segment. TP, FP and FN are counted directly; it's a few lines, no library needed. Support below 5 gives "too few to score", and the label is left out of macro-F1.
- **Primary accuracy and confusion matrix**: over activity labels plus `uncertain_mixed`.
- **Kappa**: `sklearn.metrics.cohen_kappa_score` on primary (with `uncertain_mixed` as a class) and on category.
  - The human category is the single ticked category, or "none" if none was ticked. Segments with 2 or more categories are excluded from category kappa and counted. **Confirmed default.**
  - The model category is the top cleared category by strength, or "none".
- **Plain-English line for kappa**, by band:

  | kappa | sentence |
  |---|---|
  | < 0 | worse than chance |
  | 0–0.2 | slight agreement beyond chance |
  | 0.2–0.4 | fair agreement |
  | 0.4–0.6 | moderate agreement |
  | 0.6–0.8 | substantial agreement |
  | > 0.8 | near-complete agreement |

  These are the standard Landis–Koch bands, written in plain words.

## R7. Block bootstrap (clarify Q5)

- Each eval video's labeled main-sample segments are sorted by `start_s` and cut into blocks of 10 (the last block may be shorter).
- One resample draws blocks with replacement until it holds at least as many segments as the original, then trims to that size. Every statistic is recomputed on the resample.
- 1,000 resamples, seed from config (`insights.bootstrap.seed`, default 2026). The interval is the 2.5–97.5 percentile.
- A statistic that is undefined on a resample is skipped for that resample. For example, kappa with a single class. The number of valid resamples is reported.

## R8. Fusion and indicator rules (one config block)

- **Decision**: one new `insights:` block in `config/pipeline.yaml`, validated by `config.load_pipeline()`.
- **Why**: it reuses the existing loader, validation, the `thresholds.local.yaml` merge and the test fixtures. A separate file would need its own loader and tests.
- **Rule grammar**, deliberately small: each rule names a target label, an `action` (`require`), and `any_of` conditions. A condition is one of:
  - an audio label at or above its calibrated threshold;
  - `has_speech`;
  - a `person_count` comparison;
  - a visual label clears;
  - a model primary in a list.

  Indicator rules use the same condition types. No expression language and no `eval`.
- **Thresholds**: visual conditions use `required_threshold` (dev-calibrated). Audio conditions use the fixed `insights.audio_threshold` (0.5). Apart from that and `person_count` comparisons, the rules carry no numbers.

## R9. Time base and the applause window

- **Segment time slice**: `[start_s, min(end_s, next.start_s))`, and the last segment is `[start_s, end_s)`. The slices partition the video exactly, so totals equal the duration and overlap is never counted twice.
- **Applause window (confirmed default)**: build the union of performance intervals, each extended by 10 s after its slice end. Applause time is the length of (applause slices ∩ that union), computed on interval lists. Each second counts once, even when a 10 s window overlaps the next performance segment.
- **Exclusions**: idle and staff-led slices are removed from youth denominators before any share is computed.

## R10. Verified labels for indicators

- **Order of precedence**:
  1. A segment's blind label (latest state) is verified for the groups the labeler filled in.
  2. Else, a review-mode decision on the containing moment applies to every segment in it. `accept` keeps the AI label, `edit` replaces it, `reject` removes it, and `add` adds a label. All are staff-verified.
  3. Else, the model call (model estimate).
- **Tag**: per indicator value, `verified_share` = verified time / contributing time.

  | `verified_share` | tag |
  |---|---|
  | 1.0 | `staff-verified` |
  | 0.0 | `model estimate (unvalidated)` |
  | anything else | `mixed (NN% staff-verified)` |

- **Always model estimate**, whatever the labels: positive expression rate, and the signal-based indicators when not staff-tagged.

## R11. Charts

- matplotlib with the `Agg` backend, white figure and axes background, text `#201C25`. Series colors in order: `#7851B9`, `#0099B3`, `#FF4255`. The confusion matrix uses a single-hue white → `#7851B9` scale.
- Every chart carries its tag in the title or footnote, and metric charts show 95% intervals as error bars.
- **Charts**:
  - `ablation.png`: primary accuracy and macro-F1 per system, with CIs;
  - `per_label_f1.png`: activities, scorable labels only;
  - `confusion.png`;
  - `coverage.png`: accuracy on confident vs flagged;
  - `headline.png`: early vs showcase (vs mid);
  - `time_breakdown.png`: youth, staff-led and idle time per phase.

## R12. Result folders and the 006 contract

- **Location**: `OUTPUT_DIR/results/{kind}-{YYYYmmddTHHMMSS}-{4 hex}/`, where kind is `evaluate` or `insights`. Never overwritten.
- **Completion marker**: files are written into the folder, and `manifest.json` is written **last**, through a temp file plus `os.replace`. A folder without `manifest.json` is incomplete and readers ignore it, the same pattern as `runs.parquet` for runs.
- **Manifest contents**: kind, created_at, git_commit, config_hash, the official run IDs used, the list of files with their sha256, and the schema version.
- Full contract: `contracts/results.md`.

## R13. Leak checks (SC-005)

- **Inputs are already clean**: segment IDs and video IDs are opaque, `read_official("videos")` carries `filename` but 005 drops the column on read, and transcript text never enters (only `has_speech` and `transcript_chars` are read).
- **Test**: a planted mock filename and a transcript sentinel (written into a mock `transcript.json`) must appear in no result file.
