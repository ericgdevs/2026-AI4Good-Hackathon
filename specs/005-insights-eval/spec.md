# Feature Specification: Evaluation and Program Indicators

**Feature Branch**: `005-insights-eval`

**Created**: 2026-10-24

**Status**: Clarified

**Input**: User description: "005-insights-eval: evaluate the pipeline against our blind labels and compute TPA's program-level indicators, using only aggregate, youth-only numbers."

## Context

- `main` holds 001–004, the shared blind-queue fix and the job B speed-ups. Official runs for segment, predict, moments, transcript, audio, people and expression are listed in `config/official_runs.yaml`. The default official read returns segments, predictions and moments. The signals read joins `has_speech`, `transcript_chars`, `person_count` and `faces_detected` onto segments. Audio and expression predictions are read per stage.
- **Blind labels** are the review log with mode `blind`. They come from the main sample of eval-split segments (v04 and v07; v04 holds about 94% of eval segments). About 205 main-sample segments are labeled. Guideline 0.4.0 and 0.5.0 blind labels form one eval set, because the checklist is identical (tested in 004).
- **How a blind label is stored**: one row per chosen label. The primary activity has role `primary`. Other activities, category and actor are `secondary`. Indicators are `indicator`. "Uncertain / mixed" is the rules label `uncertain_mixed`. A later save can `reject` a label. Labels are the latest state per (segment, label).
- **Labeler conventions**: `podcast_interview` was used for youth interviews to camera. `uncertain_mixed` was used for bows, thank-yous, hosting, setup, and instructors just talking.
- **Single labeler**: nobody labeled the overlap set, so there is no human-vs-human agreement. This is reported as a limitation.
- **Phase per video** already exists. Each video has `phase` ∈ {early, mid, showcase, unknown} in the local, gitignored videos config, copied into the `videos` table by the segment stage. No filename or phase appears in the repo.
- **TPA guidance (2026-10-24)**: TPA cares about the youth, not the instructors. Every TPA-facing indicator is youth-only. The actor and instruction labels exist to exclude staff time, which the case also requires.
- **Runs on the Mac**: everything reads synced Parquet runs and the local review log, and finishes in seconds. No GPU and no model is loaded.

## Clarifications

### Session 2026-10-24

- Q: How is staff-only time decided for segments without a verified actor label? → A: Verified labels decide where they exist (eval clips). For model-only segments, a segment is staff-led if `actor_staff_facilitation` OR `instruction` clears its threshold AND `actor_participant` does not. The share of time excluded this way is reported. A stated limitation: model actor and category scores are weak, so staff exclusion is mostly reliable only on reviewed clips.
- Q: How does multi-label matching work? → A: Per-activity F1 compares the set of all human activities (primary plus secondary) with the model's activities above threshold. Primary accuracy compares primary with primary. `uncertain_mixed` is not in per-activity F1; it is its own class in primary accuracy, the confusion matrix and primary kappa. Category kappa uses one category per side: the human's chosen category, and the model's top category by threshold-relative strength, or "none" if nothing clears.
- Q: May 005 derive model estimates for staff indicators? → A: Yes. Rule-based model estimates from config, tagged "model estimate":
  - `solo_on_stage` = `person_count == 1` AND (`performance` clears OR the primary is `singing`, `lyricism_spoken_word`, `acting_theater` or `dance`);
  - `group_performance` = `person_count >= 2` with the same activity condition;
  - `voice_projection` = audio `singing_voice` or `speech` above threshold during those segments.

  They are evaluated against human indicator labels in a separate "signal-based indicators" table (precision, recall, F1). This is also the main ablation evidence for 004.
- Q: What is the "too few to score" cut? → A: Support below 5.
- Q: How is the bootstrap resampled? → A: Block bootstrap: blocks of 10 consecutive segments per video.
- Q: How are phases used? → A: The `phase` column as is. `mid` is its own column if any video has it. `unknown` is excluded from the headline and listed.
- Q: What counts as youth voice time? → A: Audio `singing_voice` or `speech` above threshold, OR `has_speech`, during performance or spoken word, excluding staff-led time.
- Q: What counts as audience response? → A: Applause during performance plus the 10 s after a performance segment ends.
- Q: What is the failure gallery? → A: The top 20 wrong primary calls, ranked by the model's confidence. Segment IDs only.
- Q: What counts as a positive expression? → A: A segment is positive if `positive` is its highest expression score (argmax over the mapped labels), not a calibrated threshold. Segments with faces only.
- Q: Which Part 3 items stay? → A: The classifier is cut (no embeddings are stored). The Expression Index is the only optional Part 3 item.
- Q: How do segments get `needs_review`? → A: A segment inherits `needs_review` from the moment containing it. Segments in no moment, including idle ones, count as "model made no call": they are reported, and treated as `uncertain_mixed` for primary accuracy.
- (Gap from the code, default taken) The blind checklist lets the labeler tick several categories, or none. For category kappa, a segment with exactly one human category uses it; with none, the human side is "none"; with two or more, the segment is excluded from category kappa and counted.

## User Scenarios & Testing *(mandatory)*

The users are the three-person team preparing the Devpost submission and slides, and TPA staff, who read the results. Youth never use this feature and never appear as individuals in any output. Every scenario works on mock runs and a mock review log.

### User Story 1 - Honest evaluation against blind labels (Priority: P1) 🎯

A team member runs one command. It compares the model's per-segment output with the blind labels on eval videos only. It reports per-label precision, recall and F1 with support, macro-F1 per label group, primary-activity accuracy and a confusion matrix, model-vs-labeler kappa, accuracy on confident vs flagged segments, bootstrap 95% intervals, and a per-video breakdown. Labels with too few examples are marked "too few to score".

**Why this priority**: Constitution VIII says no number reaches the slides without a defined metric on a video-split eval set. This is the required deliverable, and it must exist before Sunday 10:00 ET.

**Independent Test**: On a mock review log and mock official runs with dev and eval videos, the command writes the metrics files. A blind label placed on a dev video changes nothing. A planted disagreement shows up in the confusion matrix.

**Acceptance Scenarios**:

1. **Given** blind labels and official model output, **When** the evaluation runs, **Then** every metric uses only segments of eval-split videos that have a blind label, and the output states the number of labeled segments per video.
2. **Given** a label whose eval support is below 5, **When** metrics are written, **Then** that label shows its counts and "too few to score" in place of precision, recall and F1, and it is left out of macro-F1. The output lists the excluded labels.
3. **Given** the human primary and the model primary per segment, **When** the evaluation runs, **Then** it reports:
   - primary accuracy on all labeled segments;
   - primary accuracy on only the segments where the human chose a real activity (not `uncertain_mixed`);
   - a confusion matrix with `uncertain_mixed` as its own class;
   - Cohen's kappa for primary activity and for category, each with one plain-English sentence (for example, "0.4 means the model agrees with the labeler well beyond chance, but far from perfectly").
4. **Given** the model's `uncertain_mixed` rows and moment `needs_review` flags (inherited by the segments a moment contains), **When** the evaluation runs, **Then** it reports how often each fires on labeled eval segments, how many segments are in no moment ("model made no call"), primary accuracy on unflagged vs flagged segments, and the share of segments that are unflagged (coverage).
5. **Given** labeled segments, **When** intervals are computed, **Then** primary accuracy, macro-F1 per group and both kappas carry a bootstrap 95% interval. Resampling is a block bootstrap over blocks of 10 consecutive labeled segments per video, with a fixed seed, so reruns give identical numbers.
6. **Given** top-up queue labels exist, **When** the evaluation runs, **Then** they are reported in a separate section, precision only, never pooled with the main sample.
7. **Given** the evaluation, **When** it finishes, **Then** it writes a failure-gallery list: the 20 wrong primary calls with the highest model confidence (threshold-relative strength), ranked. Segment IDs only, plus the model and human labels, with no media and no filenames.

---

### User Story 2 - Ablation and signal-based indicators (Priority: P1)

The same evaluation is reported for two systems side by side:
1. SigLIP2 zero-shot as shipped (the official predict and moments output);
2. the same output adjusted by simple fusion rules using the 004 signals (for example, singing requires audio `singing_voice` or `music`, and spoken word requires `speech`).

A separate "signal-based indicators" table evaluates the rule-based model estimates of `solo_on_stage`, `group_performance` and `voice_projection` against the human indicator labels. The rules live in config. Any thresholds they use come from dev videos only.

**Why this priority**: It shows whether 004 added value, and it is the main methods result for the slides. Visual zero-shot cannot propose indicators at all, so the signal-based indicators table is the clearest evidence for 004.

**Independent Test**: On mock data where the audio signal contradicts a visual `singing` call, fusion removes that call and the ablation table shows both rows. Changing an eval-video score never changes a fusion threshold.

**Acceptance Scenarios**:

1. **Given** fusion rules in config, **When** the evaluation runs, **Then** the output has one row per system with the same metric columns, and the metrics are computed on the same segments.
2. **Given** a rule needs a threshold (for example "speech score high"), **When** thresholds are set, **Then** they come from the existing per-model calibrated thresholds, which use dev videos only. Thresholds are never derived from eval labels.
3. **Given** a segment where a signal a rule needs is missing (no official audio run, or no faces), **When** fusion runs, **Then** the rule does not fire and the zero-shot call stands. The output counts how often each rule fired and how often it could not be applied.
4. **Given** the indicator rules and the human indicator labels on eval segments, **When** the evaluation runs, **Then** a "signal-based indicators" table reports support, precision, recall and F1 per indicator ("too few to score" below support 5). Segments missing `person_count` or audio count as "signal not available", not as negatives.

---

### User Story 3 - Youth-only program indicators, early vs showcase (Priority: P2)

A team member runs one command and gets TPA's program-level indicators per video and per phase. Each number is tagged "staff-verified" or "model estimate (unvalidated)". Staff-only time and idle time are kept out of the youth denominators. There is a headline comparison of early program vs final showcase.

**Why this priority**: This is what TPA shows funders. It depends on Part 1 to know how far the model estimates can be trusted.

**Independent Test**: On mock runs with a staff-only stretch, an idle stretch and overlapping windows, the indicators exclude the staff-only stretch, report idle time separately, and count each second of video once.

**Acceptance Scenarios**:

1. **Given** official runs and the review log, **When** indicators are computed, **Then** each indicator is reported per video and per phase, with its value, the time base it uses, and a tag: "staff-verified" if every contributing segment has a verified label, "model estimate (unvalidated)" if every one is a model fallback, or "mixed (x% staff-verified)" otherwise.
2. **Given** overlapping windows (8 s windows, 2 s overlap; 4 s and 1 s for clips), **When** time is summed, **Then** each segment contributes its stride (window minus overlap, so 6 s or 3 s), and the video's last segment contributes its remaining length. Total counted time never exceeds the video's duration.
3. **Given** a segment that is staff-led (verified: the human actor is staff only, or the human chose `instruction` without `actor_participant`; model-only: `actor_staff_facilitation` or `instruction` clears its threshold and `actor_participant` does not), **When** youth indicators are computed, **Then** the segment is excluded from every youth numerator and denominator. Its time is reported as "staff-led time", split into staff-verified and model-estimated time, with the share of time excluded by the model rule.
4. **Given** idle segments, **When** shares are computed, **Then** idle time is excluded from denominators and reported separately as "idle / setup time".
5. **Given** expression scores, **When** the positive expression rate is computed, **Then** it uses only segments with `faces_detected ≥ 1`. A segment counts as positive when `positive` is its highest expression score among the mapped labels. The rate states its base and is always tagged "model estimate (unvalidated)". It describes observable expressions only.
6. **Given** original work, **When** it is reported, **Then** it uses staff-tagged `original_work` only. With no staff tags it reads "not tagged yet", never a model guess.
7. **Given** a phase with no video, **When** the headline is built, **Then** that side reads "no videos in this phase", and no comparison or change is computed.

---

### User Story 4 - Files for the slides and Devpost (Priority: P2)

Results are written as JSON and CSV, chart images on a white background with legends, and a short markdown summary. Everything goes in the outputs folder and contains aggregates only: no filenames, no transcript text, no frames, nothing per person.

**Why this priority**: The deck and the Devpost write-up are built from these files.

**Independent Test**: Running both commands on mock data produces the files. A scan of every output file finds none of the mock filenames, none of a planted transcript sentinel, and no per-person field.

**Acceptance Scenarios**:

1. **Given** a finished run, **When** outputs are written, **Then** a new folder under the outputs root holds `metrics.json`, `metrics.csv`, `indicators.json`, `indicators.csv`, `failure_gallery.csv` (segment IDs only), the chart images and `summary.md`. Earlier result folders are never overwritten.
2. **Given** `--dry-run`, **When** a command runs, **Then** it prints the headline numbers and what it would write, and writes nothing.
3. **Given** any output file, **When** it is scanned, **Then** it contains no video filename, no transcript text, no frame or image of footage, and no per-person or per-face field.

---

### User Story 5 - Optional TPA Expression Index (Priority: P3, cut first)

Only after Stories 1–4 are done: a TPA Expression Index. It is 0–100, a program-level composite of the four headline indicators with equal weights. Its components are always shown next to it, and it is labeled "example, unvalidated". (The frozen-embedding classifier is cut: no per-segment embeddings are stored.)

**Why this priority**: Nice for the deck, not required. It is cut first if time runs short.

**Independent Test**: The index is reproducible from its displayed components.

**Acceptance Scenarios**:

1. **Given** the four headline indicators, **When** the index is computed, **Then** each component is scaled to 0–100 by a documented rule, averaged with equal weights, and shown with its components and the label "example, unvalidated". There is no per-person or per-video ranking.

---

### Edge Cases

- **A labeled segment the model has no output for** (a video missing from the official runs): it is counted as "not scored", excluded from metrics, and reported with its count.
- **A blind label later rejected by the same reviewer**: only the latest state per (segment, label) counts.
- **A segment saved with Skip** (`clip_unavailable`): excluded from evaluation and counted as "skipped".
- **Primary is `uncertain_mixed` on both sides**: counts as agreement in primary accuracy and kappa.
- **A segment in no moment** (idle, or no activity cleared): "model made no call". It is counted, and treated as `uncertain_mixed` for primary accuracy, the confusion matrix and primary kappa.
- **The human ticked two or more categories, or none**: two or more are excluded from category kappa (counted); none counts as "none".
- **`person_count` missing** (no official people run for the video): the signal-based indicator rules do not fire; that is reported as "signal not available".
- **Labels the model never predicts** (staff-only indicators, `confident`): they are not in model metrics. They are listed as "not model-scored".
- **Eval support of 0 or 1 for a whole group**: macro-F1 for that group reads "too few to score".
- **A video with phase `mid`**: shown as its own headline column if any video has it.
- **A video with phase `unknown`**: shown per video, left out of the headline, and listed by video ID.
- **Mock or eval videos with no audio, people or expression run**: indicators that need that signal read "signal not available" for that video, never 0.
- **Very short videos or clips**: the stride rule still sums to at most the duration.
- **A segment both staff-verified and model-scored**: the verified label wins, and the tag records it.

## Requirements *(mandatory)*

### Functional Requirements

**Data selection**

- **FR-001**: Evaluation MUST use only segments of videos whose `split` is `eval`, and only segments with a blind label in the main sample. Top-up labels are evaluated separately (precision only). Dev-video labels, if any exist, are ignored by evaluation.
- **FR-002**: The human label per segment is the latest state per (segment, label) over all blind reviews. Segments saved only with Skip are excluded. Guideline 0.4.0 and 0.5.0 labels are pooled.
- **FR-003**: Model output per segment comes from the default official reads (official moments run if listed, else predict) and the official 004 stage runs. It uses the current per-model thresholds, the same ones moments uses.

**Metrics (Part 1)**

- **FR-004**: For each label in the activity, category and actor groups, report support (human positives), predicted positives, true positives, precision, recall and F1. Labels with support below 5 (configurable) show counts and "too few to score", and are excluded from macro-F1.
- **FR-005**: Report macro-F1 per label group (activity, category, actor) over the scorable labels, with the list of excluded labels.
- **FR-006**: Report primary-activity accuracy on all labeled segments, and separately on segments whose human primary is a real activity. Report a confusion matrix over the activity labels plus `uncertain_mixed`.
- **FR-007**: Report Cohen's kappa between model and human for primary activity, and for category (FR-008 sets how multi-label category becomes one value), each with one plain-English sentence.
- **FR-008**: Matching rules:
  - **Per-activity F1:** compares the set of all human activities (primary plus secondary) with the set of model activities above threshold.
  - **Primary accuracy:** compares the human primary with the model primary.
  - **`uncertain_mixed`:** not in per-activity F1; it is its own class in primary accuracy, the confusion matrix and primary kappa.
  - **Category kappa:** one category per side. The human's single ticked category ("none" if none; segments with two or more are excluded and counted), and the model's top category by threshold-relative strength ("none" if nothing clears).
  - **Category and actor F1:** per label, multi-label on both sides.
- **FR-009**: Each segment inherits `needs_review` from the moment containing it. Segments in no moment are "model made no call" and count as model primary `uncertain_mixed`. Report on labeled eval segments:
  - how often `uncertain_mixed`, `needs_review` and "no call" occur;
  - primary accuracy on unflagged and flagged segments;
  - coverage (the unflagged share).
- **FR-010**: Bootstrap 95% intervals (percentile method, 1,000 resamples by default, fixed seed in config) for primary accuracy, each group's macro-F1, both kappas and the signal-based indicator F1s. It is a block bootstrap: each video's labeled segments, in time order, are cut into blocks of 10 consecutive segments, and blocks are resampled.
- **FR-011**: Per-video breakdown: labeled segments, primary accuracy and macro-F1 per eval video, so the v04 imbalance is visible.
- **FR-012**: Failure gallery: the 20 wrong primary calls with the highest model confidence (threshold-relative strength of the model's primary), ranked. Columns are segment ID, video ID, model label and human label only.
- **FR-013**: Limitations are written with the metrics:
  - single labeler (no human-vs-human agreement);
  - the eval-set size and the v04 share;
  - model actor and category scores are weak, so staff exclusion is mostly reliable only on reviewed clips;
  - audio conditions use a fixed 0.5 AST probability, not a dev-tuned threshold;
  - the learned system is trained on about 206 labels from two videos; its cross-validated row is the fair number, and the final model's predictions on labeled segments are in-sample;
  - review-mode decisions are per moment, so one decision verifies every segment in that moment;
  - intervals are wide (about 205 labels, mostly from v04, about 21 bootstrap blocks);
  - no number may be shown without its interval.

**Ablation**

- **FR-014**: Fusion rules live in config as a list of rules (label, required signal, condition), never in code. The first rules are:
  - `singing` requires an audio `singing_voice` or `music` score at or above its threshold;
  - `lyricism_spoken_word` requires an audio `speech` score at or above its threshold, or `has_speech`.

  A rule that removes a label can make the segment's primary change to the next cleared label, or to `uncertain_mixed` if none is left.
- **FR-015**: Visual conditions use the calibrated per-model thresholds (dev videos only). Audio conditions use a fixed AST probability threshold from config (`insights.audio_threshold`, default 0.5), because the calibrated top-10% cut sits near 0.95 for `music` and `speech` and would veto real singing. The ablation never fits anything on eval labels.
- **FR-016**: The ablation table has one row per system ("zero-shot", "zero-shot + signal rules", "learned (cross-validated on our labels)") with the same metric columns on the same segments. It reports how often each rule fired and how often it was not applicable.
- **FR-016a** (added 2026-10-24, after real results showed miscalibrated top-10% thresholds): the **learned** system.
  - **Features:** every official per-segment signal: all SigLIP label scores (activity, category, actor, null), the 5 AST probabilities, `has_speech`, `transcript_chars`, `person_count`, `faces_detected` and the expression scores. A missing signal becomes 0 plus a "missing" flag. No embeddings, no frames.
  - **Models:** primary = multinomial logistic regression with `uncertain_mixed` as a class; activities, category and actor = one-vs-rest logistic regression with `class_weight="balanced"`. Features are standardized, with a fixed seed.
  - **Honest evaluation:** grouped cross-validation on the eval labels, where the groups are the bootstrap blocks (10 consecutive labeled segments per video), with 5 folds (fewer if there are fewer blocks). Every prediction comes from a model that never saw its block. Same metrics, CIs and kappa as the other rows, and the signal-based indicators are recomputed on the learned labels.
  - **Final model:** fit on all labels; its predictions for every segment of every video are written to the result folder, tagged `model estimate (learned)`. Predictions on the labeled eval segments are in-sample, so only the cross-validated row is a fair accuracy.
- **FR-017**: Signal-based indicator estimates, rule-based, from config, always tagged "model estimate (unvalidated)":
  - `solo_on_stage` = `person_count == 1` AND (`performance` clears its threshold OR the model primary is one of `singing`, `lyricism_spoken_word`, `acting_theater`, `dance`);
  - `group_performance` = `person_count >= 2` AND the same activity condition;
  - `voice_projection` = audio `singing_voice` or `speech` at or above threshold on a segment where `solo_on_stage` or `group_performance` fires.

  They are evaluated against the human indicator labels on eval segments in a separate "signal-based indicators" table (support, precision, recall, F1, interval). They feed the indicators as model fallback where no staff tag exists. `labels.yaml` is unchanged: the labels stay `source: staff`, and these estimates never appear as model predictions in the official runs.

**Indicators (Part 2)**

- **FR-018**: Each indicator is computed per video and per phase, from verified labels first, then model output as fallback. The model fallback is the learned final model by default (`insights.labels_source: learned`), or zero-shot (`zero-shot`). Each value carries a tag: "staff-verified", "model estimate (unvalidated)" or "mixed (x% staff-verified)". A value without a tag MUST NOT be written.
- **FR-019**: Time base: each segment contributes its stride (window minus overlap from the video's effective settings); the last segment of a video contributes its own length minus the overlap with the previous one. Video totals never exceed the video's duration.
- **FR-020**: Youth-only. Staff-led segments are excluded from every TPA-facing numerator and denominator.
  - **With a verified label:** staff-led when the human chose `actor_staff_facilitation` without `actor_participant`, or `instruction` without `actor_participant`.
  - **Model only:** staff-led when `actor_staff_facilitation` or `instruction` clears its threshold and `actor_participant` does not.
  - **Reporting:** "staff-led time" is split into staff-verified and model-estimated, with the share of total time excluded by the model rule.
- **FR-021**: Idle segments (`is_idle`) are excluded from denominators and reported as "idle / setup time".
- **FR-022**: The indicators:
  - **performance share**: performance time / youth time;
  - **active participation share**: youth doing an activity, not `audience_watching` and not instruction;
  - **solo moments**: count of contiguous runs and minutes, from staff `solo_on_stage`, else the FR-017 model estimate;
  - **youth voice time**: minutes of performance or spoken word (`lyricism_spoken_word`) where audio `singing_voice` or `speech` is at or above threshold, OR `has_speech` is true, excluding staff-led time;
  - **activity mix and diversity**: time share per activity, and a diversity value (number of activities with at least 1 minute, plus normalized entropy);
  - **audience response**: minutes with audio `applause` at or above threshold during performance segments, or within 10 s after a performance segment ends, per performance minute;
  - **positive expression rate**: share of segments with `faces_detected ≥ 1` whose highest expression score (argmax over the mapped labels) is `positive`; unvalidated;
  - **original work**: staff-tagged only;
  - **instruction vs participation ratio**: instruction time / participation time, with instruction time reported as context (it is staff-led);
  - **coverage**: share of segments with faces, and reviewed vs all segments.
- **FR-023**: Headline: early vs showcase (plus a `mid` column if any video has `mid`) for performance share, solo moments, youth voice time and positive expression rate. Each has the values, their tags and the absolute change. Videos with phase `unknown` are excluded from the headline and listed. There is no significance claim unless an interval exists.
- **FR-024**: Never per person: no person or face IDs, no individual scores, no rankings of participants or of named groups. Videos are identified by opaque video ID only.

**Outputs and command**

- **FR-025**: Commands `tpa-index evaluate` and `tpa-index insights` (names may change in planning) each support `--dry-run`, which prints headline numbers and the planned files and writes nothing.
- **FR-026**: Each non-dry run writes a new, never-overwritten folder under the outputs root with the JSON, CSV, chart images (light background, legends) and `summary.md` listed in US4-AS1. Every number in the summary carries its tag and, for metrics, its interval.
- **FR-027**: No output contains a video filename, transcript text, frames or images of footage, or per-person fields. Charts plot aggregates only.

**Optional (Part 3, cut first; the frozen-embedding classifier is cut)**

- **FR-028**: The TPA Expression Index: equal-weight mean of the four headline indicators, each scaled to 0–100 by a documented rule, shown with its components, labeled "example, unvalidated", program-level only.

**Privacy and data (override everything)**

- **FR-030**: Building and testing MUST NOT read DATA_DIR, `~/tpa-data`, `/Volumes/T7`, the real review log, `config/videos.local.yaml` or any `config/*.local.yaml`. Tests use mock runs and a mock review log via the existing `TPA_*` overrides.
- **FR-031**: No identity, no per-person rows, no clinical or psychological wording. Expression results are "observable, unvalidated" everywhere they appear.

### Key Entities

- **Evaluation set**: eval-video segments with a main-sample blind label, plus per-segment human labels (primary, other activities, category, actor, uncertain) and model labels (cleared labels, primary, `uncertain_mixed`, `needs_review`).
- **Metric record**: system, group, label, support, TP/FP/FN, precision, recall, F1 (or "too few to score"), interval where applicable.
- **Ablation row**: system name, plus the headline metrics with intervals, plus rule fire counts.
- **Indicator record**: indicator, scope (video ID or phase), value, unit, time base, tag (staff-verified / model estimate / mixed x%).
- **Fusion rule** (config): target label, required signal(s), threshold source, action (keep or drop).
- **Indicator rule** (config): indicator name, `person_count` condition, activity condition, optional audio condition. Its output is always a model estimate.
- **Signal-based indicator record**: indicator, support, TP/FP/FN, precision, recall, F1 (or "too few to score"), interval, not-available count.
- **Result folder**: one per run, holding JSON, CSV, charts and summary.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A test proves metrics use eval-video segments only: adding or changing blind labels on dev videos changes 0 metric values.
- **SC-002**: A test proves youth-only indicators exclude staff-only segments: a mock staff-only stretch adds 0 seconds to every youth numerator and denominator, and appears in "staff-led time".
- **SC-003**: A test proves overlap is not double-counted: summed indicator time per video equals the video's duration minus idle and staff time, within one stride, and never exceeds the duration.
- **SC-004**: A test proves every number in the indicator outputs carries a tag (staff-verified, model estimate, or mixed with a percentage). 0 untagged values.
- **SC-005**: A test proves no output file contains any mock video filename or a planted transcript sentinel, and no per-person or per-face field. 0 occurrences.
- **SC-006**: Bootstrap intervals are reproducible: two runs with the same seed give identical numbers.
- **SC-007**: Both commands finish in under 30 seconds on the Mac for the full dataset (about 12 videos, 205 labeled segments).
- **SC-008**: Every metric shown in `summary.md` has its support and 95% interval next to it, satisfying Constitution VIII. A test checks that each metric line in the summary has an interval.
- **SC-009**: The ablation table compares systems on identical segment sets: a test checks the segment counts per row are equal.
- **SC-010**: A test proves a segment in no moment counts as model `uncertain_mixed` in primary accuracy, and that `needs_review` is inherited from the containing moment.
- **SC-011**: A test proves the block bootstrap never splits a block: every resample is made of whole 10-segment blocks from one video.
- **SC-012**: A test proves the signal-based indicator rules follow config: changing the `person_count` condition in config changes the estimates, with no code change.

## Assumptions

- The current per-model thresholds (`thresholds.local.yaml` at run time; mock thresholds in tests) define the model's yes/no decisions, as they do for moments. 005 does not recalibrate.
- **Model primary per segment** is the activity with the highest strength (score ÷ threshold) that clears its threshold, else `uncertain_mixed`. This is the 002 rule. A segment in no moment is treated as `uncertain_mixed` ("model made no call").
- Phase comes from the existing `videos.phase` field. "Early program" = `early` and "final showcase" = `showcase`, plus `mid` as a column if present.
- Category and actor are multi-label on both sides for F1. Category kappa uses one value per side (FR-008).
- The review log is read from the local database at run time only, never while building or testing.
- Charts use matplotlib, which is already locked, so there is no new dependency.

## Out of Scope

- Any change to the app, blind mode, the review log, the official runs or the moments rules.
- Recomputing any model output, embeddings or signals.
- The frozen-embedding classifier (no per-segment embeddings are stored).
- Changing `labels.yaml`: indicator labels stay `source: staff`. The signal-based estimates exist only in 005 outputs.
- Human-vs-human agreement (no overlap labels exist).
- Per-person, per-face or per-participant output of any kind.
- Publishing results anywhere outside the outputs folder.

## Open Items

None blocking. Notes for planning:

1. Indicator rules and fusion rules share one config block, so `voice_projection` reuses the same audio thresholds as fusion.
2. The positive expression rate uses argmax, so it needs no calibrated threshold. The `positive` threshold from 004 calibration is not used here.
