# Feature Specification: Visual Pipeline

**Feature Branch**: `002-visual-pipeline`

**Created**: 2026-10-23

**Status**: Draft

**Input**: User description: "002-visual-pipeline: zero-shot activity scoring and moments. Builds on 001-foundation. Score every visual label per segment with an open image–text model, detect idle segments, derive primary/secondary/uncertain labels, merge moments with per-video window settings, a `predict` command (one run per video, CUDA or Apple GPU), a Slurm wrapper, official-run selection, and code/output sync helpers for UNF HPC. Out of scope: audio, transcript, person counting, expressions, review UI, metrics."

## User Scenarios & Testing *(mandatory)*

Users are the team running the pipeline and building the review app (003), and indirectly TPA staff, who will see the resulting moments as unverified suggestions. Everything is demonstrable on mock data and synthetic clips.

### User Story 1 - Score every visual label for every segment (Priority: P1)

A team member runs the predict command on a video. For each segment, the system looks at the window's sampled frames and gives every visual label (category, actor, activity and status groups) a score between 0 and 1. It records which model produced the score and in which run.

**Why this priority**: Moments, review, insights and evaluation all depend on these scores. Without them there is nothing to review.

**Independent Test**: Run predict on a synthetic clip with the small model and confirm there is exactly one score per segment × visual label, every score is in 0–1, and every row carries the model name and run ID. Run it twice with the same input and model and confirm the scores are identical.

**Acceptance Scenarios**:

1. **Given** a segmented video and the label config, **When** predict runs, **Then** every segment has one score for every label in the category, actor, activity and status groups, and none for audio, expression or indicator labels.
2. **Given** a label with 3–5 prompt phrasings, **When** it is scored, **Then** all of its phrasings contribute to its single score.
3. **Given** a portrait and a landscape video, **When** scored, **Then** the full frame is used for both, with nothing cropped away.
4. **Given** the same input, config and model, **When** predict runs twice, **Then** the scores match to within a small tolerance (rounding differences between hardware are allowed).

---

### User Story 2 - Get moments with a primary label, secondary labels and an uncertainty flag (Priority: P1)

After scoring, each segment gets the following labels:
- a **primary activity**: the top activity label above the threshold;
- **secondary labels**: the other labels above the threshold;
- an **uncertain/mixed** flag when no activity clears the threshold or the top two activities are too close to call.

Consecutive segments are then merged into moments using the existing merge rules and the video's own window settings. Moments are written for review.

**Why this priority**: Moments are what staff review (Constitution VI). Making the uncertainty visible stops the system from overstating weak guesses.

**Independent Test**: Feed hand-built scores into the labeling and merge step and confirm primary, secondary and uncertain/mixed come out as specified, and that merged moments respect the video's window settings.

**Acceptance Scenarios**:

1. **Given** a segment whose top activity scores above the threshold and clearly ahead of the runner-up, **When** labeled, **Then** that activity is primary and the segment is not flagged uncertain/mixed.
2. **Given** a segment where no activity clears the threshold, **When** labeled, **Then** it has no primary activity and is flagged uncertain/mixed.
3. **Given** a segment where the top two activities are within the configured margin, **When** labeled, **Then** it is flagged uncertain/mixed.
4. **Given** a video whose kind uses shorter windows (for example clips), **When** moments are merged, **Then** the gap and minimum-length rules use that video's window length, not the default.
5. **Given** segments flagged idle, **When** moments are built, **Then** idle segments never start or extend a moment.
6. **Given** any segment, **When** labeled, **Then** an `uncertain_mixed` status row is stored for it: score 1 when flagged and 0 otherwise, under the model name `rules`. The model-scored `uncertain` and `mixed` status labels are removed from the label file, and the guideline version moves to 0.2.0.

---

### User Story 3 - Skip idle time (Priority: P2)

Segments showing setup, waiting or an empty room are marked idle. Review can then skip them, and indicators can leave them out.

**Why this priority**: Long recordings contain a lot of dead time. Hiding it saves staff review hours, but scoring and moments work without it.

**Independent Test**: On synthetic clips, a static empty frame sequence is marked idle and a clip with clear motion and people-like content is not. Both outcomes come from the configured idle rule.

**Acceptance Scenarios**:

1. **Given** a window that matches the idle description strongly, **When** idle detection runs, **Then** the segment is marked idle.
2. **Given** a window of a single person speaking with a static camera, **When** idle detection runs, **Then** the segment is not marked idle merely because the picture barely changes.
3. **Given** a window that matches the idle description strongly but has clear visual change (for example, people walking in to set up), **When** idle detection runs, **Then** it is not marked idle. Both signals must agree.

---

### User Story 4 - Run predict on HPC or the Mac, one video at a time (Priority: P2)

A team member runs predict for all mapped videos or selected ones, with the large model on UNF HPC or the small model on the Mac for benchmarking. Each video is its own run with recorded runtime, peak memory and hardware. A failure on one video does not lose the others. A Slurm wrapper runs predict over all videos on one GPU.

**Why this priority**: Constitution II requires runtime and memory figures for both machines, and the large model only fits on HPC.

**Independent Test**: Run predict on three synthetic clips, one of them broken, and confirm two complete runs with cost figures, a listed failed video ID, and a non-zero exit. Confirm the Slurm wrapper requests one GPU of the specified type and calls predict.

**Acceptance Scenarios**:

1. **Given** a model name from config, **When** predict runs, **Then** that model is used, and its name is recorded on every prediction and run row.
2. **Given** three videos where one cannot be read, **When** predict runs, **Then** two complete runs exist, the failed video ID is reported, and the command exits non-zero.
3. **Given** a GPU or Apple GPU is available, **When** predict runs, **Then** frames are processed in batches on it. Otherwise it falls back to CPU, and the run records which hardware was used.
4. **Given** the optional motion model is enabled, **When** predict runs, **Then** its scores for the motion labels are written as additional rows under its own model name, alongside the main model's rows.

---

### User Story 5 - Only official runs count (Priority: P2)

A team member lists finished runs (ID, stage, video, hardware, counts), picks the ones that count, and records them in an official-runs file per stage. Every reader then uses only those runs by default, so test, mock and benchmark runs never leak into results, the review app, or Devpost numbers.

**Why this priority**: Constitution VIII requires every reported number to be traceable. Mixing a Mac benchmark run with the HPC run would double-count moments.

**Independent Test**: Write three runs, mark one official, and confirm default reads return only that one. An explicit request for other runs still works, and the run listing shows all three.

**Acceptance Scenarios**:

1. **Given** an official-runs file naming one run for a stage, **When** a feature reads that stage's tables by default, **Then** only that run's rows are returned.
2. **Given** a run ID in the official-runs file that does not exist or is unfinished, **When** reading, **Then** the reader fails with a message naming the bad run ID.
3. **Given** the list-runs command, **When** run, **Then** it shows, for each finished run: ID, stage, video ID, hardware, segment count and prediction count. It shows no filenames or paths.

---

### User Story 6 - Move code to HPC and results back (Priority: P3)

A team member syncs the repo to their HPC work folder and pulls HPC outputs back to their local output folder with one command each. Host and user come from the local environment file. Videos and secrets never travel.

**Why this priority**: It saves time and prevents mistakes, but it can be done by hand.

**Independent Test**: Run each script in dry-run mode against a local target and confirm the exclusion list (environment file, virtual environment, models, caches, outputs, video files) and that nothing is sent anywhere if host or user is unset.

**Acceptance Scenarios**:

1. **Given** host and user set, **When** code sync runs, **Then** the repo is copied to `~/tpa-work/repo` on HPC without the environment file, virtual environment, model weights, caches, outputs or any video file.
2. **Given** host or user missing, **When** either script runs, **Then** it stops with a message naming the missing setting and transfers nothing.
3. **Given** output sync, **When** run, **Then** HPC outputs are copied into the local output folder (which the data guard confirms is outside the repo), and no video files are copied.

---

### Edge Cases

- **Videos that yield no segments** (very short or unreadable): the run fails for that video only, with a redacted message.
- **A label in the visual groups with no prompts:** rejected by config validation before any scoring.
- **The model can't be downloaded or found locally:** fails before any video is processed, naming the model, not a path.
- **Out of GPU memory:** the batch size can be lowered in config. The failure is reported per video, and the other videos continue.
- **The motion model only scores some labels:** rows for labels it does not cover are not written. Missing scores never default to zero.
- **Every segment is idle:** no moments are written for that video, and the run still completes.
- **Visual change can't be computed** (only 1 frame per window): the segment is never marked idle.
- **The official-runs file lists two runs for the same video and stage:** reading fails, naming both run IDs.
- **Mac and HPC scores for the same video:** these are kept as separate runs and are never merged.
- **Re-running predict on a video:** creates a new run and never overwrites the old one.

## Requirements *(mandatory)*

### Functional Requirements

**Scoring**

- **FR-001**: The system MUST score every label in the category, actor, activity and status groups for every segment, with a score from 0 to 1. The rule-derived `uncertain_mixed` label is the exception: the rules produce it (FR-007), not the model.
- **FR-002**: Each label's score MUST use all of its configured prompt phrasings, and each segment's score MUST use all of its sampled frames.
- **FR-003**: Frames MUST be used whole (padded, never cropped), for both portrait and landscape videos.
- **FR-004**: Each score MUST be stored as one prediction row: segment, label, label group, score, model name and run ID.
- **FR-005**: The main model and an optional motion model MUST be chosen in config, with separate choices for HPC and for the Mac benchmark. The motion model's scores MUST be stored under its own model name, so that the two can be compared later.
- **FR-006**: Scoring MUST be repeatable: the same input, config and model give the same scores, within a small tolerance across hardware.

**Labels per segment and moments**

- **FR-007**: For each segment, the system MUST derive:
  - a primary activity: the top activity at or above the label threshold;
  - secondary labels: the other labels at or above the threshold;
  - an uncertain/mixed result, when no activity clears the threshold or the top two activities are within the configured margin. It is stored as an `uncertain_mixed` status prediction row (score 1 when flagged, 0 otherwise) under the model name `rules`, alongside the model rows.
- **FR-008**: Moments MUST be built with the existing merge rules, using each video's effective window settings, including per-kind and per-video overrides. The current shortcut that always uses the default window MUST be removed.
- **FR-009**: Idle segments MUST NOT start or extend moments.
- **FR-010**: The predict run MUST write the segments (with the idle signal filled in), predictions and moments for each video as a new run. Existing runs MUST NOT be modified.

**Idle**

- **FR-011**: A segment MUST be marked idle only when the idle description scores at or above its threshold **and** the visual change across the window's frames is below its threshold. The idle description and both thresholds MUST be config, not code.

**Command and environment**

- **FR-012**: A predict command MUST run for all mapped videos or selected video IDs, and accept a model override.
- **FR-013**: Predict MUST process one video per run, and MUST record runtime, peak memory, hardware and segment count per run. A failure on one video MUST NOT discard the other videos' runs. The command MUST report failed video IDs and exit non-zero if any video fails.
- **FR-014**: Predict MUST run on an NVIDIA GPU, an Apple GPU or the CPU without code changes, processing frames in batches whose size is set in config.
- **FR-015**: A Slurm wrapper MUST run predict over all videos on one GPU of the specified type, with an explicit time limit. It MUST set the model cache location explicitly (`HF_HOME=$HOME/hf-cache`), so jobs use the weights already on HPC and never depend on shell startup files.

**Official runs**

- **FR-016**: An official-runs config file MUST list, per stage, which run IDs count.
- **FR-017**: Readers MUST default to the official runs. An explicit run selection MUST still be possible. An unknown or unfinished official run ID MUST fail loudly, naming the ID.
- **FR-018**: A list-runs command MUST show, for each finished run: ID, stage, video ID, hardware, segment count and prediction count. It MUST NOT show filenames or paths.

**Sync**

- **FR-019**: A code-sync script MUST copy the repo to `~/tpa-work/repo` on HPC, excluding the environment file, virtual environment, model weights, caches, outputs and all video files.
- **FR-020**: An output-sync script MUST copy the HPC output folder into the local output folder, and MUST refuse to run if the local output folder fails the data guard.
- **FR-021**: Both scripts MUST read host and user from the local environment file, MUST stop if either is missing, and MUST offer a dry-run mode.

**Constraints**

- **FR-022**: Only open-weight models MAY be used, with weights downloaded and run locally or on HPC. No hosted inference.
- **FR-023**: No identity data MAY be produced: no person or face embeddings, and no per-person rows. Only per-segment scores are stored.
- **FR-024**: Errors and logs MUST stay redacted, as in 001.

### Key Entities

- **Prediction** (from 001): one score per segment × label × model × run. 002 fills it for the visual groups, under the main model and, optionally, the motion model.
- **Segment** (from 001): 002 writes a new version per predict run, with `is_idle` filled in. The other signals stay empty.
- **Moment** (from 001): built per video from the main model's scores, idle-aware, using the effective window settings.
- **Run** (from 001): one per video per predict invocation, with model, hardware, runtime and peak memory.
- **Official runs**: a per-stage list of the run IDs that count for results.
- **Model settings**: the main model and optional motion model for HPC and for the Mac, the batch size, the idle description and thresholds, and the uncertainty margin.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of segments in a predict run have exactly one score per visual label from the main model, and every score is between 0 and 1.
- **SC-002**: Repeating a run with the same input, config and model reproduces every score to within 0.001 on the same machine.
- **SC-003**: On a synthetic test set with known content, the labeling step produces the expected primary label and uncertain/mixed flag in 100% of hand-built cases.
- **SC-004**: When one of three videos is broken, the other two videos' results are kept 100% of the time.
- **SC-005**: Every predict run reports runtime, hardware and segment count. Peak memory is reported wherever the platform supports it.
- **SC-006**: With the official-runs file set, default reads include 0 rows from non-official runs.
- **SC-007**: On HPC with one GPU, the large model processes at least 1 hour of video in under 15 minutes. On the Mac, the small model's runtime per hour of video is measured and reported, with no target.
- **SC-008**: In dry-run mode, the sync scripts list 0 environment files, video files or model weights in the transfer.
- **SC-009**: Errors in tests that involve data files print 0 data paths or filenames.

## Clarifications

### Session 2026-10-23

- Q: How is uncertain/mixed stored? → A: As a rule-based `uncertain_mixed` status prediction row (score 1/0, model `rules`). It replaces the model-scored `uncertain` and `mixed` labels, and the guideline version becomes 0.2.0.
- Q: When is a segment idle? → A: Only when the idle description scores above its threshold **and** visual change is very low.
- Q: Model weights on HPC? → A: They're already in `~/hf-cache`. The Slurm wrapper exports `HF_HOME=$HOME/hf-cache` explicitly.

## Assumptions

- **Model weights:** only the weights are fetched, never hosted inference (Constitution II). On HPC the weights are already in `~/hf-cache`, and the Slurm wrapper points to it. On the Mac they are downloaded once into a cache outside the repo or gitignored.
- **Scores:** the main model's native sigmoid score is used as the 0–1 value; there is no extra calibration in this feature, which is left to evaluation (005).
- **Motion model:** off by default and scores only the motion labels it is configured for (for example dancing and acting).
- **Label vocabulary:** the existing label names are used (`dancing`, `acting`, …). The status group becomes `uncertain_mixed` (rule-derived) plus `needs_review`, and the guideline version moves to 0.2.0. Mock data and tests follow the new vocabulary.
- **Indicators:** indicator, audio and expression labels get no scores in 002, so moments' `indicators` and some secondary groups stay empty until 004.
- **Official runs:** `config/official_runs.yaml` holds only run IDs (no filenames or metadata), so it can be committed. Mock runs can be listed in it for UI development.
- **Video IDs in the listing:** the run listing gets each run's video ID from that run's own videos table, so the runs table schema stays unchanged.
- **Sync scripts:** use rsync over SSH with the team's existing HPC accounts. They never touch `DATA_DIR` contents on either side.
- **Performance target:** the HPC target in SC-007 is an estimate for 4 frames per 6-second stride, to be confirmed by the first real run.
- **Out of scope:** audio, transcripts, person counts, expressions (004), the review UI (003) and metrics (005).
