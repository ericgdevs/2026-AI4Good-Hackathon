# Feature Specification: Shared Data Foundation

**Feature Branch**: `001-foundation`

**Created**: 2026-10-23

**Status**: Draft

**Input**: User description: "001-foundation: the shared data foundation every later feature builds on. No model inference in this feature. Provides the output contract (six tables), config files, a mock-data generator, a segmenter, run tracking, a data guard, and a Mac + UNF HPC environment with a Slurm job template. Out of scope: model inference, the review UI, metrics."

## User Scenarios & Testing *(mandatory)*

The "users" of this feature are the three team members building later features (pipeline stages, the review app, insights, evaluation) and, indirectly, TPA staff who will run the finished system on their own laptop. Every story below must be demonstrable without any real footage.

### User Story 1 - Build against a fixed contract with mock data (Priority: P1)

A team member starting the review app, insights, or evaluation work generates a full set of fake tables (videos, segments, predictions, moments, runs, reviews) with one command and builds their feature against them, confident that the real pipeline will later produce tables of exactly the same shape.

**Why this priority**: Constitution VII (Contracts First) makes the fixed schema and mock data the gate for all other feature work. Without it, three people cannot work in parallel during a 48-hour event.

**Independent Test**: Run the mock-data generator into an empty output folder outside the repo, then load each of the six tables and confirm every row passes schema validation, cross-table references resolve, and no identity columns exist.

**Acceptance Scenarios**:

1. **Given** an empty output folder configured outside the repo, **When** the mock generator runs, **Then** all six tables are created, with several fake videos, a few hundred segments, label scores for every segment, merged moments, run records, and review log entries.
2. **Given** the generated tables, **When** each is validated against the contract, **Then** every table passes, every `segment_id` in predictions exists in segments, every `video_id` in segments and moments exists in videos, and every `run_id` exists in runs.
3. **Given** the generator is run twice with the same seed, **When** the outputs are compared, **Then** they are identical.
4. **Given** a writer attempts to save a table with a missing column, an extra identity-like column, a score outside 0–1, or an unknown enum value, **When** the write is attempted, **Then** it is rejected with an error naming the table and the offending column, and nothing is written.

---

### User Story 2 - Keep data out of the repo and out of assistant context (Priority: P1)

A team member configures where footage lives and where outputs go through a local environment file. If either location is inside the repo, every entry point refuses to run. Any error or log line that would mention a data path or video filename shows a redacted placeholder instead.

**Why this priority**: Constitution I and III are non-negotiable. A single leaked filename in a traceback pasted into an assistant chat violates them, so the guard must exist before any code touches footage.

**Independent Test**: Point the data or output location inside the repo and confirm the guard stops with a clear message. Trigger an error involving a file under the data location and confirm the message and logs contain no part of the real path or filename.

**Acceptance Scenarios**:

1. **Given** the data location or output location resolves to a path inside the repo (including via symlink or relative path), **When** any pipeline command or the mock generator starts, **Then** it exits immediately with a non-zero status and a message naming which setting is wrong, without printing the path itself.
2. **Given** either setting is missing, **When** a command that needs it starts, **Then** it exits with a message naming the missing setting.
3. **Given** a stage fails while reading a file under the data location, **When** the error is printed or logged, **Then** the data root and the filename are replaced by a placeholder (e.g. `<DATA_DIR>/<video:abc123>`), while the error type and non-data context remain readable.
4. **Given** the real video metadata file exists locally, **When** the repo's status is checked, **Then** that file is ignored by version control and the assistant's permission settings deny reading it and the data location.

---

### User Story 3 - Segment a video into fixed windows (Priority: P2)

A pipeline stage hands the segmenter a video located under the data location. The segmenter returns one segment row per window (8 s windows with 2 s overlap by default) and, per window, a small number of evenly spaced frames plus that window's audio, held in memory or in temporary files outside the repo, ready for later model stages.

**Why this priority**: Every later inference stage consumes segments; the segment is the unit of analysis (Constitution IV). It is P2 only because mock data unblocks parallel work before the segmenter exists.

**Independent Test**: Run the segmenter on the public test video and confirm the segment count, boundaries, IDs, frame count per window, and audio duration per window match the pipeline config.

**Acceptance Scenarios**:

1. **Given** a 60-second video and the default config (8 s window, 2 s overlap, 4 frames), **When** it is segmented, **Then** windows start every 6 s, each segment ID is `{video_id}_{start_ms}`, each window yields exactly 4 frames, and the audio covers the window's time range.
2. **Given** the window length changes in the pipeline config, **When** the same video is segmented again, **Then** segment boundaries follow the new value with no code change.
3. **Given** segmentation completes, **When** the produced segment rows are validated, **Then** signal columns (person count, faces detected, speech, transcript length, idle) are empty, awaiting later features, and the rows pass the segments contract.
4. **Given** segmentation finishes or fails, **When** the run ends, **Then** no frames or audio are left inside the repo, and temporary files are removed.
5. **Given** the same video and config, **When** segmented on the Mac and on UNF HPC, **Then** the segment rows are identical.

---

### User Story 4 - Record every stage run with cost (Priority: P2)

Any pipeline stage wraps its work in a run-tracking helper. When the stage ends, a run row is recorded with the stage name, model name, a fingerprint of the config used, the hardware name, the number of segments processed, runtime, peak memory, and start time.

**Why this priority**: Constitution II and VIII require runtime and peak memory for every reported number, on both Mac and HPC. Building it in now means no stage can skip it later.

**Independent Test**: Wrap a dummy stage that allocates a known amount of memory and sleeps briefly; confirm the recorded run row has plausible runtime, non-zero peak memory, a hardware name, and a config fingerprint that changes when the config changes.

**Acceptance Scenarios**:

1. **Given** a stage runs on a machine with a GPU, **When** it finishes, **Then** peak memory reflects GPU memory and the hardware name identifies the GPU model.
2. **Given** a stage runs on the Mac, **When** it finishes, **Then** peak memory reflects the Mac's process or unified memory use and the hardware name identifies the Mac chip.
3. **Given** two runs with identical config, **When** compared, **Then** their config fingerprints match; changing any config value changes the fingerprint.
4. **Given** a stage raises an error, **When** the run ends, **Then** the failure is surfaced (not swallowed) and no partial pipeline output for that run is presented as complete.

---

### User Story 5 - Same code on the Mac and UNF HPC (Priority: P3)

A team member installs the project on the Mac or on UNF HPC with the same steps, imports the package as `tpa_index`, and submits a batch job on HPC from a provided job template that requests one GPU on the correct partition with a time limit.

**Why this priority**: Needed before heavy processing, but mock-data work and the contract can proceed on the Mac alone.

**Independent Test**: On each machine, install, import the package, run the environment check (which reports the device in use and the guard status), and on HPC submit the job template running that same check.

**Acceptance Scenarios**:

1. **Given** a fresh checkout on either machine, **When** the documented install steps run, **Then** the package imports as `tpa_index` and the environment check passes.
2. **Given** the HPC job template, **When** submitted, **Then** it requests the GPU partition, one GPU of the specified type, and an explicit time limit, and runs the environment check on a GPU node.
3. **Given** no GPU is available, **When** the environment check runs, **Then** it reports the fallback device (Apple GPU or CPU) instead of failing.

---

### User Story 6 - Change labels and pipeline settings without code (Priority: P3)

A team member edits the label vocabulary (names, groups, prompt phrasings, display colors) or pipeline settings (window, overlap, frames per window, minimum face size, label threshold, moment merge rules) in config files, and every component picks up the change. The label file carries a guideline version that is stamped on every review entry.

**Why this priority**: Constitution XI. Needed for later features, but defaults can ship first.

**Independent Test**: Load the configs, change one value, reload, and confirm validation catches malformed entries (unknown group, too few or too many phrasings, missing color, missing guideline version).

**Acceptance Scenarios**:

1. **Given** the shipped label config, **When** loaded, **Then** every label has a name, a group from the allowed set, 3–5 prompt phrasings, and a display color, and the file has a guideline version.
2. **Given** a label with 2 phrasings or an unknown group, **When** the config loads, **Then** loading fails with a message naming the label and the problem.
3. **Given** the repo, **When** inspected, **Then** it contains only a mock video metadata example; the real metadata file is local-only.

---

### Edge Cases

- Video shorter than one window: produces exactly one segment covering the whole video.
- Final partial window: a trailing window shorter than the full length is kept if it is at least half a window, otherwise merged into the previous segment's coverage, so the whole video is covered with no window of near-zero length.
- Video with no audio track: segments are still produced; the audio for each window is silence of the right length, and the absence is recorded rather than raising.
- Variable frame rate or corrupt frames: frames are sampled by timestamp; an unreadable frame is replaced by the nearest readable one, and a fully unreadable video fails with a redacted error.
- Duplicate segment IDs (e.g. two windows rounding to the same millisecond): rejected at write time.
- Re-running a stage: produces a new run ID and new output files; earlier run outputs are never overwritten (immutable per run).
- Concurrent review writes from two reviewers: both entries are appended; neither is lost.
- A review correcting an earlier review: appended as a new entry; prior entries are never edited or deleted.
- Data or output setting given as a relative path, with `~`, or through a symlink into the repo: resolved before the inside-repo check.
- Error messages raised by third-party libraries that embed file paths: still redacted before reaching logs or the console.
- Peak memory unavailable on a platform: recorded as empty with the reason logged, rather than a fake zero.

## Requirements *(mandatory)*

### Functional Requirements

**Output contract**

- **FR-001**: The system MUST define six tables with fixed schemas: videos, segments, predictions, moments, runs, and reviews (columns listed under Key Entities).
- **FR-002**: Pipeline outputs (videos, segments, predictions, moments, runs) MUST be written as columnar files, one set per run, and never overwritten once written. Reviews MUST be stored in a single local database file as an append-only log.
- **FR-003**: Every write MUST validate rows against the schema: required columns present, no extra columns, correct types, enums limited to allowed values, scores and confidence within 0–1, end time after start time, and unique primary keys. A failing write MUST write nothing.
- **FR-004**: No table may contain identity data: no person IDs, track IDs, per-face rows, face crops, or embeddings. Validation MUST reject any column outside the contract.
- **FR-005**: Segment IDs MUST follow `{video_id}_{start_ms}`. Video IDs MUST be opaque short identifiers that do not reveal the original filename.
- **FR-006**: Signal columns on segments (person count, faces detected, speech present, transcript length, idle) MUST be allowed to be empty until a later feature fills them.
- **FR-007**: The reviews log MUST support adding entries only; there MUST be no operation that updates or deletes an existing entry.
- **FR-008**: Every review entry MUST record reviewer, timestamp, guideline version, source (AI or staff), and mode (blind or review).

**Configuration**

- **FR-009**: A label config MUST list each label with name, group (activity, category, actor, audio, expression, indicator, or status), 3–5 prompt phrasings, and a display color, and MUST carry a guideline version.
- **FR-010**: A pipeline config MUST hold window length (default 8 s), overlap (default 2 s), frames per window (default 4), minimum face size (default 48 px), label threshold (default 0.5), and moment merge rules.
- **FR-011**: The repo MUST ship a mock video metadata example only. Real video metadata MUST live in a local file that is ignored by version control and denied to assistants.
- **FR-012**: Configs MUST be validated on load, with errors that name the entry and the problem.
- **FR-013**: The label vocabulary MUST NOT include clinical, psychological, health, disability, demographic, or mental-state labels (Constitution V).

**Mock data**

- **FR-014**: A mock-data generator MUST produce all six tables with several fake videos, a few hundred segments, label scores, moments, run rows, and review entries covering every action, source, and mode value.
- **FR-015**: Mock output MUST be reproducible from a seed, MUST pass the same validation as real output, and MUST use obviously fake names (no real filenames or dataset details).
- **FR-016**: Mock scores and moments MUST be realistic enough to exercise downstream features: varied score distributions, some moments flagged as needing review, and reviews that both agree and disagree with AI labels.

**Segmenter**

- **FR-017**: Given a video under the data location, the segmenter MUST produce segment rows and, per window, the configured number of evenly spaced frames and the window's audio.
- **FR-018**: Frames and audio MUST be held in memory or in temporary files outside the repo, and temporary files MUST be removed when processing ends, including on failure.
- **FR-019**: The segmenter MUST run on the Mac (Apple GPU or CPU) and on UNF HPC (Linux with NVIDIA GPU) with identical segment rows for the same input and config.

**Run tracking**

- **FR-020**: A run-tracking helper MUST record, for each stage run: run ID, stage, model, config fingerprint, hardware name, segments processed, runtime, peak memory, and start time.
- **FR-021**: Peak memory MUST measure GPU memory when a GPU is used and process/unified memory on the Mac.

**Data guard**

- **FR-022**: The data location and output location MUST come from the local environment file only.
- **FR-023**: Every entry point MUST check, before doing any work, that both locations resolve to paths outside the repo, and MUST fail loudly otherwise.
- **FR-024**: All errors and logs MUST redact the data location, the output location, and video filenames, including messages raised by third-party libraries.

**Environment**

- **FR-025**: The project MUST be importable as `tpa_index` and install with the same steps on the Mac and on UNF HPC.
- **FR-026**: The repo MUST include a batch job template for UNF HPC that requests the GPU partition, exactly one GPU of the specified type, and an explicit time limit.
- **FR-027**: An environment check command MUST report the compute device in use, guard status, and config validity, without printing data paths.

### Key Entities *(include if feature involves data)*

- **Video**: One source recording. `video_id` (opaque), `filename` (relative to the data location; never shown in logs or errors), `duration_s`, `fps`, `phase` (early, mid, showcase, unknown), `split` (dev, eval; assigned per video so no video is in both).
- **Segment**: One fixed time window of a video; the unit of analysis. `segment_id` (`{video_id}_{start_ms}`), `video_id`, `start_s`, `end_s`, and nullable aggregate signals: `person_count`, `faces_detected`, `has_speech`, `transcript_chars`, `is_idle`.
- **Prediction**: One label score for one segment from one model run (long format). `segment_id`, `label`, `label_group` (activity, category, actor, audio, expression, indicator, status), `score` (0–1), `model`, `run_id`.
- **Moment**: A contiguous stretch of segments merged into one reviewable event. `moment_id`, `video_id`, `start_s`, `end_s`, `primary_label`, `secondary_labels` (list), `indicators` (list), `confidence` (0–1), `needs_review` (yes/no), `run_id`.
- **Run**: One execution of one pipeline stage. `run_id`, `stage`, `model`, `config_hash`, `hardware`, `n_segments`, `runtime_s`, `peak_mem_gb`, `started_at`.
- **Review**: One staff action on an AI output or a staff-added label; append-only. `review_id`, `target_type` (segment or moment), `target_id`, `label`, `label_role` (primary, secondary, indicator), `action` (accept, edit, reject, defer, add), `new_label` (set for edit/add), `source` (ai, staff), `mode` (blind, review), `guideline_version`, `reviewer`, `note`, `created_at`.
- **Label definition**: Config entry with name, group, 3–5 prompt phrasings, display color; the file carries a guideline version.
- **Pipeline settings**: Config with window, overlap, frames per window, minimum face size, label threshold, and moment merge rules.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A team member with no footage can generate the full mock dataset and load all six tables in under 1 minute on the Mac.
- **SC-002**: 100% of mock rows pass contract validation, and 100% of deliberately malformed test rows (one per rule) are rejected.
- **SC-003**: In tests that trigger errors on files under the data location, 0 occurrences of the data path or filename appear in console output or logs.
- **SC-004**: Pointing either data setting inside the repo stops every entry point within 1 second, 100% of the time.
- **SC-005**: Segmenting the public test video produces the expected number of segments for its length and configured window, with exactly the configured frame count per window, and identical segment rows on the Mac and on HPC.
- **SC-006**: Every stage run recorded during this feature's tests has non-empty runtime, hardware name, and config fingerprint, and peak memory is non-empty where the platform supports it.
- **SC-007**: A new team member can install the project and pass the environment check on either machine in under 15 minutes following the README.
- **SC-008**: The HPC job template, submitted unchanged, starts on a GPU node and completes the environment check.
- **SC-009**: The repo contains 0 real filenames, real video metadata, or data files, verified by a check before each commit.

## Assumptions

- The public test video is a short, openly licensed clip with audio, stored under the data location (or another location outside the repo) and never committed. Automated tests that need a real video may also use a tiny synthetic clip generated at test time.
- Moment merge rules default to: merge adjacent segments sharing the same top label above threshold, allow gaps of up to one window, and set `needs_review` when confidence is below the label threshold or labels conflict. Exact values live in the pipeline config and can be tuned later.
- The starting label vocabulary covers each group with a small set of observable labels; the expression group uses Confident, Upset, Positive, Neutral, Negative per the constitution. Final wording is owned by the team and versioned via the guideline version.
- `reviewer` is a staff-chosen display name or initials, not linked to any account system; there is no authentication in this feature.
- Video IDs are assigned in the local metadata file; mock IDs are clearly fake.
- `peak_mem_gb` may be empty on platforms that cannot measure it; that is preferred over a misleading value.
- The existing ignore rules for data, model weights, and local config remain in place; this feature only adds to them.
- Out of scope: any model inference, the review interface, evaluation metrics, and filling the segment signal columns.
