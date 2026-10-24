# Feature Specification: Extra Per-Segment Signals

**Feature Branch**: `004-signals`

**Created**: 2026-10-24

**Status**: Clarified

**Input**: User description: "004-signals: add four extra per-segment signals (speech transcript, audio cues, person count, facial expressions) as new pipeline stages that write to the existing contract, without changing any official 002 run or the moments rules."

## Context

- `main` holds 001 (contracts), 002 (visual pipeline, calibrate, moments, official runs), the label fix and 003 (blind labeling and review mode). The official segment, predict and moments runs are listed in `config/official_runs.yaml`. The eval split is v04 and v07. Every other video is dev.
- Blind labeling on real data is running right now from the main checkout. This feature MUST NOT change blind mode, the reviews database, or any official run.
- The 001 contract already has the columns this feature fills: `segments.person_count`, `faces_detected`, `has_speech` and `transcript_chars` (all nullable, still null today), plus the `audio` and `expression` label groups in `predictions.label_group`. No table gains or loses a column.
- The feature reuses what 002 built: window and frame sampling, per-video tracked runs (`runs` row with model, git_commit, runtime and peak memory), the `official_runs.yaml` mechanism, `read_table(run_ids=...)` and `read_official(table, stage=...)`, and calibrate.

## Clarifications

### Session 2026-10-24

- Q: Audio label names? → A: Keep `music`, `speech`, `applause` and `silence` as they are. Add one audio label, `singing_voice` (not `singing`, which is an activity label). Bump guideline_version to 0.5.0. The blind checklist built from 0.5.0 MUST be identical to the 0.4.0 one, so 005 can treat 0.4.0 and 0.5.0 blind reviews as one eval set.
- Q: Face detector and weight sources? → A: YuNet (OpenCV FaceDetectorYN, MIT licence, weights from HF `opencv/face_detection_yunet`), detection only. YOLO11 (person) and EmotiEffLib weights may come from their official GitHub releases: pre-downloaded on the login node, version-pinned, stored under the weights cache, and never downloaded inside a job. This clarifies Principle II (open weights, downloaded, run locally) and does not amend it. The model card lists every weight source.
- Q: Transcript location and format? → A: HPC only (and scratch on the Mac for mock and benchmark runs). One JSON file per video per run, under the run folder, with segment and word timestamps. Never synced down. The output sync stays Parquet-only, and a test proves a transcript file is not picked up.
- Q: AudioSet mapping? → A: In config, using exact AudioSet display names. The stage fails at startup if a configured name is missing from the model's labels. singing_voice: Singing, Choir, Male singing, Female singing, Child singing, Rapping, Humming, Chant. music: Music. speech: Speech, "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking", Conversation, "Narration, monologue". applause: Applause, Clapping, Cheering. silence: Silence. Label score = the highest sigmoid probability among its mapped classes.
- Q: Expression model and mapping? → A: EmotiEffLib `enet_b0_8_va_mtl`. Happiness → positive. Neutral → neutral. Anger, Contempt, Disgust → negative. Sadness → upset. Fear and Surprise are dropped (both misfire on open mouths while singing). Per face, a label's score is the sum of its mapped class probabilities, with no renormalization after dropping. The segment score is the mean over all usable faces in the window's frames. Valence and arousal are not stored. `confident` stays unscored.
- Q: Thresholds? → A: Person confidence 0.4, YuNet face score 0.8, minimum face 48 px (all config). Audio and expression labels use calibrate's existing top 10% from dev videos only; 005 may revisit. `person_count` includes any visible audience (a known limitation for the model card).
- Q: faces_detected? → A: The largest number of usable faces in any one sampled frame.
- Q: Word assignment? → A: A word belongs to every segment that contains its midpoint.
- Q: Audio input shorter than the model's input length? → A: Pad with silence.
- Q: Resume key? → A: Skip a (video, stage) pair when a finished run exists with the same model and config hash. `--force` reruns it.
- Q: Slurm layout? → A: One script with a `--stages` option, submitted as two parallel jobs with one GPU each. Job A runs transcript and audio (shared audio extraction). Job B runs people and expression (shared frame decode). Each job: `--partition=gpu --gres=gpu:6KSE:1 --cpus-per-task=8 --mem=64G --time=03:00:00`. A stage failing on one video logs the failure and moves on to the next.

## User Scenarios & Testing *(mandatory)*

The users are the three-person team running the pipeline on UNF HPC and the Mac, and later specs (005 fusion and indicators, 006 export and demo) that will read these signals. TPA staff do not see anything from this feature yet, because there is no app UI in this feature. Youth appear in the footage but are never identified or counted as individuals. Every scenario can be shown with mock data and synthetic clips (tones, silence, solid frames, simple drawn shapes).

### User Story 1 - Run the four signal stages over all videos on HPC (Priority: P1) 🎯

A team member submits two GPU jobs on HPC from one script: job A for transcript and audio, job B for people and expression. For every mapped video, each stage writes its own run with its own run ID, model and code version. If one stage fails on one video, the others carry on. If a job is cut off, resubmitting it picks up where it stopped and does not redo finished (video, stage) pairs. Every run records its runtime and peak memory.

**Why this priority**: Without the stage runs there is nothing to calibrate, join or fuse. The deadline is Sunday morning, so this has to be dependable.

**Independent Test**: On mock data with synthetic clips and stand-in (fake) models, run all four stages over three videos, with one stage made to fail on one video. Confirm that 11 complete runs exist, the failure is reported with its video ID only, and a rerun creates only the missing run.

**Acceptance Scenarios**:

1. **Given** mapped videos and the official segment runs, **When** the signals command runs, **Then** each (video, stage) pair gets one complete run, recorded in `runs` with stage `transcript`, `audio`, `people` or `expression`, its model ID, config hash, hardware, n_segments, runtime, peak memory and git commit.
2. **Given** a stage raises an error on one video, **When** the command continues, **Then** the other stages for that video and all stages for the other videos still finish. The command exits non-zero and lists the failed (video, stage) pairs by video ID and stage name only.
3. **Given** a complete run already exists for a (video, stage) pair with the same model and config hash, **When** the command is run again without `--force`, **Then** that pair is skipped and reported as skipped. With `--force` it is rerun as a new run.
4. **Given** people and expression are both selected for a video, **When** they run, **Then** the video's frames are decoded once and shared. **Given** transcript and audio are both selected, **Then** the audio track is extracted once into one temporary file, which is deleted when the video is done, whether the stages succeeded or failed.
5. **Given** the segment IDs a stage would produce for a video do not match that video's official segment IDs, **When** the stage runs, **Then** it fails for that video, writes nothing, and says the windows do not match.

---

### User Story 2 - Privacy and identity guarantees are provable (Priority: P1)

A team member (or a judge reading the governance note) needs evidence that the new stages store nothing that identifies a person and nothing of what anyone said. Tests prove it. No identity columns exist, no face crop or temporary file survives a run, transcript text never appears in logs or test output and never syncs down, segments with no usable face get no expression rows, and the official 002 runs are byte-for-byte unchanged.

**Why this priority**: Constitution I, III, IV and V are non-negotiable. A signal that breaks them cannot ship, however useful it is.

**Independent Test**: Run the privacy test group on synthetic data. Every test in it passes, and it covers each of the checks in SC-001 to SC-005 and SC-011.

**Acceptance Scenarios**:

1. **Given** any run written by a new stage, **When** its files are inspected, **Then** they contain only contract tables (`videos`, `segments`, `predictions`, `runs`) plus, for transcript runs only, the one transcript JSON file. None has a box, track, embedding, landmark, crop, face or person ID column, or any per-face or per-person row.
2. **Given** a completed or failed run, **When** temporary folders are inspected, **Then** no audio temp file and no image or crop file is left anywhere, inside or outside the output folder.
3. **Given** a transcriber that returns known sentinel text, **When** the transcript stage runs, succeeds, fails mid-way or raises, **Then** the sentinel appears in no log line, no stdout or stderr, no exception message or traceback, and no Parquet file. It appears only in the run's transcript JSON file.
4. **Given** a segment whose frames contain no face at or above the minimum size and face score, **When** the expression stage runs, **Then** that segment has `faces_detected = 0` and no expression rows at all. It never gets a `neutral` row or any other default.
5. **Given** the official segment, predict and moments runs, **When** all four stages run, **Then** every file in those run folders and the existing sections of `official_runs.yaml` are unchanged (same bytes).
6. **Given** a run folder holding a transcript JSON file, **When** the output sync's file filter is applied to it, **Then** only Parquet files are selected and the transcript file is not.

---

### User Story 3 - Later specs read signals joined onto segments (Priority: P2)

A later spec (005) asks for the official segments with the four signal columns filled in. Columns come from the official run for that stage and are null where a stage has not run (or has no official run) for that video. The existing official read used by the app, blind mode and moments returns exactly what it returns today.

**Why this priority**: The signals are only useful once something can read them. But 005 can start from the stage runs directly, so this comes second.

**Independent Test**: On mock data, list official people runs for two of three videos. Read the joined segments and confirm that `person_count` is filled for those two videos and null for the third, that `has_speech` is null everywhere (no official transcript runs), and that the existing official read is unchanged.

**Acceptance Scenarios**:

1. **Given** official runs for some stages and videos, **When** the joined segments are read, **Then** there is one row per official segment, with the 001 `segments` columns. `person_count`, `faces_detected`, `has_speech` and `transcript_chars` come from the matching official stage run and are null otherwise.
2. **Given** official audio or expression runs, **When** their predictions are read with an explicit stage, **Then** they are returned. **When** the existing default official predictions are read, **Then** they do not include them, so moments, needs_review and uncertain_mixed results are unchanged.
3. **Given** `official_runs.yaml` lists two runs of the same stage covering the same video, **When** anything reads that stage, **Then** it fails with the existing "both cover video" error.

---

### User Story 4 - Calibrate audio and expression thresholds on dev data (Priority: P2)

A team member runs calibrate for the audio stage and for the expression stage. Each proposes a per-label threshold for that stage's model, from official dev-split runs only, with the same rule as for visual labels (top 10%, no special cases), and writes it to the gitignored local thresholds file next to the existing visual thresholds.

**Why this priority**: 005 fusion needs thresholds to turn scores into yes/no, and Constitution VIII needs every reported number tied to a dev/eval split. It can follow the stage runs.

**Independent Test**: On mock audio and expression runs over dev and eval videos, calibrate each stage. Thresholds use dev scores only (changing an eval score changes nothing). The local file keeps the existing visual model's thresholds and gains one block per new model.

**Acceptance Scenarios**:

1. **Given** official audio runs, **When** calibrate runs for the audio stage on the dev split, **Then** each of `singing_voice`, `music`, `speech`, `applause` and `silence` gets a proposed threshold (or "too few") under the audio model's name, by the same percentile and floor rules as 002.
2. **Given** official expression runs, **When** calibrate runs for the expression stage, **Then** `positive`, `neutral`, `negative` and `upset` get thresholds. `confident` is not proposed, because no model scores it (it is derived later, out of scope), and the printout says so.
3. **Given** calibrate is asked for the eval split, **When** it runs, **Then** it refuses, as it does today.
4. **Given** the local thresholds file already holds visual thresholds, **When** audio or expression thresholds are written, **Then** the visual entries are kept unchanged.

---

### Edge Cases

- **Video with no audio track**: transcript gives `has_speech = false`, `transcript_chars = 0` for every segment. The audio stage scores silence (the same convention as 002's segmenter) and logs "no audio track" by video ID only.
- **Silent or music-only audio**: VAD finds no speech, so `has_speech = false` and `transcript_chars = 0`. The transcript JSON holds an empty transcript.
- **Sung lyrics**: the transcriber may write out sung words. `has_speech` means "words were transcribed", which can include singing. This is documented and not filtered here (005 decides how to use it, for example together with `singing_voice`).
- **A word spanning two overlapping windows**: a word counts in every segment that contains its midpoint, so with 2 s overlap the same word can count in two adjacent segments. This is expected.
- **Audio window shorter than the audio model's input** (8 s windows, 4 s for clips, the last window of a video): padded with silence to the model's input length.
- **Clip-kind videos with 4 s windows**: stages use each video's effective window settings, so segment IDs match the official segments.
- **An even number of frames per window**: the person-count median is rounded to an integer by a fixed rule (FR-018).
- **Audience in frame**: `person_count` counts every visible person, audience included. This is a known limitation, noted in the model card and left for 005.
- **Faces below 48 px, below face score 0.8, or partly outside the frame**: dropped before scoring. If no face is left in any of a segment's frames, FR-022 applies.
- **A model class with no mapping** (Fear, Surprise, AudioSet classes not in the mapping): dropped and never folded into another label. Expression scores are not renormalized after dropping, so a face's mapped scores can sum to less than 1.
- **A mapping names a label that is not in `labels.yaml`, or a class name missing from the model's labels**: a config error at startup, before any video is processed (exit 3).
- **Model weights missing from the local cache**: exit 3 before any video is processed. Nothing is downloaded, inside a job or anywhere else at run time.
- **A run is interrupted** (killed job): no `runs.parquet` is written, so the run stays incomplete and invisible to reads. A rerun redoes that pair. Temp files from the killed process are removed at the next start of the command (FR-008).
- **Any edit to `labels.yaml` or `pipeline.yaml`** changes the config hash, so the next run redoes every pair (resume matches model and config hash exactly).
- **Mac with the labeling app open**: the Mac uses the small models from config and processes one video at a time, so memory stays low.

## Requirements *(mandatory)*

### Functional Requirements

**Stages and runs**

- **FR-001**: The system MUST provide four new stages, `transcript`, `audio`, `people` and `expression`. Each writes one run per video, tracked with the existing run tracking (run ID, stage, model, config hash, hardware, n_segments, runtime, peak memory, started_at, git commit).
- **FR-002**: Each stage run MUST write only contract tables through the existing validated writer: `videos` (that video's row), `segments` (one row per official segment of that video, with only the stage's own columns filled and all other optional columns null), `predictions` (audio and expression only) and `runs`. Transcript runs also write the one transcript JSON file (FR-013). Existing tables MUST NOT gain columns.
- **FR-003**: Stage runs MUST be new, immutable run folders. No stage may write into, rewrite or delete an existing run folder, including any official segment, predict or moments run.
- **FR-004**: Each stage MUST produce exactly the segment IDs of that video's official segments, using the video's effective window settings. A mismatch fails that (video, stage) pair and writes nothing (US1-AS5).
- **FR-005**: One command MUST run the stages chosen with `--stages` (any subset of the four, default all) over the selected videos (all mapped videos by default; the same `--video-id` and `--kind` selection as predict). A failure in one (video, stage) MUST be logged and MUST NOT stop the rest. The exit code is 1 if any pair failed, and failed pairs are named by video ID and stage only.
- **FR-006**: The command MUST be resumable per (video, stage). A pair is skipped when a complete run of that stage already covers the video with the same model and the same config hash. `--force` reruns it as a new run.
- **FR-007**: When people and expression run on the same video, frames MUST be decoded once and shared. When transcript and audio run on the same video, the audio MUST be extracted once into a single temporary file and shared.
- **FR-008**: Temporary audio files MUST be created under a configurable scratch location (never inside the repo) and deleted when the video finishes, on success, error or interrupt. Leftovers from a killed process MUST be removed when the command next starts.
- **FR-009**: Model IDs and sizes MUST come from config, with a separate HPC (CUDA) and Mac (MPS/CPU) entry per stage, like 002's `models.main` (transcript: faster-whisper large-v3 on HPC and a small model on the Mac; audio: an AST AudioSet-finetuned model; people: YOLO11; face detection: YuNet; expression: EmotiEffLib `enet_b0_8_va_mtl`). Weights MUST load from the local weights cache only. Missing weights exit 3 before any video is processed.
- **FR-009a**: Weight sources (a clarification of Principle II, not an amendment): AST, faster-whisper and YuNet (`opencv/face_detection_yunet`) come from Hugging Face. YOLO11 and EmotiEffLib come from their official GitHub releases. All are open weights, pre-downloaded on the HPC login node (and once on the Mac), version-pinned in config, stored under the weights cache, and never downloaded at run time inside a job. The model card MUST list every weight source, version and licence.

**Transcript**

- **FR-010**: The transcript stage MUST transcribe each video's audio once (not per window), with voice activity detection on, using the configured model.
- **FR-011**: Per segment it MUST write `has_speech` (true if at least one transcribed word is assigned to the segment) and `transcript_chars` (number of characters of the words assigned to it). It writes no predictions rows.
- **FR-012**: Words MUST be assigned to segments by timestamp: a word belongs to every segment whose [start_s, end_s) contains the word's midpoint.
- **FR-013**: The full transcript MUST be written only to one JSON file per video per run, inside that run's folder, holding segment-level and word-level text with timestamps. Transcripts stay on HPC (or in scratch on the Mac for mock and benchmark runs) and are never synced down. They MUST NOT appear in any log, printed output, exception message, traceback, Parquet file or test output. Transcript content MUST be stripped from any error raised inside the transcript stage before it is logged.
- **FR-013a**: The output sync MUST keep copying Parquet files only (no change to its allowlist). A test MUST prove that a transcript JSON file in a run folder is not selected by the sync's filter.

**Audio**

- **FR-014**: The audio stage MUST score each segment's audio window with the AST AudioSet-finetuned classifier, padding windows shorter than the model's input length with silence. It writes one `predictions` row per audio label per segment, with `label_group = audio`, the model ID and the run ID. Scores lie in 0–1.
- **FR-015**: The mapping from AudioSet classes to audio labels MUST live in config, listed as exact AudioSet display names (some contain commas, so each is a quoted string). A label's score is the highest sigmoid probability among its mapped classes. Unmapped classes are dropped. At startup the stage MUST fail (exit 3) if any configured name is not in the model's labels or any mapped label is not an `audio` label in `labels.yaml`. Initial mapping:
  - `singing_voice`: Singing, Choir, Male singing, Female singing, Child singing, Rapping, Humming, Chant
  - `music`: Music
  - `speech`: Speech, "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking", Conversation, "Narration, monologue"
  - `applause`: Applause, Clapping, Cheering
  - `silence`: Silence
- **FR-016**: The audio labels are the `audio` group of `labels.yaml`. The existing `music`, `speech`, `applause` and `silence` keep their names, and one label is added: `singing_voice` (group `audio`, distinct from the activity label `singing`). `guideline_version` goes from 0.4.0 to 0.5.0. Because label validation requires prompts for every model-scored label, `singing_voice` gets 3–5 observable sound prompts like the other audio labels, although the audio model does not use prompts.
- **FR-016a**: The blind-mode checklist built from the 0.5.0 `labels.yaml` MUST be identical (same groups, labels, order and keys) to the one built from 0.4.0, and a test MUST prove it. 005 can then treat 0.4.0 and 0.5.0 blind reviews as one eval set. No other label, group or observation type changes.

**People**

- **FR-017**: The people stage MUST detect only the person class in each sampled frame with YOLO11 (detection only, no tracking), keeping detections with confidence ≥ 0.4 (config).
- **FR-018**: It MUST write `segments.person_count` as the median of the per-frame person counts over the segment's sampled frames, rounded to an integer (halves round down). It MUST NOT store boxes, track IDs, crops or per-frame counts. `person_count` counts every visible person, audience included (a known limitation for the model card).

**Expression**

- **FR-019**: The expression stage MUST detect faces in the sampled frames with YuNet (detection only), keeping faces with face score ≥ 0.8 and a shorter side ≥ `min_face_px` (existing config, 48 px). The facial landmark points YuNet returns MUST be discarded immediately and never stored or logged.
- **FR-020**: Usable faces MUST be scored for expression in memory only with EmotiEffLib `enet_b0_8_va_mtl`. Face crops and face images MUST NOT be written to disk, logged or kept after the frame is processed. The model's valence and arousal outputs MUST NOT be stored.
- **FR-021**: The mapping from the expression model's classes to expression labels MUST live in config. Initial mapping: Happiness → `positive`; Neutral → `neutral`; Anger, Contempt, Disgust → `negative`; Sadness → `upset`; Fear and Surprise dropped (both misfire on open mouths while singing). Per face, a label's score is the sum of its mapped class probabilities, with no renormalization after dropping. A label's segment score is the mean of that per-face score over all usable faces in the segment's sampled frames. The stage fails at startup (exit 3) if a mapped class is not in the model's classes or a mapped label is not an `expression` label.
- **FR-022**: Per segment, the stage MUST write `faces_detected` (the largest number of usable faces in any one sampled frame) and, only if `faces_detected >= 1`, one `predictions` row per mapped expression label with `label_group = expression`. With 0 usable faces it MUST write no expression rows.
- **FR-023**: `confident` MUST NOT be model-scored by this stage. It is derived later (out of scope).

**Read path and official runs**

- **FR-024**: `config/official_runs.yaml` MUST gain one section per new stage (`transcript`, `audio`, `people`, `expression`), with the same rules as today: run IDs only, at most one run per video per stage, identical on HPC and in the repo. Existing sections MUST NOT be changed by this feature.
- **FR-025**: A new read function MUST return the official segments joined with the four signal columns by segment ID, each column taken from that stage's official run and null where the stage has no official run for the video. If a stage's official run for a video lacks one of the official segment IDs, the read MUST fail and say which stage and video.
- **FR-026**: The existing official reads (default `read_official` for segments, predictions and moments) MUST return exactly the same rows as before. Stage predictions are reachable only by reading with an explicit stage.

**Calibrate**

- **FR-027**: Calibrate MUST accept `--stage audio` and `--stage expression`. It proposes per-label, per-model thresholds from official dev-split runs only, by the same rules as 002 (the existing `calibrate.top_pct` of 10, null floor and minimum), with no stage-specific rules. Where a rule has no input (no null labels, no idle scores), it is reported as not applied, as today. Results go to the gitignored local thresholds file under the stage's model name, keeping all other models' entries. 005 may revisit these thresholds.

**Scripts**

- **FR-028**: One Slurm script MUST run the signals command and pass `--stages` and the other arguments through. It is submitted as two parallel jobs: job A with `--stages transcript,audio` (shared audio extraction) and job B with `--stages people,expression` (shared frame decode). Each job requests `--partition=gpu --gres=gpu:6KSE:1 --cpus-per-task=8 --mem=64G --time=03:00:00`, sets `HF_HOME=$HOME/hf-cache` and offline mode, and can be resubmitted to resume.

**Privacy and identity (override everything above)**

- **FR-029**: No stage may compute or store face embeddings, face or person tracks, re-identification features, same-person matches, landmarks, boxes, or any per-face or per-person row. The unit of analysis is the segment.
- **FR-030**: Expression outputs are observable-expression suggestions ("AI suggestion, unvalidated"). No label, config key, prompt, log line or doc string may describe them as feelings, emotions felt, mental state or any psychological or clinical label. (The model's class names, such as Sadness, appear only in the mapping config as the model's own class keys.)
- **FR-031**: No code, test or fixture may read, list or open the data directory, `~/tpa-data`, `/Volumes/T7`, `config/videos.local.yaml` or any `config/*.local.yaml` except through the existing guarded paths. Tests use `TPA_*` overrides that point at scratch folders.
- **FR-032**: Tests use synthetic media only (tones, silence, solid frames, simple drawn shapes). No real face image or speech recording enters the repo. Tests that load real model weights are marked `slow`, use the smallest model sizes, and are not run by default.

### Key Entities

- **Stage run**: one run folder per (video, stage) with `videos`, `segments`, optional `predictions` and `runs`. Immutable, keyed by run ID. Transcript runs also hold the one transcript JSON file.
- **Segment signal columns** (existing 001 columns): `has_speech`, `transcript_chars` (transcript), `person_count` (people), `faces_detected` (expression). They are aggregates per segment, never per person.
- **Audio prediction**: `predictions` row with `label_group = audio`, one per segment for each of `singing_voice`, `music`, `speech`, `applause` and `silence`.
- **Expression prediction**: `predictions` row with `label_group = expression`, one per segment for each of `positive`, `neutral`, `negative` and `upset`, present only when the segment has at least one usable face.
- **Class mappings** (config): AudioSet display name → audio label (max rule), expression-model class → expression label (sum rule). Unmapped classes are dropped.
- **Transcript file**: one JSON per video per transcript run with segment and word text and timestamps. Lives only in the run folder on HPC (or Mac scratch). Never printed, synced or committed.
- **Official stage section**: one list of run IDs per new stage in `official_runs.yaml`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A test proves no output schema and no new-stage run file contains an identity column (box, track, embedding, landmark, crop, face or person ID) or a per-face or per-person row. 0 such columns or rows.
- **SC-002**: A test proves that after a successful, a failed and an interrupted run, the scratch folder holds 0 temporary audio files and the output and scratch folders hold 0 image or crop files.
- **SC-003**: A test with a sentinel transcript proves the sentinel appears 0 times in captured logs, stdout, stderr, exception messages, tracebacks, Parquet files and pytest output, on success and on a forced error, and appears only in the run's transcript JSON file.
- **SC-004**: A test proves that segments with 0 usable faces (solid frames, faces below the minimum size) get `faces_detected = 0` and 0 expression rows.
- **SC-005**: A test proves that every file in the official segment, predict and moments run folders, and the existing sections of `official_runs.yaml`, are byte-identical before and after running all four stages. The default official reads return identical rows.
- **SC-006**: On mock data, a rerun after one forced failure creates runs only for the failed (video, stage) pairs (0 duplicate runs). One failing stage leaves 100% of the other pairs complete.
- **SC-007**: Every new-stage run has runtime and peak memory recorded (non-null where measurable) on both HPC and the Mac.
- **SC-008**: Each of the two signals jobs finishes all mapped videos on one HPC GPU within its 3-hour limit. If one does not, resubmitting it completes the work without redoing finished pairs.
- **SC-009**: Audio and expression calibration use dev-split data only: changing any eval-split score changes 0 proposed thresholds.
- **SC-010**: The default test suite (no slow tests) passes on the Mac with no real model loaded, so it does not disturb the running labeling app.
- **SC-011**: A test proves the output sync filter selects 0 transcript files from a run folder that contains one.
- **SC-012**: A test proves the blind-mode checklist from the 0.5.0 `labels.yaml` is identical to the one from 0.4.0.
- **SC-013**: A test proves a misspelled AudioSet name or expression class in the mapping config stops the command at startup (exit 3) before any video is processed.

## Assumptions

- The four signal columns already in `segments` are the intended outputs. Filling them is "writing to the existing contract", and no table gets a new column. Each stage's run also writes a `segments` table (other optional columns null), the same way predict and moments runs already repeat segments.
- The required libraries (faster-whisper, ultralytics, emotiefflib, transformers, opencv) are already locked in `pyproject.toml`. This feature adds no new dependency. YuNet runs through OpenCV's built-in face detector.
- One run per (video, stage) matches 002's per-video runs and the "one run per video" rule of `official_runs.yaml`. Jobs A and B write different stages, so they never race on the same (video, stage).
- Stage predictions stay out of the default official predictions read, so moments, needs_review and uncertain_mixed results cannot change. Moments' `secondary_groups` already names audio and expression, but nothing reads stage runs by default, so nothing changes until 005 chooses to.
- The blind checklist is built only from `app.blind_groups` (activity, category, actor, indicator). Adding an `audio` label therefore leaves it unchanged. The only visible change for labelers after merge is that new reviews carry guideline version 0.5.0.
- `has_speech` with VAD on is a "words were transcribed" flag, not a speech-versus-singing classifier.
- Only merged code reaches the main checkout, where blind labeling runs. Developing on this branch cannot affect labeling in progress.

## Out of Scope

- Any change to moments, needs_review or uncertain_mixed rules.
- Fusing signals into labels, indicators (solo_on_stage, voice_projection and the rest), and `expr_confident`. These come in 005 and 006.
- Any app UI, demo view or export.
- Storing valence or arousal.
- Running any stage on real data, any HPC transfer, or running the sync scripts as part of this feature's work.

## Open Items

None blocking. Two notes for planning:

1. Whether every configured AudioSet name (for example Humming and Chant) is in the AST model's labels is checked at startup (FR-015). If one is missing, the mapping config is corrected, not the code.
2. The prompt wording for `singing_voice` (FR-016) is a config detail for the plan.
