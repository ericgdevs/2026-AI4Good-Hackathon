# Tasks: Extra Per-Segment Signals (004)

**Input**: `/specs/004-signals/` (plan.md, spec.md, research.md, data-model.md, contracts/config.md, contracts/interfaces.md, quickstart.md)

**Tests**: required. The spec's success criteria (SC-001 to SC-013) each name a test, and each gets its own task.

**Phase order**: Setup and Foundational come first. Then the phases follow the requested checkpoints, not story order: Phase 3 is job A (transcript and audio, **Checkpoint 1**), Phase 4 is job B (people and expression, **Checkpoint 2**), and Phase 5 is the read path, calibrate, official runs, Slurm and docs (**Checkpoint 3**). Story labels show which spec story a task serves: US1 is stage runs, US2 is the privacy guarantees, US3 is the read path, US4 is calibrate.

**Rules for every task**:
- Never read, list or open DATA_DIR, `~/tpa-data`, `/Volumes/T7`, `config/videos.local.yaml` or any `config/*.local.yaml`. Tests rely on the autouse `_isolated_env` fixture in `tests/conftest.py`.
- Synthetic media only: `tests/clips.py`, tones, silence, solid frames, drawn shapes.
- No stage runs on real data, no ssh or rsync to HPC, never start the app.
- No edits inside `.venv` or site-packages. Library overrides live in our own code.
- No commits and no `git add`.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Generate `tests/data/blind_checklist_0.4.0.json` from the **current, unmodified** `config/labels.yaml` (guideline 0.4.0) **before any label change**. Use `uv run python -c` with `tpa_index.review.checklist(load_labels(), load_pipeline())`, convert tuples to lists, and dump `{"guideline_version": "0.4.0", "checklist": ...}` with `sort_keys=False, indent=2`. Assert in the one-liner that `load_labels()["guideline_version"] == "0.4.0"` so it cannot be generated after the bump.
- [X] T002 [P] Create `scripts/download_weights.sh`, a login-node-only script with `set -euo pipefail`.
  - Refuse to run with exit 2 if `SLURM_JOB_ID` is set, since nothing is downloaded inside a job.
  - Defaults: `HF_HOME=${HF_HOME:-$HOME/hf-cache}`, `TPA_WEIGHTS_DIR=${TPA_WEIGHTS_DIR:-$HF_HOME/tpa-weights}`.
  - Hugging Face downloads with `uv run hf download`: `Systran/faster-whisper-large-v3`, `Systran/faster-whisper-base`, `Systran/faster-whisper-tiny` (slow tests), `MIT/ast-finetuned-audioset-10-10-0.4593`, and `opencv/face_detection_yunet face_detection_yunet_2023mar.onnx`.
  - GitHub release assets with `curl -fL` into `$TPA_WEIGHTS_DIR`: `yolo11n.pt` and `yolo11m.pt` from `https://github.com/ultralytics/assets/releases/download/v8.3.0/`, and `enet_b0_8_va_mtl.onnx` from `https://github.com/sb-ai-lab/EmotiEffLib/raw/<pinned commit>/models/affectnet_emotions/onnx/enet_b0_8_va_mtl.onnx` (the commit hash goes in a variable at the top of the script).
  - Print `sha256sum` of the three GitHub files and of the YuNet file in the HF cache, as `file  sha256` lines ready to paste into `config/pipeline.yaml`.
  - No data paths, ever.
- [X] T003 [P] Add `transcript*.json` to `.gitignore` (in the "Data" block) and to `scripts/sync.exclude`.

---

## Phase 2: Foundational (blocks every later phase)

- [X] T004 Edit `config/labels.yaml`.
  - Set `guideline_version: "0.5.0"`.
  - Insert after `silence` in the audio block: `name: singing_voice`, `group: audio`, `prompts: ["a person singing", "people singing together", "a person rapping into a microphone", "a choir singing"]`, `color: "#c51b8a"`, with the comment `# audio cue; not the activity label "singing"`.
  - No other change.
- [X] T005 Edit `config/pipeline.yaml`. Add `models.transcript`, `models.audio`, `models.people`, `models.expression` and the `signals:` block (people.person_conf 0.4; expression.face_score 0.8 and its mapping; audio.mapping) exactly as in `specs/004-signals/contracts/config.md`. Use `sha256: null` for the GitHub-sourced files, and `models.transcript.hpc_device: cuda`. Keep every existing key unchanged.
- [X] T006 In `src/tpa_index/config.py`:
  - Add `SIGNAL_STAGES = ("transcript", "audio", "people", "expression")`.
  - Add the load-time validation rules from `contracts/config.md` to the pipeline checks, with a `ConfigError` naming the key:
    - audio mapping keys equal the audio-group labels;
    - expression mapping keys are expression labels, without `confident`;
    - no class appears under two labels;
    - `person_conf` and `face_score` are in (0, 1);
    - `hpc_device` is cuda or cpu;
    - `file` entries are bare filenames;
    - `sha256` is null or 64 hex characters.
  - Add `weights_dir()` (`TPA_WEIGHTS_DIR`, else `$HF_HOME/tpa-weights`, `HF_HOME` defaulting to `~/.cache/huggingface`).
  - Add `pinned_weight(entry) -> Path`, raising `ConfigError` for a missing file (`weights not found: <file>; run scripts/download_weights.sh on the login node`), a null sha256 (`weights not pinned: set sha256 for <file>`) or a mismatch (`weights checksum mismatch: <file>`). Hash in 1 MB chunks.
- [X] T007 In `src/tpa_index/config.py`, add `stage_config_hash(stage, labels, pipeline, video) -> str`. It calls the existing `config_hash` on only the per-stage keys in research R1: the effective `window_s`/`overlap_s` (and `frames_per_window` for people and expression) from `effective_settings(pipeline, video)`, `audio_sample_rate` for transcript and audio, `models.<stage>`, `signals.<stage>` when present, `min_face_px` for expression, and the sorted label *names* of the stage's group for audio and expression. Raise `ConfigError` for an unknown stage.
- [X] T008 [P] In `tests/test_config.py`, add tests:
  - `singing_voice` loads, guideline is 0.5.0, and `singing_voice` is in group `audio`;
  - each new validation rule raises with its key named (unknown mapped label, missing audio label, `confident` in the expression mapping, duplicate class, bad sha256, path in `file`, `face_score` 1.5);
  - `pinned_weight` errors for missing, unpinned and mismatched files (scratch files under `tmp_path`, `TPA_WEIGHTS_DIR` monkeypatched);
  - `stage_config_hash` is unchanged after editing any label prompt or another stage's `signals` key, changes after editing its own model or mapping, and a `by_video` override changes only that video's hash.
- [X] T009 [P] [US2] **SC-012 test**: in `tests/test_review.py`, add `test_blind_checklist_identical_to_0_4_0`. Build `checklist(load_labels(), load_pipeline())` and compare it, with tuples converted to lists, against the `checklist` in `tests/data/blind_checklist_0.4.0.json`. Also assert that the current guideline_version is "0.5.0".
- [X] T010 In `src/tpa_index/runs.py`, add optional `RunContext.runtime_s: float | None = None` and `RunContext.peak_mem_gb: float | None = None`. `track_run` records them in place of the measured values when they are set. In `tests/test_runs.py`, add one test that the overrides are written and one that the default behaviour is unchanged.
- [X] T011 In `src/tpa_index/schema.py`, add `find_complete_run(stage, video_id, model, cfg_hash, root=None) -> str | None`. It scans `runs/*` folders that have a `runs.parquet`, matches stage, model and config_hash, and requires `video_id` in that folder's `videos.parquet`; it returns the newest matching run_id. In `tests/test_official.py`, add tests for match, wrong hash, wrong model, an incomplete folder (no runs.parquet) and another video.
- [X] T012 Create `src/tpa_index/signals.py` with:
  - `class TranscriptError(Exception)`.
  - `scratch_dir()`: `TPA_SCRATCH_DIR`, else `$TMPDIR`, else `tempfile.gettempdir()`. Raises `GuardError` if the result is inside `repo_root()`. Creates the folder.
  - `clean_stale_scratch(scratch) -> int`: removes `tpa-audio-<pid>-*.wav` whose pid is not alive (`psutil.pid_exists`). Returns the count.
  - `official_segments(video_id, root=None) -> list[dict]`: default `read_official("segments")` rows for the video, else `read_official("segments", stage="segment")`. Raises `ConfigError(f"no official segments for {video_id}")`. Rows are sorted by start_s.
  - `check_windows(video, settings, official) -> None`: recomputes `segment.windows(probe(data_dir()/filename)["duration_s"], ...)` IDs and raises `ValueError("windows do not match official segments")` on any difference.
  - `base_rows(official) -> list[dict]`: copies of the official segment rows with every optional column set to None.
- [X] T013 [P] In `tests/test_signals.py`, add tests for T012: `scratch_dir` honours `TPA_SCRATCH_DIR` and refuses a folder inside the repo; `clean_stale_scratch` removes a dead-pid file and keeps a live-pid file (`os.getpid()`); `official_segments` prefers official moments or predict, falls back to segment, and errors when none exists; `check_windows` passes on a synthetic clip with matching mock segments and fails on a shifted ID.
- [X] T014 [P] [US2] In `tests/test_predict.py`, add `test_siglip_predict_scores_no_audio_or_expression_labels`. Run `predict_video` with `tests.fakes` FakeEmbedder on `clip_videos` using the 0.5.0 labels, and assert that no prediction row has `label_group` in {audio, expression} and that `singing_voice` never appears.
- [X] T015 [P] [US2] In `tests/test_official.py`, add `test_moments_and_calibrate_rebuild_runs_with_old_config_hash`:
  - generate mock runs with `tpa_index.mock.generate` while `tpa_index.mock.config_hash` is monkeypatched to return `"000000000000"`;
  - list them as the official predict runs in the tmp `official_runs.yaml` (`TPA_OFFICIAL_RUNS`);
  - run `cli.main(["moments"])` and `cli.main(["calibrate", "--stage", "predict", "--split", "dev", "--dry-run"])`;
  - assert both return 0 and that the moments rows exist.
- [X] T016 In `tests/fakes.py`, add fakes matching the Protocols in `contracts/interfaces.md`, with call counters:
  - `FakeTranscriber(words=..., sentinel=None, raise_with=None)`: returns segments with words, or raises `RuntimeError(raise_with)`;
  - `FakeAudioTagger(labels, value=0.6)`;
  - `FakePersonCounter(count=2)`;
  - `FakeExpressionScorer(labels, faces_per_frame=...)`: 0 faces when a frame is uniform (`frame.std() == 0`), otherwise N fixed crops, and fixed scores;
  - `FailingStage`: raises on its k-th call.

**Checkpoint F**: `uv run pytest` passes with the 0.5.0 labels. No stage exists yet.

---

## Phase 3: Job A, transcript and audio (US1 + US2), Checkpoint 1

**Goal**: `tpa-index signals --stages transcript,audio` runs on mock data with synthetic audio, using fakes in tests and real small models in slow tests.
**Independent test**: T022, T027, T028 to T031, T032 all pass.

- [X] T017 [US1] In `src/tpa_index/signals.py`, add `assign_words(words, segments) -> dict[str, tuple[bool, int]]`: a word with midpoint m belongs to every segment where `start_s <= m < end_s`, and the last segment of the video also includes `m == end_s`. Chars are `sum(len(w["word"].strip()))`. Add `transcript_rows(official, transcript_segments) -> list[dict]`, which fills `has_speech` and `transcript_chars` on `base_rows` (`False, 0` where no word lands).
- [X] T018 [P] [US1] In `tests/test_signals.py`, test `assign_words`: a word in one segment, a word in a 2 s overlap (counts in both), a midpoint exactly on a boundary, the end-of-video word, no words (all `False, 0`), and whitespace-padded words count only their stripped characters.
- [X] T019 [US1] In `src/tpa_index/signals.py`, add `extract_audio(video, scratch, rate=16000) -> Path`.
  - Write `scratch / f"tpa-audio-{os.getpid()}-{video_id}.wav"` with PyAV: decode audio, `AudioResampler(format="s16", layout="mono", rate=rate)`, then mux into a `wav` container with codec `pcm_s16le`.
  - A video with no audio track gets zeros for `probe(...)["duration_s"]` and logs `"video %s has no audio track; using silence"` with the video ID only.
  - Add `read_clip(wav_path, start_s, end_s, n_samples) -> np.ndarray`: float32 read with `soundfile.read(start=, frames=)`, zero-padded or cut to `n_samples`.
  - Add `map_scores(probs, class_index: dict[str, list[int]]) -> np.ndarray`, the max rule.
  - In `tests/test_signals.py`, test that the WAV of clip t01 is 16 kHz mono with the right length, that t03 (no audio) gives all zeros, that `read_clip` pads with zeros past the end, and that `map_scores` takes the max per label.
- [X] T020 [US1] In `src/tpa_index/signals.py`, add `run_signals(videos, stages, models, pipeline, labels, force=False, root=None) -> dict`, transcript and audio branch first (see `contracts/interfaces.md` for the orchestration). Per video:
  1. `official_segments` and `check_windows`. On error, mark the selected stages `failed:<Type>` and continue.
  2. For each selected stage in (transcript, audio): compute `stage_config_hash`; if `find_complete_run(...)` matches and not `force`, mark it `skipped:<run_id>`.
  3. Otherwise, inside `try/finally`: create the WAV lazily, in the first stage that needs it, inside that stage's `track_run(stage, model=..., cfg_hash=...)` block, and `unlink(missing_ok=True)` it in the `finally`.
  4. Transcript: `models["transcript"].transcribe(wav)`. Write `videos`, `segments` (`transcript_rows`) and `transcript.json` (`{"video_id", "model", "language", "segments"}`) into `runs/<run_id>/`. The whole body sits in `try: ... except Exception as e: raise TranscriptError(type(e).__name__) from None`.
  5. Audio: batch `read_clip` per segment (padded to the model's input samples, `models["audio"].input_samples`) by `pipeline["models"]["batch_windows"][device]`. Write `videos`, `segments` (`base_rows`) and `predictions` (one row per audio label per segment, `label_group="audio"`).
  6. Catch exceptions per (video, stage), log `"%s failed for %s: %s"` with stage, video_id and the type name only, and continue.
  7. Call `clean_stale_scratch(scratch_dir())` once at the start.
  - The video row comes from `probe` plus the official `videos` row (reuse `cli._video_row` logic by moving it to `signals.py` or importing it).
- [X] T021 [P] [US1] In `src/tpa_index/signals.py`, add the module-level logger setup: `logging.getLogger("faster_whisper").setLevel(logging.WARNING)` at import.
- [X] T022 [US1] In `tests/test_signals.py`, add orchestration tests using `FakeTranscriber` and `FakeAudioTagger`, mock official runs from `mock.generate` with `--clips`-style synthetic clips (`tpa_index.mock.write_clips`), and `TPA_SCRATCH_DIR=tmp_path/"scratch"`:
  - one complete run per (video, stage) with the right `runs.stage`, model and stage config hash;
  - a transcriber that fails on one video leaves its audio run and the other videos' runs complete (results dict shows exactly one `failed:`);
  - a rerun gives `skipped:` for done pairs and a run only for the failed one (**SC-006**);
  - `force=True` creates new runs;
  - a shifted official segment ID fails that video and writes no folder with a runs.parquet;
  - the WAV is extracted once per video when both stages run (spy on `extract_audio`);
  - the no-audio clip gives `has_speech=False` and audio rows for every segment.
- [X] T023 [US1] Create `src/tpa_index/signal_models.py` with the Protocols from `contracts/interfaces.md` and `WhisperTranscriber(size, device, compute_type)`.
  - Device: `models.transcript.hpc_device` when torch reports CUDA, else `cpu`; `compute_type[device]`.
  - Load with `faster_whisper.WhisperModel(size, device=..., compute_type=..., local_files_only=True)`.
  - On a CUDA load failure, raise `ConfigError(f"transcript model could not load on cuda: missing {lib}; set models.transcript.hpc_device: cpu to run on CPU")`. `lib` is the first `lib[\w.+-]*\.so[\d.]*` match in the exception text, else the exception type name. Never retry on CPU automatically.
  - Model not in the cache gives `ConfigError("model not found locally: faster-whisper-<size>")`.
  - `transcribe(wav)` uses `vad_filter=True, word_timestamps=True, condition_on_previous_text=False` and returns plain dicts.
  - `name = f"faster-whisper-{size}"`.
- [X] T024 [P] [US1] In `tests/test_signals.py`, add `test_whisper_cuda_failure_names_library_and_does_not_fall_back`: monkeypatch `faster_whisper.WhisperModel` to raise `RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")` on `device="cuda"` and to record calls. Assert the `ConfigError` mentions `libcublas.so.12` and `hpc_device`, and that no call used `device="cpu"`.
- [X] T025 [US1] In `src/tpa_index/signal_models.py`, add `AstTagger(model_id, device, mapping, batch)`.
  - Load `ASTForAudioClassification` and `AutoFeatureExtractor` with `local_files_only=True` (an `OSError` gives `ConfigError("model not found locally: ...")`).
  - Resolve `class_index` from `model.config.id2label`. Any configured name that is missing gives `ConfigError(f"signals.audio.mapping: class {name!r} not in model labels")`.
  - `input_samples = feature_extractor.max_length * 160` (10 ms hop at 16 kHz).
  - `score(clips)` gives `sigmoid(logits)`, passed through `signals.map_scores`.
  - `name = model_id.rsplit("/", 1)[-1]`; `labels` are the mapping keys in order.
- [X] T026 [US1] In `src/tpa_index/cli.py`, add the `signals` subcommand.
  - Options: `--stages` (comma list, default all; an unknown name raises `ConfigError`), `--videos`, `--video-id` (append), `--kind` (append) and `--force`. Reuse `_select_videos`.
  - Set `os.environ.setdefault("HF_HUB_OFFLINE", "1")` and `os.environ["YOLO_OFFLINE"] = "1"`.
  - Build only the selected stages' models **once, before any video**, through a `_make_signal_models(stages, pipeline, dev)` function that tests can monkeypatch.
  - Call `signals.run_signals`, print one line per pair (`<stage> <video_id>: done <run_id>` / `skipped (done: <run_id>)` / `failed <Type>`), and return 1 if any pair failed (listing them on stderr), else 0.
- [X] T027 [P] [US1] In `tests/test_cli.py`, add tests: `--stages bogus` exits 3; a `_make_signal_models` that raises `ConfigError` exits 3 and creates no run folder; a failing fake pair exits 1 and stderr lists only `<stage> <video_id>`.
- [X] T028 [P] [US2] **SC-003 test**: in `tests/test_signals_privacy.py`, add `test_transcript_text_never_leaks`. Use `FakeTranscriber(sentinel="ZQXJ-SENTINEL-7781")`, first succeeding, then on a second video raising `RuntimeError("ZQXJ-SENTINEL-7781")`. With `caplog` at DEBUG and `capsys`, run through `cli.main(["signals", "--stages", "transcript"])` with `_make_signal_models` monkeypatched. Assert the sentinel is absent from `caplog.text`, the captured stdout and stderr, every `*.parquet` under OUTPUT_DIR (read as bytes) and any raised exception, and present only in the successful run's `transcript.json`.
- [X] T029 [P] [US2] **SC-002 test**: in `tests/test_signals_privacy.py`, add `test_no_temp_or_image_files_survive`:
  - after a successful transcript+audio run, a run where the audio fake raises, and a simulated kill (a pre-created `tpa-audio-<dead pid>-v01.wav` in scratch, then a new run);
  - assert `TPA_SCRATCH_DIR` holds no `*.wav`;
  - assert OUTPUT_DIR and scratch hold no `*.png|*.jpg|*.jpeg|*.bmp|*.npy` files.
- [X] T030 [P] [US2] **SC-013 test (audio half)**: in `tests/test_signals_privacy.py`, add `test_bad_audio_mapping_name_exits_before_any_video`. Monkeypatch `AstTagger` loading to a tiny fake model config whose `id2label` lacks `"Chant"`, and run `cli.main(["signals", "--stages", "audio"])`. Assert exit 3, a message naming `'Chant'`, and no run folders. The expression half is T043.
- [X] T031 [P] [US2] **SC-011 test**: in `tests/test_scripts.py`, add `test_output_sync_skips_transcript_json`, following the pattern of `test_output_sync_real_rsync_from_local_folder`. The fake HPC folder has `runs/r1/segments.parquet` and `runs/r1/transcript.json`. Run `scripts/sync_outputs_from_hpc.sh` with `TPA_SYNC_REMOTE=""`, then assert `segments.parquet` arrived and `transcript.json` did not.
- [X] T032 [P] [US1] In `tests/test_signals_slow.py` (module marked `pytest.mark.slow`, skipped when weights are not in the local cache), add:
  - `test_whisper_tiny_on_tone`: 3 s 440 Hz tone WAV in tmp, `WhisperTranscriber("tiny", "cpu", "int8")`, assert no words (VAD);
  - `test_ast_mapping_resolves_and_scores_silence`: real AST on 2 zero clips; mapping names resolve; scores in 0–1, shape (2, 5).
  - Each test deletes its model and runs `gc.collect()` (plus `torch.mps.empty_cache()` on MPS) in a `finally`.
- [X] T033 **Checkpoint 1**: run `uv run pytest` (all default tests pass). If weights are cached, also run `uv run pytest -m slow tests/test_signals_slow.py -k "whisper or ast"`. Then, with `DATA_DIR`, `OUTPUT_DIR` and `TPA_SCRATCH_DIR` pointed at a scratch folder outside the repo, run `tpa-index mock --clips` and `tpa-index signals --stages transcript,audio --kind clip`, and confirm 2 runs per mock clip video in `tpa-index runs` and an empty scratch folder.

---

## Phase 4: Job B, people and expression (US1 + US2), Checkpoint 2

**Goal**: `tpa-index signals --stages people,expression` runs on synthetic clips in one shared frame pass.
**Independent test**: T039 to T043 and T045 pass.

- [X] T034 [P] [US1] In `src/tpa_index/signals.py`, add `median_count(counts) -> int` (`floor(statistics.median(counts))`, halves round down) and `aggregate_expression(per_frame_scores, labels) -> tuple[int, dict | None]`. Each element of `per_frame_scores` is an `(n_faces, n_labels)` array. The result is `faces_detected = max n_faces`, and mean scores over all faces in all frames, or `None` when there are no faces. In `tests/test_signals.py`, test `median_count([1,2])==1`, `[1,2,3,4]==2` and `[3]==3`, and `aggregate_expression` with 0 faces (`(0, None)`), faces in two frames (max and mean), and per-face sums below 1 kept unrenormalized.
- [X] T035 [US1] In `src/tpa_index/signal_models.py`, add `YoloPersonCounter(weights_entry, device, conf)`.
  - Set `os.environ["YOLO_OFFLINE"] = "1"` **before** `from ultralytics import YOLO`.
  - Get the path from `config.pinned_weight(weights_entry)`, which runs first, so a missing file never reaches ultralytics.
  - `count(frames)` runs `model.predict(list(frames), classes=[0], conf=conf, verbose=False, device=...)` and returns `[len(r.boxes) for r in results]`, keeping nothing else.
  - `name = Path(file).stem`.
- [X] T036 [US1] In `src/tpa_index/signal_models.py`, add `FaceExpression(detector_entry, model_entry, mapping, face_score, min_face_px)`.
  - YuNet path from `huggingface_hub.hf_hub_download(repo, file, local_files_only=True)` (an error gives `ConfigError`), then `cv2.FaceDetectorYN.create(path, "", (320, 320), face_score)`.
  - EmotiEffLib: `path = config.pinned_weight(model_entry)` first. Then `import emotiefflib.facial_analysis as fa` and set `fa.get_model_path_onnx = lambda name: str(path)` (comment `# shortcut: library has no path argument; drop when it gets one`). Then construct `fa.EmotiEffLibRecognizerOnnx(model_entry["name"])`.
  - Check the mapping classes against `recognizer.idx_to_emotion_class.values()`; any miss gives `ConfigError(f"signals.expression.mapping: class {c!r} not in model classes")`.
  - `faces(frame_rgb)` does `setInputSize((w, h))`, runs `detect` on the BGR copy, and keeps rows with `score >= face_score` and `min(w, h) >= min_face_px`. It clips each box to the frame, cuts the RGB crop in memory, and discards the row (landmarks included). It returns crops only.
  - `score(crops)` calls `predict_emotions(crops, logits=False)`, takes `[:, :8]` (dropping valence and arousal), and sums the mapped class columns per label.
  - `name = model_entry["name"]`.
- [X] T037 [P] [US1] In `tests/test_signals.py`, add `test_missing_weights_raise_our_error_and_never_download`:
  - monkeypatch `urllib.request.urlretrieve` and `emotiefflib.utils.download_model` to fail the test if called, and `ultralytics.YOLO` to fail the test if constructed;
  - with an empty `TPA_WEIGHTS_DIR`, assert `YoloPersonCounter(...)` and `FaceExpression(...)` each raise `ConfigError` containing `weights not found`;
  - assert `os.environ["YOLO_OFFLINE"] == "1"` after the attempt.
- [X] T038 [US1] In `src/tpa_index/signals.py`, add the people and expression branch of `run_signals`.
  - Resume check per stage as in T020.
  - One `segment_video(video, effective_settings(pipeline, video), with_audio=False)` pass.
  - For each window, each active stage's update runs in its own `try/except`. A stage that raises is marked `failed:<Type>`, logged by stage, video and type only, and dropped for the rest of the pass.
    - people: `median_count(counter.count(frames))`.
    - expression: `faces = [scorer.faces(f) for f in frames]`, `scores = [scorer.score(c) if c else np.zeros((0, L)) for c in faces]`, then `aggregate_expression`. `del` the crops right after scoring.
  - Measure the pass's wall time and `runs._peak_mem_gb(dev)` right after the pass.
  - For each surviving stage: `track_run(...)` with `ctx.runtime_s` and `ctx.peak_mem_gb` set. Write `videos` and `segments` (`person_count` or `faces_detected` on `base_rows`), and for expression `predictions` (rows only where `faces_detected >= 1`, `label_group="expression"`).
  - Frames are never written anywhere.
- [X] T039 [US1] In `tests/test_signals.py`, add orchestration tests with `FakePersonCounter` and `FakeExpressionScorer`:
  - one run per pair with the right stage and hash;
  - `segment_video` is called once per video when both stages run (spy);
  - an expression fake raising mid-pass leaves the people run complete and no expression runs.parquet;
  - resume skips and `--force` reruns;
  - both runs record the same runtime_s.
- [X] T040 [P] [US2] **SC-004 test**: in `tests/test_signals_privacy.py`, add `test_zero_faces_give_no_expression_rows`. Use the static synthetic clip (t04, uniform frames) with `FakeExpressionScorer` (0 faces on uniform frames), plus a case where the scorer returns only faces below `min_face_px` (filtered by the real size rule through a small `FaceExpression`-like filter test of `faces()` with a fake detector output). Assert `faces_detected == 0` for every segment, and that `predictions.parquet` has 0 rows for those segments and never a `neutral` row.
- [X] T041 [P] [US2] **SC-001 test**: in `tests/test_signals_privacy.py`, add `test_no_identity_columns_or_files`.
  - After running all four stages with fakes, every stage run folder contains only `{videos,segments,predictions,runs}.parquet` (plus `transcript.json` for transcript runs).
  - No column name in any written Parquet, or in `schema.SCHEMAS`, matches `(?i)(embed|track|landmark|bbox|box|crop|face_id|person_id|identity)`.
  - Expression and people tables have at most one row per (segment_id, label, model, run_id), and no table has more rows per segment than labels.
- [X] T042 [P] [US1] In `tests/test_signals_slow.py`, add:
  - `test_yolo11n_counts_zero_on_solid_frame`: real yolo11n, one 640×640 gray frame, gives `[0]`;
  - `test_yunet_emotiefflib_no_faces_on_drawn_shapes`: real `FaceExpression` on a frame with a drawn circle and rectangle (cv2), gives `faces() == []`; a direct `score()` on one synthetic 224×224 gray crop returns shape (1, 4) in 0–1, with no file written.
  - One model per test, released in `finally`.
- [X] T043 [P] [US2] **SC-013 test (expression half)**: in `tests/test_signals_privacy.py`, add `test_bad_expression_mapping_class_exits_before_any_video`. Monkeypatch the recognizer to a fake with the 8 standard classes, set a mapping containing `"Happyness"`, and run `cli.main(["signals", "--stages", "expression"])`. Assert exit 3, a message naming `'Happyness'`, and no run folders.
- [X] T044 [US2] In `tests/test_signals_privacy.py`, extend T029's file check to the people and expression run: after a job-B run (and one with a failing expression fake), OUTPUT_DIR and scratch contain no image or `*.npy` files.
- [X] T045 **Checkpoint 2**: (real-model run pending: yolo11n.pt, YuNet and enet_b0_8_va_mtl.onnx not on the Mac yet; fakes cover the pass) `uv run pytest` passes. If weights are cached, run `uv run pytest -m slow tests/test_signals_slow.py -k "yolo or yunet"`, one at a time. Then, in the scratch setup from T033, run `tpa-index signals --stages people,expression --kind clip` and confirm 2 runs per clip video, `faces_detected == 0` on the synthetic clips, and no image files anywhere.

---

## Phase 5: Read path, calibrate, official runs, Slurm, docs (US3 + US4 + US2), Checkpoint 3

**Goal**: later specs can read the signals, thresholds can be calibrated, and HPC can run both jobs.
**Independent test**: T047 to T053 pass.

- [X] T046 [US3] In `src/tpa_index/schema.py`, add `read_signals(root=None) -> pa.Table`. The base is `read_official("segments", root=root)`, else `read_official("segments", stage="segment", root=root)`. For each stage and its columns (transcript: `has_speech`, `transcript_chars`; people: `person_count`; expression: `faces_detected`), use `_official_by_video(load_official(), stage, root)` and that run's segments. Overwrite those columns by segment_id for the covered videos and leave them null elsewhere. If an official stage run lacks an official segment_id of its video, raise `SchemaError(stage, "segment_id", f"official {stage} run {run_id} misses segments of video {vid}")`. Return a table matching `SCHEMAS["segments"]`. Do not modify `read_official`.
- [X] T047 [P] [US3] In `tests/test_official.py`, test `read_signals`: official people runs for 2 of 3 mock videos fill `person_count` for those videos and leave it null for the third; `has_speech` is null with no official transcript runs; a stage run missing a segment raises naming the stage and video; two people runs for one video raise the existing "both cover video" error; `read_official("segments")` and `read_official("predictions")` return the same rows before and after official audio and expression runs are listed (US3-AS2).
- [X] T048 [P] [US2] **SC-005 test**: in `tests/test_signals_privacy.py`, add `test_official_002_runs_unchanged`.
  - Create mock official segment, predict and moments runs and list them in the tmp `official_runs.yaml`.
  - Record the sha256 of every file under those run folders, the bytes of `official_runs.yaml`, and `read_official` results for segments, predictions and moments.
  - Run all four stages with fakes, then append the new sections with the new run IDs to `official_runs.yaml` in the way the team will (text append).
  - Assert every recorded file hash is unchanged, the original bytes are a prefix of the new file, and the default `read_official` results are equal.
- [X] T049 [US4] In `src/tpa_index/cli.py` `_calibrate`, add one generic line after the label table: `labels in scored groups without scores: <names or none>`. Scored groups are the `label_group` values of the model's dev predictions; the names are labels of those groups in `labels.yaml` with `source: model` and no dev scores. Confirm `--stage` already accepts any stage name (it does: a free string); add `choices` only if needed for help text. No other calibrate change.
- [X] T050 [P] [US4] In `tests/test_calibrate.py`, write mock official audio and expression runs over dev and eval videos (fake predictions with `label_group` audio and expression, models `ast-finetuned-audioset-10-10-0.4593` and `enet_b0_8_va_mtl`). Then test:
  - `calibrate --stage audio` writes thresholds for all five audio labels under the AST model;
  - `--stage expression` writes `positive`, `neutral`, `negative` and `upset` and prints `confident` in the "without scores" line;
  - changing an eval-split score changes no threshold (**SC-009**);
  - the visual model's block already in the thresholds file is kept;
  - no idle threshold is written for these stages;
  - `--split eval` exits 3.
- [X] T051 [US1] Append to `config/official_runs.yaml`, after the `moments` list, with existing bytes untouched: a comment line `# 004 signal stages (one run per video per stage)`, then `transcript: []`, `audio: []`, `people: []`, `expression: []`. In `tests/test_official.py`, add a test that the repo file loads, that each new stage is an empty list, and that the segment, predict and moments lists still match their current run IDs.
- [X] T052 [P] [US1] Create `scripts/slurm/signals.sbatch`, modelled on `scripts/slurm/predict.sbatch`:
  - directives: `--job-name=tpa-signals`, `--partition=gpu`, `--gres=gpu:6KSE:1`, `--cpus-per-task=8`, `--mem=64G`, `--time=03:00:00`, `--output=slurm-%x-%j.out`;
  - env: `HF_HOME=$HOME/hf-cache`, `HF_HUB_OFFLINE=1`, `YOLO_OFFLINE=1`, `TPA_WEIGHTS_DIR=$HOME/hf-cache/tpa-weights`;
  - body: `nvidia-smi -L`, `uv sync --frozen`, `uv run tpa-index signals "$@"`;
  - usage comments for the two submissions (`--stages transcript,audio` and `--stages people,expression`) and for resubmitting to resume.
  - In `tests/test_scripts.py`, add `test_signals_sbatch_syntax_and_directives` (`bash -n` plus every directive and env line present), and `test_download_weights_script_refuses_in_job` (`bash -n scripts/download_weights.sh`; running it with `SLURM_JOB_ID=1` and a stub `PATH` exits 2 before any download).
- [X] T053 [P] Create `docs/weights.md`.
  - A table with one row per weight (faster-whisper large-v3, base and tiny; AST; YuNet; yolo11n and yolo11m; EmotiEffLib enet_b0_8_va_mtl ONNX): source (HF repo or GitHub release URL), version or commit, licence (check each upstream licence page when filling), sha256 (`TBD: run scripts/download_weights.sh` until known), and which machine uses it.
  - A "Known limitations" section: `person_count` includes the audience; `has_speech` can include sung lyrics; transcript peak memory on CUDA is approximate (ctranslate2 allocates outside torch); people and expression runtimes overlap (shared pass), so do not sum them; expression labels are unvalidated observable-expression suggestions, never feelings.
  - A note that the GitHub sources are a clarification of Principle II (spec FR-009a).
- [X] T054 **Checkpoint 3**: `uv run pytest` passes; `bash -n` passes on both scripts; `config/official_runs.yaml` has the four empty sections; `docs/weights.md` exists. Running `tpa-index calibrate --stage audio --split dev --dry-run` in the T033 scratch setup (after listing the mock audio runs as official in a scratch `TPA_OFFICIAL_RUNS` file) prints thresholds for the five audio labels.

---

## Phase 6: Polish and cross-cutting

- [X] T055 [P] Update `specs/004-signals/contracts/interfaces.md` with the deviations made during implementation: add `scripts/download_weights.sh` to the Scripts table; document the CUDA-failure message from T023 (it names the missing library, with no automatic CPU fallback); record any signature changes.
- [X] T056 Final hygiene:
  - run `uv run pytest` and `uv run pytest tests/test_repo_hygiene.py`;
  - `git status --short` shows no data files, no `*.parquet`, `*.wav` or `transcript*.json`, and no `.local.yaml`;
  - `git diff --stat -- .venv` is empty and no file under `.venv` was edited;
  - grep `src/ tests/ config/` for feelings or emotion wording in labels, prompts, logs or docstrings (FR-030), allowing model class names only inside `signals.expression.mapping`.
- [ ] T057 (Team, after merge, on the HPC login node) Run `scripts/download_weights.sh`, paste the printed sha256 values into `config/pipeline.yaml` (`models.people.hpc.sha256`, `.mac.sha256`, `models.expression.model.sha256`) and `docs/weights.md`, and keep `config/pipeline.yaml` identical on HPC and in the repo. Checksums of public weights may be committed.

---

## Dependencies and execution order

- **Phase 1**: T001 must run before T004 (it captures the 0.4.0 checklist). T002 and T003 can run anytime.
- **Phase 2**: T004 → T005 → T006 → T007. T008 and T009 follow T004 to T007. T010 and T011 are independent of the config tasks. T012 follows T006. T013 follows T012. T014 and T015 follow T004. T016 can run anytime in Phase 2.
- **Phase 3 (Checkpoint 1)** needs Phase 2: T017 → T018; T019; T020 needs T012, T017 and T019; T021; T022 needs T016 and T020; T023 → T024; T025; T026 needs T020, T023 and T025; T027 to T031 need T026; T032 needs T023 and T025; T033 is last.
- **Phase 4 (Checkpoint 2)** needs Checkpoint 1 (it shares `run_signals` and the CLI): T034; T035 and T036 need T006; T037 needs T035 and T036; T038 needs T034; T039 needs T038; T040, T041 and T043 need T038 and T026; T042 needs T035 and T036; T044 needs T029 and T038; T045 is last.
- **Phase 5 (Checkpoint 3)**: T046 → T047; T048 needs T038; T049 → T050; T051, T052 and T053 can run anytime after Phase 2; T054 is last.
- **Phase 6** follows Checkpoint 3. T057 is a team task after merge.

## Parallel examples

- Phase 2: T008, T009, T010, T011, T014 and T015 (different files) once T004 to T007 are done.
- Phase 3: T018, T021 and T024 alongside T019 and T025. After T026: T027, T028, T029, T030 and T031 together (different test files or test functions).
- Phase 4: T034 alongside T035 and T036. After T038: T040, T041 and T043 together.
- Phase 5: T051, T052 and T053 in parallel with T046 to T050.

## Implementation strategy

1. **MVP is Checkpoint 1**: job A (transcript and audio) works end to end with all its privacy tests. That alone unblocks the HPC smoke test (quickstart step 4), which de-risks ctranslate2 on CUDA 13 early.
2. **Checkpoint 2** adds job B in the same pass structure.
3. **Checkpoint 3** makes the results usable (read path, calibrate) and runnable on HPC (Slurm, official sections, weight docs).
4. Stop at any checkpoint with a green test suite. Each one is independently mergeable, because no official run, default read or blind-mode path changes at any point.
