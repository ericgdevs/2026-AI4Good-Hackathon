# Research: Extra Per-Segment Signals (004)

Every item was checked against the code on this branch and the installed packages (`uv run --frozen`). No weights were downloaded and no data was read.

## R1. How 002 computes config_hash, and a per-stage hash

- **Today**: `config.config_hash(labels, pipeline)` is the first 12 hex characters of sha256 over `json.dumps({"labels": labels, "pipeline": pipeline}, sort_keys=True)`. `runs.track_run` writes it into `runs.config_hash` (default: the whole of labels plus pipeline). Nothing reads it back: grep finds no comparison in calibrate, moments or `read_official`, and the only other users are `mock.py` and the tests.
- **Decision**: add `stage_config_hash(stage, labels, pipeline, video)`. It calls the existing `config_hash` on only what the stage uses, and the stage passes the result to `track_run(cfg_hash=...)`. It covers:

  | stage | labels part | pipeline part |
  |---|---|---|
  | transcript | none | effective `window_s`, `overlap_s`; `audio_sample_rate`; `models.transcript` |
  | audio | names of `audio` group labels | effective `window_s`, `overlap_s`; `audio_sample_rate`; `models.audio`; `signals.audio` |
  | people | none | effective `window_s`, `overlap_s`, `frames_per_window`; `models.people`; `signals.people` |
  | expression | names of `expression` group labels | effective `window_s`, `overlap_s`, `frames_per_window`; `min_face_px`; `models.expression`; `signals.expression` |

  "Effective" means `effective_settings(pipeline, video)`, so a `by_video` override changes only that video's hash. Prompts are never part of a stage hash, so editing a prompt reruns nothing.
- **Rationale**: one existing function is reused. The column keeps its meaning ("the config this run depended on"), just scoped.
- **Alternatives**: the global hash (rejected: any label edit, including this feature's own 0.5.0 bump, would rerun every pair); hashing the code version too (rejected: `git_commit` is already recorded, and `--force` handles code fixes).

## R2. The 0.5.0 bump does not affect 002

- **predict**: `score.predict_video` scores only `model_labels(labels, set(pipeline["visual_groups"]) | {null_group})`, where `visual_groups = [category, actor, activity]`. Audio and expression labels, including the new `singing_voice`, are never scored by SigLIP. A new test runs predict with a fake embedder on synthetic clips and asserts that no `audio` or `expression` rows appear.
- **calibrate and moments**: neither compares `runs.config_hash`. Both read official runs by run ID. A new test writes mock official runs with `config_hash = "000000000000"`, then runs `tpa-index moments` and `tpa-index calibrate --dry-run` and asserts exit 0 with the expected rows.
- **Blind checklist**: `review.checklist` reads only `app.blind_groups` (activity, category, actor, indicator) plus the primary group, so an audio label cannot change it. A test compares the checklist from the 0.5.0 labels against a frozen snapshot of the 0.4.0 checklist (`tests/data/blind_checklist_0.4.0.json`, generated from 0.4.0 before the bump).
- **mock**: `mock._predictions` scores every model label for the `mock` model, so mock data gains `singing_voice` rows. This is expected, and nothing official is involved.

## R3. Transcript: faster-whisper 1.2.1

- **Decision**: `WhisperModel(size, device, compute_type, local_files_only=True)`, resolving `Systran/faster-whisper-<size>` from the HF cache. HPC: `large-v3`, `cuda`, `float16`. Mac: `base`, `cpu`, `int8` (`tiny` in slow tests). `transcribe(wav_path, vad_filter=True, word_timestamps=True, condition_on_previous_text=False)`. The VAD model (`silero_vad_v6.onnx`) ships inside the package, so nothing is downloaded.
- **Word to segment**: a word belongs to every segment with `start_s <= (w.start + w.end) / 2 < end_s`; the last segment includes its end. `transcript_chars` is the sum of `len(word.strip())` over assigned words.
- **Leak control**: the `faster_whisper` logger is set to WARNING (its INFO lines carry durations, not text). The stage body catches any exception and re-raises `TranscriptError(type(e).__name__)` `from None`, so neither the message nor the exception chain can carry text. Text is held only in local variables and written once, to `transcript.json`.
- **Risk, flagged**: ctranslate2 4.8.3 wheels are built against CUDA 12 and cuDNN 9, while torch on HPC is cu130. If ctranslate2 cannot find `libcublas.so.12` on the node, loading on `cuda` fails. The loader turns that into exit 3 before any video ("transcript model could not load on cuda"), and `models.transcript.hpc_device` can be set to `cpu` (int8) as a slower fallback. This must be tried on HPC with one promo video before the full job.
- **Alternatives**: transformers Whisper (rejected: slower, more memory, no built-in VAD); per-window transcription (rejected by the spec).

## R4. Audio: AST

- **Decision**: `MIT/ast-finetuned-audioset-10-10-0.4593` through `ASTForAudioClassification` and `AutoFeatureExtractor`, with `local_files_only=True`, on cuda/mps/cpu. Each window's waveform is read from the temp WAV, zero-padded (silence) or cut to the model's input of 1024 frames × 10 ms = 10.24 s (163 840 samples at 16 kHz), then batched by `models.batch_windows[device]`. Probabilities are `sigmoid(logits)`. A label's score is the max over its mapped class indices, which are resolved once from `model.config.id2label`.
- **Startup check**: every configured display name must be in `id2label.values()`, or exit 3 naming the missing class (AudioSet names are not sensitive).
- **Why pad the waveform rather than rely on the extractor**: the feature extractor pads in log-mel space after normalization, which is not silence. Zero samples are.
- `predictions.model` = the model ID's last path part, as SigLIP does: `ast-finetuned-audioset-10-10-0.4593`.

## R5. Shared audio extraction

- **Decision**: decode the audio track with PyAV (its bundled ffmpeg libraries) to 16 kHz mono and write it into one WAV file in the scratch folder using PyAV's `wav` muxer (pcm_s16le). Transcript passes the path to faster-whisper. Audio reads per-window slices with `soundfile` (already installed), so the whole track is never held as float32 next to the models.
- **Why not the `ffmpeg` command**: this matches the user's "ffmpeg to 16 kHz mono temp file" through the same ffmpeg libraries PyAV wraps, without assuming an `ffmpeg` binary on HPC compute nodes (only the Mac has one, from Homebrew, for 003). Output is the same.
- A video with no audio track gets a WAV of zeros of the video's length, so both stages follow the 002 "silence" convention.
- **Lifetime**: the file is named `tpa-audio-<pid>-<video_id>.wav` in `scratch_dir()` (`TPA_SCRATCH_DIR`, else `$TMPDIR`, else the system temp; refused if inside the repo). It is created before the first audio stage, and a `finally` deletes it after the last. At command start, any `tpa-audio-<pid>-*` file whose pid is not alive (`psutil.pid_exists`) is removed. Slurm gives each job its own `$TMPDIR`, so jobs A and B never see each other's files.

## R6. People: YOLO11 (ultralytics 8.4.184)

- **Decision**: `YOLO(<absolute path>)` with `predict(frames, classes=[0], conf=0.4, verbose=False)`. We read only `len(result.boxes)` per frame and immediately drop the result objects. `person_count = floor(median(counts))`, which means halves round down, as FR-018 requires.
- **No download**: ultralytics downloads when given a bare name, so the loader passes an absolute path under the weights folder, checks it exists, and checks its sha256 first. `YOLO_OFFLINE=1` is set in the Slurm script and by the loader.
- HPC: `yolo11m.pt`. Mac: `yolo11n.pt`. GitHub release assets (`ultralytics/assets`, version-pinned by sha256 in config).

## R7. Face detection: YuNet through OpenCV

- **Decision**: `cv2.FaceDetectorYN.create(model, "", (w, h), score_threshold=0.8)` (present in opencv-python 5.0.0.93), with `setInputSize` per frame size. Frames are RGB from 002's sampler and are converted to BGR for YuNet. Each detection row is `[x, y, w, h, 10 landmark values, score]`. We read the box only long enough to cut the in-memory crop, then discard the row, landmarks included. Faces with `min(w, h) < min_face_px` are dropped. Boxes are clipped to the frame.
- **Weights**: `face_detection_yunet_2023mar.onnx` from HF `opencv/face_detection_yunet` via `hf_hub_download(..., local_files_only=True)`.
- **Nothing beyond opencv is needed for YuNet.**
- Frames are 002's letterboxed, full-resolution frames. Letterboxing pads and does not rescale, so `min_face_px` is in source pixels.

## R8. Expression: EmotiEffLib 1.1.1

- **Findings**: the torch engine loads a pickled timm model (`torch.load(weights_only=False)`), but `timm` is not installed. The ONNX engine needs `onnx` and `onnxruntime`, which are installed (1.23.2 and 1.31.0). Both engines call `utils.download_model`, which fetches from GitHub into `~/.emotiefflib` if the file is missing.
- **Decision**: use `EmotiEffLibRecognizerOnnx("enet_b0_8_va_mtl")` on CPU (faces are small, so CPU is enough on both machines). Before constructing it, the loader points `emotiefflib.facial_analysis.get_model_path_onnx` at our pinned, sha256-checked file under the weights folder, so the library can never download. This is marked in code as `shortcut: library has no path argument; drop when it gets one`.
- Classes come from the library's `idx_to_emotion_class` for 8-class models: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise. `predict_emotions(crops, logits=False)` softmaxes the 8 emotion columns and returns valence and arousal as the last 2 columns, which we slice off and never store. A label's per-face score is the sum of its mapped classes' probabilities (no renormalization). A label's segment score is the mean over all usable faces in the window's frames. `faces_detected` is the max usable faces in any one frame.
- `predictions.model` = `enet_b0_8_va_mtl`.
- **No new dependency.** Adding `timm` for the torch engine was rejected: a new dependency, and pickle loading.

## R9. Running people and expression in one pass

- **Problem**: full-resolution frames for a long video do not fit in memory at once (1080p letterboxed ≈ 11 MB per frame; 4 frames × about 600 windows per hour ≈ 26 GB). The two stages must share a streaming pass, yet each must fail independently and write its own run.
- **Decision**: a single pass over `segment_video(..., with_audio=False)`. For each window, each active stage's `update(window)` is called inside its own try/except. A stage that raises is marked failed (error redacted, type name only) and is not called again for that video, while the other continues. Only small per-segment results are kept. After the pass, each surviving stage writes its run through `track_run`. The pass's wall time and peak memory are measured before any run starts, and handed to `track_run` through two new optional `RunContext` fields (`runtime_s`, `peak_mem_gb`). The two runs therefore report the same shared-pass runtime, documented as "overlapping, do not sum".
- Transcript and audio run one after the other (whole-file transcription, then per-window scoring). Each uses its own `track_run`, and the shared extraction happens inside the first one's block.

## R10. Reading signals

- `read_official(table, stage="audio")` already returns one stage's official rows, so stage predictions need no new code.
- **New**: `read_signals(root=None)` starts from the default `read_official("segments")` (falling back to `stage="segment"` when it is empty, as `review._official` does). For each of the four stages it takes `_official_by_video(load_official(), stage)` and copies that stage's column(s) by `segment_id`. If an official stage run lacks one of the official segment IDs for its video, it raises a `SchemaError` naming the stage and video. Default `read_official` is not touched.

## R11. Resume

- `find_complete_run(stage, video_id, model, cfg_hash, root)` scans complete run folders: `runs.parquet` with that stage, model and config_hash, and `videos.parquet` containing the video. If one exists, the pair is skipped ("skipped (done: <run_id>)"). `--force` ignores it.

## R12. Weights location and pinning

- Hugging Face models (faster-whisper, AST, YuNet) resolve from `HF_HOME` (HPC: `~/hf-cache`), offline.
- GitHub-release weights (YOLO11, EmotiEffLib) live in `weights_dir()` = `TPA_WEIGHTS_DIR`, else `$HF_HOME/tpa-weights`. In config, each has `file` and `sha256`. A missing file, missing sha256 or wrong sha256 exits 3 before any video. The sha256 values are filled in when the team pre-downloads on the login node (quickstart step 1). Committed config ships `sha256: null`, so a real run refuses to start until weights are pinned.
- Every source, version and licence is listed in `docs/weights.md`, which the 006 model card includes.

## R13. Mac memory

- Default tests use fakes only, so no model is loaded (`addopts = -m 'not slow'` already). Slow tests (`tests/test_signals_slow.py`) load one model per test, use the smallest sizes (whisper `tiny`, `yolo11n`, AST, YuNet with ONNX EmotiEffLib), run on 2 to 4 synthetic frames or seconds, then `del` the model, `gc.collect()` and empty the MPS cache. They skip when the weights are not in the local cache.
