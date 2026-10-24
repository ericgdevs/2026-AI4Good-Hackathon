# Contract: Python API, CLI, Scripts (004)

This extends the 001 and 002 interface contracts, and everything there still holds. Existing signatures are unchanged except where marked *(changed)*. Every change there is backwards compatible.

## Python (`tpa_index`)

```python
# config.py (new)
stage_config_hash(stage: str, labels: dict, pipeline: dict, video: dict) -> str   # 12 hex; scoped per research R1
weights_dir() -> Path                       # TPA_WEIGHTS_DIR, else $HF_HOME/tpa-weights
pinned_weight(entry: dict) -> Path          # {file, sha256} -> verified path; ConfigError if missing/unpinned/mismatch
SIGNAL_STAGES = ("transcript", "audio", "people", "expression")

# runs.py (changed, backwards compatible)
RunContext.runtime_s: float | None = None    # if set, recorded instead of the measured wall time
RunContext.peak_mem_gb: float | None = None  # if set, recorded instead of the measured peak

# schema.py (new)
find_complete_run(stage, video_id, model, cfg_hash, root=None) -> str | None
read_signals(root=None) -> pa.Table
#   official segments (default read_official, else stage="segment") with person_count, faces_detected,
#   has_speech, transcript_chars taken from each stage's official run; null where none.
#   SchemaError if an official stage run lacks an official segment_id of its video.
# read_official and read_table: unchanged

# signal_models.py (new): thin wrappers; all load from local files only; each raises ConfigError on load failure
class Transcriber(Protocol):
    name: str
    def transcribe(self, wav_path: Path) -> list[dict]: ...    # [{"start","end","text","words":[{"start","end","word"}]}]
class AudioTagger(Protocol):
    name: str
    labels: list[str]                                           # audio labels, in mapping order
    def score(self, clips: list[np.ndarray]) -> np.ndarray: ... # (n, len(labels)) in 0-1; clips 16 kHz float32
class PersonCounter(Protocol):
    name: str
    def count(self, frames: np.ndarray) -> list[int]: ...       # (N,S,S,3) RGB -> person count per frame
class ExpressionScorer(Protocol):
    name: str
    labels: list[str]                                           # mapped expression labels
    def faces(self, frame: np.ndarray) -> list[np.ndarray]: ... # usable in-memory crops (score, min size applied)
    def score(self, crops: list[np.ndarray]) -> np.ndarray: ... # (n_faces, len(labels)), mapped sums, no renorm

WhisperTranscriber(size, device, compute_type)
AstTagger(model_id, device, mapping, batch)
YoloPersonCounter(weights_path, device, conf)
FaceExpression(yunet_path, emotiefflib_path, model_name, mapping, face_score, min_face_px)

# signals.py (new)
assign_words(words: list[dict], segments: list[dict]) -> dict[str, tuple[bool, int]]   # segment_id -> (has_speech, chars)
median_count(counts: list[int]) -> int                                                  # floor(median)
aggregate_expression(per_frame_scores: list[np.ndarray], labels) -> tuple[int, dict[str, float] | None]
#   (faces_detected = max faces in one frame, mean scores or None when 0 faces)
official_segments(video_id, root=None) -> list[dict]       # FR-004 source of truth; ConfigError if none
extract_audio(video, scratch: Path) -> Path                # PyAV -> 16 kHz mono pcm_s16le WAV
scratch_dir() -> Path                                      # TPA_SCRATCH_DIR / $TMPDIR / system temp; never in repo
clean_stale_scratch(scratch: Path) -> int                  # removes tpa-audio-<dead pid>-*.wav
run_signals(videos, stages, models, pipeline, labels, force=False, root=None) -> dict[tuple[str, str], str]
#   -> {(video_id, stage): "done:<run_id>" | "skipped:<run_id>" | "failed:<ErrorType>"}; never raises for one pair
class TranscriptError(Exception)                           # message = exception type name only
```

The orchestration in `run_signals`, for each video:
1. Read the official segments and check the IDs from `windows(...)` against them (FR-004).
2. If transcript or audio is selected, then inside `try/finally`: extract the WAV once, run transcript (its own `track_run`), run audio (its own `track_run`), and delete the WAV in `finally`.
3. If people or expression is selected: one streaming pass over the frames, with each stage's `update` isolated per stage. Then write each surviving stage's run with the shared pass's runtime and peak memory.

## CLI

| command | does | exit codes |
|---|---|---|
| `tpa-index signals [--stages transcript,audio,people,expression] [--videos PATH] [--video-id ID …] [--kind KIND …] [--force]` | loads each selected model once, then runs the selected stages per video (see orchestration). Prints one line per (video, stage): `done <run_id>`, `skipped (done: <run_id>)` or `failed <ErrorType>`, by video ID and stage only | 0; 1 if any pair failed (failed pairs listed); 2 guard; 3 config, weights or mapping error before any video |
| `tpa-index calibrate --stage audio\|expression --split dev [...]` *(changed: any stage name accepted; one generic line added)* | same as 002. Also prints `labels in scored groups without scores: …` (for expression: `confident`). Writes the stage model's block to `thresholds.local.yaml` and keeps the other models' blocks | 0; 3 if eval split or no official dev runs |

The `--stages` default is all four. An unknown stage name exits 3.

## Scripts

| script | does |
|---|---|
| `scripts/slurm/signals.sbatch [signals args]` *(new)* | `--partition=gpu --gres=gpu:6KSE:1 --cpus-per-task=8 --mem=64G --time=03:00:00`. Sets `HF_HOME=$HOME/hf-cache`, `HF_HUB_OFFLINE=1`, `YOLO_OFFLINE=1`, `TPA_WEIGHTS_DIR=$HOME/hf-cache/tpa-weights`. Runs `uv sync --frozen`, then `uv run tpa-index signals "$@"`. Submitted twice: `--stages transcript,audio` and `--stages people,expression` |
| `scripts/sync_outputs_from_hpc.sh` | **unchanged**: the Parquet-only allowlist keeps `transcript.json` on HPC (test added) |
| `scripts/sync.exclude` *(changed)* | adds `transcript*.json` (defence in depth for the code sync) |

## Logging and errors (privacy)

- Log lines name the video ID, stage, run ID and error type only. They never contain transcript text, AudioSet scores per window, or face information.
- The `faster_whisper` logger is set to WARNING. The transcript stage re-raises any error as `TranscriptError(<type name>)` `from None`.
- The existing redaction (`install_redaction`) stays on for every command.
