# Research: Shared Data Foundation

All Technical Context items are resolved. Decisions below follow the team's tech choices; where one changed after checking, the change is marked **Deviation**.

## 1. Torch wheels on Linux (CUDA) and macOS

- **Decision**: One `pyproject.toml`. On Linux, `torch`, `torchvision`, and `torchaudio` come from the PyTorch **cu130** index via `[tool.uv.sources]` with `marker = "sys_platform == 'linux'"` and an `explicit = true` index. macOS uses default PyPI wheels (MPS).
- **Deviation**: The input asked for CUDA 12.8 wheels. Checked on 2026-10-23: the cu128 index stops at `torch 2.11.0`, while the lock pins `torch 2.14.1`. The cu130 index has exactly the locked versions (`torch 2.14.1`, `torchvision 0.29.1`, `torchaudio 2.11.0`). UNF HPC's CUDA driver 13.0 runs cu130 wheels, and Blackwell (sm_120) is supported.
- **Alternatives**: cu128 with torch pinned to 2.11 on Linux only (rejected: different torch versions on the two machines break "same code"); cu129 (rejected: stops at 2.13).

## 2. Audio extraction

- **Decision**: Decode audio with PyAV (`av.AudioResampler(format="flt", layout="mono", rate=16000)`), once per video, into one float32 array. Slice it per window.
- **Deviation**: The input said "ffmpeg for audio extraction". PyAV *is* ffmpeg: it bundles libavcodec and libswresample in its wheel. Using it avoids depending on an `ffmpeg` binary being on PATH on HPC nodes, where availability is unknown. Output is identical: 16 kHz mono.
- **No audio track**: return zeros of the right length and set a per-video `has_audio=False` flag. Do not raise.
- **Memory**: 1 h of audio at 16 kHz float32 is about 230 MB. That is acceptable on a 16 GB laptop. Shortcut: switch to per-window decoding only if videos exceed about 2 h.

## 3. Frame sampling and letterbox

- **Decision**: PyAV. For each window, compute N timestamps at the centres of N equal sub-intervals (`start + (i + 0.5) * len / N`), seek to the keyframe before each one, and decode forward to the first frame at or after it. Convert to RGB `uint8`, then pad to a square with black borders, centred (never crop). No resize: later model processors resize.
- **Unreadable frame**: use the nearest decoded frame in the window. If none decodes, fail the video with a redacted error.
- **Alternatives**: OpenCV `VideoCapture` (rejected: less reliable timestamp seeking on variable-frame-rate files); decoding every frame (rejected: Constitution II efficiency).

## 4. Window boundaries

- **Decision**: `stride = window - overlap` (default 6 s). Starts are `0, stride, 2·stride, …` while `start < duration`, with `end = min(start + window, duration)`.
  - A trailing window shorter than `window / 2` is dropped, and the previous segment's `end_s` is extended to `duration`.
  - A video shorter than one window gives a single segment `[0, duration]`.
- `start_ms = round(start_s * 1000)`. Duplicate IDs are caught by validation.

## 5. Peak memory and hardware name

- **CUDA**: `torch.cuda.reset_peak_memory_stats()` at start and `torch.cuda.max_memory_allocated()` at end. Hardware is `torch.cuda.get_device_name(0)`.
- **MPS**: `torch.mps.driver_allocated_memory()` at end. This is current allocation, not true peak. It is combined with the process peak resident memory (RSS) by taking the max, so it never under-reports host memory. Hardware is `sysctl -n machdep.cpu.brand_string` (for example "Apple M3 Pro"), because torch exposes no chip name.
- **CPU fallback**: peak RSS via stdlib `resource.getrusage(RUSAGE_SELF).ru_maxrss`, in bytes on macOS and KiB on Linux.
- **Deviation**: The input said "psutil RSS". psutil gives *current* RSS, but `ru_maxrss` gives the true process *peak* with no extra dependency. Limit: it is a process-lifetime peak, so it is accurate when one stage runs per process (the CLI and Slurm pattern).
- **Unmeasurable**: record `null` and log the reason.
- **Torch import**: lazy (inside run tracking only), so the data guard and `check` start in under 1 s (SC-004).

## 6. Config hash

- **Decision**: `sha256(json.dumps({"labels": labels, "pipeline": pipeline}, sort_keys=True, default=str))[:12]`, computed over the loaded dicts. Video metadata is excluded because it is local-only and would leak through hash comparisons across machines.

## 7. Schema validation (no extra library)

- **Decision**: The pyarrow schemas in `schema.py` are the single source of truth, plus a small per-table dict of enum columns, 0–1 columns, and primary keys. `validate(table_name, data)` does the following:
  1. exact column set: missing or extra columns are an error, which also blocks identity columns;
  2. `pa.Table.from_pylist` or `.cast(schema)` (type errors);
  3. null checks on non-nullable fields;
  4. enum membership (`phase` values come from `pipeline.yaml`);
  5. 0–1 range checks;
  6. `end_s > start_s`;
  7. unique primary key.
- The checks run before any bytes are written, so a failed write leaves no file.

## 8. Storage layout and immutability

- **Pipeline outputs**: `{OUTPUT_DIR}/runs/{run_id}/{table}.parquet`. Writing to an existing path raises an error. Readers use DuckDB `read_parquet('{OUTPUT_DIR}/runs/*/{table}.parquet')`.
- **Run ID**: `{stage}-{YYYYmmddTHHMMSS}-{4 hex}`.
- **Reviews**: `{OUTPUT_DIR}/reviews.duckdb`, table `reviews`, created from the pyarrow schema. The API exposes `append_reviews` only, with no update or delete.
- **Concurrency**: DuckDB allows one writer process. The review app is one process, and writes are serialised with a lock in that process. A second process opening the file for write gets a clear error, not silent loss.

## 9. Data guard and redaction

- **Decision**: `python-dotenv` loads `.env` from the repo root.
  - `DATA_DIR` and `OUTPUT_DIR` must be set, and are checked after `expanduser().resolve()` (which follows symlinks). They must not satisfy `is_relative_to(repo_root)`, where `repo_root = Path(__file__).resolve().parents[2]`.
  - On failure, exit with a message naming the setting, not the path.
- **Redaction**: `redact(text)` replaces:
  - the resolved and unresolved `DATA_DIR` and `OUTPUT_DIR` strings with `<DATA_DIR>` and `<OUTPUT_DIR>`;
  - each filename in `videos.local.yaml` (relative path, basename, and stem) with `<video:{video_id}>`;
  - any remaining `<DATA_DIR>/…` token tail with `<file>`.
- **Where it applies**: `install_redaction()` adds a `logging.Filter` on the root handler and replaces `sys.excepthook`, so third-party exception text and tracebacks are covered.
- **Shortcut**: stems shorter than 4 characters are not redacted, to avoid mangling unrelated text. Revisit if real names are that short.
- **Discovery**: files under `DATA_DIR` that are not in `videos.local.yaml` are reported only as a count.
- **Alternatives**: masking in each `raise` site (rejected: misses library errors).

## 10. Video discovery

- **Decision**: `DATA_DIR.rglob("*")`, keeping video extensions (`.mp4 .mov .m4v .mkv .avi`). Skip any path where any part starts with `.` (this covers macOS `._*` resource forks and `.Trashes`).

## 11. Per-video overrides

- **Decision**: `pipeline.yaml` has `overrides.by_kind.{kind}` and `overrides.by_video.{video_id}`, each limited to `window_s`, `overlap_s`, and `frames_per_window`. Precedence is defaults < kind < video_id. Unknown keys fail validation.

## 12. Moment merging (pure logic, no inference)

- **Decision**: For each segment, the primary candidate is the top label in `moments.primary_group` (default `activity`) with score ≥ `label_threshold`.
  - Consecutive segments with the same primary label merge when the gap between them is ≤ `max_gap_windows × window_s`.
  - Merged runs shorter than `min_windows × window_s` (default 1 window) are dropped.
  - `secondary_labels` are labels from `moments.secondary_groups` whose mean score is ≥ threshold. `indicators` are labels from group `indicator` whose mean score is ≥ threshold.
  - `confidence` is the mean primary score.
  - `needs_review` is set when confidence < `needs_review_below`, or when the runner-up primary label is within `conflict_margin` of the top label.
- The mock generator calls the same function, so mock moments follow the real rules.

## 13. Synthetic test clips

- **Decision**: In a pytest session fixture, encode 20 s clips at 10 fps with PyAV into `tmp_path`, which is outside the repo:
  - a 160×90 landscape clip and a 90×160 portrait clip, both with a 440 Hz sine on an AAC track;
  - a third clip with no audio track.
- Codec is `mpeg4`, which is native to ffmpeg and always present in PyAV wheels, unlike `libx264`.

## 14. Package rename

- **Decision**: Rename the project to `tpa-index` and move `src/2026_ai4good_hackathon/` to `src/tpa_index/`. Change the script to `tpa-index = "tpa_index.cli:main"`, then run `uv lock`. `uv_build` infers the module name `tpa_index` from the project name.

## 15. Slurm

- **Decision**: `scripts/slurm/job.sbatch` with `--partition=gpu`, `--gres=gpu:6KSE:1`, `--cpus-per-task=8`, `--mem=32G`, `--time=02:00:00`, `--job-name=tpa-index`, and `--output=slurm-%x-%j.out`. Add `slurm-*.out` to `.gitignore`.
- The body runs `cd "$SLURM_SUBMIT_DIR"`, then `nvidia-smi -L`, then `uv sync --frozen`, then `uv run "$@"`. Usage: `sbatch scripts/slurm/job.sbatch tpa-index check`.
- Logs are already redacted by the app.
