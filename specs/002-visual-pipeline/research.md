# Research: Visual Pipeline

Checked on 2026-10-23 against the locked environment (transformers 5.20.2, torch 2.14.1) and the Mac's local Hugging Face cache, which holds `google/siglip2-base-patch16-224`, `google/siglip2-so400m-patch14-384` and `microsoft/xclip-base-patch32`. On HPC the same weights are in `~/hf-cache`.

## 1. Loading SigLIP2

- **Decision**: Load with `AutoModel.from_pretrained(id, local_files_only=True, dtype=...)` and `AutoProcessor.from_pretrained(id, local_files_only=True)`. Both checkpoints report `model_type: siglip`, so they load as `SiglipModel`; the NaFlex variants are not used.
  - Their processor (`SiglipImageProcessor`) resizes straight to a square (224 or 384) and has no center crop. Our 001 frames are already letterboxed to a square, so nothing is cropped or stretched (FR-003).
  - Set `HF_HUB_OFFLINE=1` in the Slurm job and the CLI as a second guard against network use.
- **Alternatives**: `pipeline("zero-shot-image-classification")`. Rejected: it applies its own scoring and can't batch frame averaging.

## 2. Text side and prompt ensembling

- **Decision**: Tokenize with `padding="max_length", max_length=64`. SigLIP was trained this way, and shorter padding changes the scores. Lowercase the prompts.
  - Compute `text_embeds = model.get_text_features(...)` once per run (FR-002), L2-normalize each prompt, average a label's 3–5 prompts, then **renormalize**. A plain average of unit vectors is shorter than unit length and would push the score down.
- **Alternatives**: Score each prompt and average the probabilities. Rejected: the result depends more on how prompts are worded, and it costs more.

## 3. Image side and the 0–1 score

- **Decision**: `image_embeds = get_image_features(pixel_values)`. L2-normalize each frame, average a window's N frames, then renormalize. The score is `sigmoid(cos · exp(logit_scale) + logit_bias)`, using the model's own learned scale and bias.
  - Each label gets its own independent multi-label score, with no softmax across labels.
  - Compute the similarity and sigmoid in float32, even when the model runs in bf16.
- **Score range (risk)**: SigLIP's raw sigmoid scores are often low for true zero-shot matches (around 0.05–0.3). With the 001 default `label_threshold: 0.5`, very few labels may clear the bar.
  - Mitigation: per-label thresholds (§15) and `tpa-index calibrate` (§16), which sets them from official dev-split predictions. The global `label_threshold: 0.5` remains the fallback. Proper calibration belongs to 005.
- **Alternatives**: Softmax across labels. Rejected: the spec needs independent multi-label scores, and softmax forces the scores to compete.

## 4. Device, precision and batching

- **Decision**: Device comes from `runs.device()` (001). Use bf16 on CUDA and float32 on MPS and CPU.
  - Batch by windows: `batch_windows: {cuda: 64, mps: 8, cpu: 4}` in `pipeline.yaml`. That is `B × frames_per_window` images per forward pass, so 256 images at batch 64 on CUDA.
  - Wrap inference in `torch.inference_mode()`.
- **Why**: Decoding happens on the CPU through the 001 segmenter, so the GPU waits on decoding. The default batch sizes balance memory against throughput, and an out-of-memory error is fixed in config (spec edge case).

## 5. Predict reuses the 001 segmenter

- **Decision**: Add `with_audio: bool = True` to `segment_video`. Predict passes `False`, which skips decoding the whole audio track; predict has no use for it.
  - Predict consumes the windows, embeds them in batches, and fills `is_idle` on each window's row before writing.
  - Only frames from the current batch are held in memory.

## 6. Idle detection (spec Q2 = B)

- **Decision**: A segment is idle only when both of these hold:
  - `idle_score ≥ idle.score_threshold`, where `idle_score` is the SigLIP sigmoid score of the ensembled `idle.prompts` (setup, waiting, empty room) in `pipeline.yaml`. These prompts are not labels, so no prediction rows are written for them.
  - `change < idle.change_threshold`, where `change` is the mean absolute difference between consecutive frames. Each frame is converted to grayscale, downscaled to 64×64, and scaled to [0, 1].
- A window with fewer than 2 frames is never idle.
- Starting values: `score_threshold: 0.1`, `change_threshold: 0.02`. Both are tuned on the dev split.
- **Alternatives**: Optical flow. Rejected: heavy, and an image difference is enough to tell "nothing moves" apart.

## 7. `uncertain_mixed` (spec Q1 = A)

- **Decision**: After model scoring, for each segment take the activity labels' scores:
  - `flag = top1 < label_threshold or (top1 - top2) < uncertain_margin`. `uncertain_margin` defaults to 0.1 in `pipeline.yaml`.
  - Write one row: label `uncertain_mixed`, group `status`, score `1.0` or `0.0`, model `"rules"`, same `run_id`.
  - In `labels.yaml`, add `source: rules` to `uncertain_mixed`; a rules label needs no prompts. Remove `uncertain` and `mixed`, and move `guideline_version` to `0.2.0`.
  - `needs_review` is also `source: rules`, with its prompts removed (§17). After this, the status group has no model-scored labels, so `visual_groups` is `[category, actor, activity]`.
  - "Clears the threshold" means a score ≥ that label's own threshold (§15). top1 is the best activity that clears it; top2 is the next-highest activity score overall.

## 8. X-CLIP (optional, off by default)

- **Decision**: `models.motion.enabled: false`. When it is on, X-CLIP runs on the same windows with its own frame sampling.
  - `xclip-base-patch32` was trained with exactly 8 frames, and its frame-position embedding is sized to `num_frames`. A second `segment_video` pass with `frames_per_window: 8` feeds it.
  - It scores only `models.motion.labels` (default `dancing`, `acting`).
  - X-CLIP has no sigmoid bias, so its 0–1 score is a softmax over `[motion labels…, "other activity"]`. The `"other activity"` prompt is in config.
  - Rows are written under model `xclip-base-patch32`. These scores are relative, not independent; that is documented so evaluation compares them fairly.
- **Alternatives**: Feed 4 frames. Not possible: the position embedding shape doesn't match.

## 9. Moments: remove the per-video window shortcut

- **Decision**: `merge_moments(segments, predictions, pipeline, run_id, window_s=None, model=None)`. Predict calls it once per video, with `effective_settings(pipeline, video)["window_s"]` and the main model id (for per-label thresholds).
  - Idle segments are excluded before merging (FR-009).
  - Only the main model's rows are used.
  - The mock generator passes each mock video's effective window as well, which removes its matching shortcut.

## 10. Official runs

- **Decision**: `config/official_runs.yaml` is committed; it holds run IDs only. Its format is `{stage: [run_id, …]}`.
  - Add a new helper, `read_official(table, stage="predict", root=None)`. It resolves the run IDs and calls `read_table(table, root, run_ids=...)`.
  - It raises an error when a listed ID has no complete run folder, or when two listed runs of one stage share a `video_id`.
  - `read_table` keeps its 001 meaning: `None` means all runs. It is the low-level reader for tools and tests. Result-facing code (003, 004, 005, the deck numbers) MUST use `read_official`. This is written into the contract.
  - Mock data for UI work is listed as `predict: [mock-seed0]`.
- **Why not change `read_table`'s default?** That would silently break 001 callers and tests, and leave no way to read everything. A separate name makes the choice visible in code review.

## 11. `tpa-index runs`

- **Decision**: For each complete run folder, print one line with `run_id`, `stage`, `video_id`s (from that folder's `videos.parquet`, comma-joined), hardware, `n_segments`, prediction count, runtime and peak memory.
  - The prediction count comes from the Parquet metadata, so the file is not read.
  - Video IDs are opaque. No filenames or paths are printed.

## 12. Sync scripts

- **Decision**: bash with `set -euo pipefail`. Each script reads `.env` with a `KEY=value` parser and never `source`s it, so values are never executed. Required keys are `HPC_USER` and `HPC_HOST`; the output sync also needs `HPC_OUTPUT_DIR`. `--dry-run` is passed through to rsync as `-n`.
  - **Code sync**: `rsync -az --exclude-from=scripts/sync.exclude ./ $HPC_USER@$HPC_HOST:~/tpa-work/repo/`.
    - Excludes: `.venv/`, `.env`, `.env.*`, `models/`, `__pycache__/`, `outputs/`, `*.mp4`, `*.mov`, `*.m4v`, `*.mkv`, `*.avi`, `*.wav`, `*.mp3`, `*.m4a`, `*.parquet`, `*.duckdb`, `config/videos.local.yaml`, `slurm-*.out`.
    - No `--delete`, so HPC-only files such as logs are never removed.
  - **Output sync**: `rsync -az --ignore-existing --exclude-from=scripts/sync.exclude --exclude=reviews.duckdb $HPC:$HPC_OUTPUT_DIR/runs/ "$LOCAL/runs/"`.
    - It copies only `runs/`. Runs are immutable, so existing local runs are never overwritten.
    - `reviews.duckdb` is never copied, because a copy would overwrite the local review log, which is staff work.
    - `$LOCAL` comes from `uv run python -c "from tpa_index.paths import output_dir; print(output_dir())"`, which applies the data guard.
- **Alternatives**: `--delete` for a mirror. Rejected: risk of data loss on HPC.

## 13. Slurm

- **Decision**: `scripts/slurm/predict.sbatch` is based on `job.sbatch`:
  - `--mem=64G`, `export HF_HOME=$HOME/hf-cache HF_HUB_OFFLINE=1`.
  - Then `uv sync --frozen`, then `uv run tpa-index predict "$@"`.
  - Run as `sbatch scripts/slurm/predict.sbatch [--video-id …]` from `~/tpa-work/repo`.

## 14. Testing without downloads

- **Decision**: An `Embedder` protocol with `embed_texts(list[str]) -> np.ndarray (n, d)`, `embed_images(np.ndarray (n, S, S, 3)) -> np.ndarray (n, d)`, `logit_scale: float`, `logit_bias: float` and `name: str`.
  - The real `SiglipEmbedder` wraps transformers. The unit tests use a `FakeEmbedder`: a deterministic hash of the text, plus the mean colour of the image projected to `d = 8`. The tests can then steer which label wins.
  - One `@pytest.mark.slow` integration test uses `siglip2-base` on a synthetic clip. It is skipped when the weights are not in the local cache.
  - Register the `slow` marker in `pyproject.toml`, and skip slow tests by default with `-m "not slow"`.

## 15. Per-label thresholds

- **Decision**: `threshold(pipeline, model, label) = pipeline.get("label_thresholds", {}).get(model, {}).get(label, pipeline["label_threshold"])`.
  - Thresholds are keyed by **model and label**, because base and so400m score on different scales. A threshold calibrated for one model must not silently apply to the other.
  - The values come from an optional, gitignored `config/thresholds.local.yaml` (`{label_thresholds: {model: {label: t}}}`), merged over `pipeline.yaml` by `load_pipeline`. A hand-set `label_thresholds` in `pipeline.yaml` also works. The local file wins.
  - Validation: each value must be in [0, 1], and each label must exist and be model-scored.
- **Deviation**: The input asked for `calibrate` to write into `pipeline.yaml`. Calibrated thresholds are values derived from footage scores. Constitution I says data-derived outputs live only locally or on HPC, and `pipeline.yaml` is committed and pushed to GitHub. They go to the gitignored `*.local.yaml` file instead. Rewriting `pipeline.yaml` would also strip its comments, because PyYAML doesn't round-trip them.

## 16. `tpa-index calibrate --stage predict --split dev [--top-pct 10]`

- **Decision**: Read `read_official("predictions", stage)` and `read_official("videos", stage)`.
  - Keep only the segments of videos with `split == "dev"`. `--split eval` is rejected outright, with exit 3: eval never feeds a threshold (Constitution VIII).
  - Keep only main-model rows. For each model-scored label, the proposed threshold is the `(100 − top_pct)`th percentile of its dev scores, so about `top_pct`% of dev segments clear it.
  - Labels with fewer than 20 dev segments keep the global fallback and are reported as "too few".
  - Print a table of label, proposed threshold and the number of segments clearing it. The table holds only aggregate numbers that name no video, so it may be shared under Principle III. Write `config/thresholds.local.yaml` (`--dry-run` skips the write).
  - A percentile sets a target rate of positives, not accuracy. It's a starting point until staff reviews give 005 real labels.
- **Alternatives**: Fitting to staff labels. Not possible yet, because there are no reviews; that belongs to 005.

## 17. `needs_review` (rules) and `tpa-index moments`

- **Rule**: a moment's `needs_review` is `confidence < threshold(model, primary_label)` **or** any segment in the moment has `uncertain_mixed = 1`.
  - The old moment-level keys `moments.needs_review_below` and `moments.conflict_margin` are removed, replaced by per-label thresholds and `uncertain_margin`.
  - 001's `test_moments.py` and the mock generator are updated to match.
- **`tpa-index moments [--video-id …]`**: for each official predict run, read its `segments` and `predictions` (main model and rules rows) and rebuild moments with the current thresholds.
  - The `uncertain_mixed` rows are recomputed too, because they depend on thresholds.
  - It writes a **new run per video** with `stage = "moments"`, containing `videos` (copied), `predictions` (rules rows only, recomputed), `moments` and `runs`. Model rows are never copied or changed. The runs row's `model` is the main model id, and its `config_hash` includes the thresholds.
  - Predict runs are never modified (runs are immutable).
- **Reading after a rebuild**: `read_official("moments")` uses `official_runs.yaml: moments` if it is non-empty, else `predict`. `read_official("predictions")` returns the predict run's model rows plus, when a moments run is official for that video, that run's recomputed `uncertain_mixed` rows in place of the predict run's.
