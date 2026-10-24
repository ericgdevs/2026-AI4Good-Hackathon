# Contract: Python API, CLI, Scripts (002)

This contract extends `specs/001-foundation/contracts/interfaces.md`, and everything there still holds.

## Python (`tpa_index`)

```python
# embed.py
class Embedder(Protocol):
    name: str                 # model id as recorded in predictions.model / runs.model
    logit_scale: float        # already exp()'d
    logit_bias: float
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...            # (n, d) float32, L2-normalized
    def embed_images(self, images: np.ndarray) -> np.ndarray: ...         # (n, S, S, 3) uint8 -> (n, d) float32, L2-normalized

SiglipEmbedder(model_id, device, batch_images)   # transformers, local_files_only=True, bf16 on cuda

# score.py
label_embeddings(embedder, labels: list[dict]) -> dict[str, np.ndarray]   # prompt-ensembled, renormalized
window_embedding(frame_embeds: np.ndarray) -> np.ndarray                   # mean of N, renormalized
sigmoid_scores(window_embeds, label_embeds, scale, bias) -> np.ndarray     # (windows, labels) float32 in 0-1
visual_change(frames: np.ndarray) -> float | None                          # None if < 2 frames
is_idle(idle_score, change, idle_cfg) -> bool
uncertain_mixed(activity_scores: dict[str, float], thresholds: dict[str, float], margin) -> bool
predict_video(video, pipeline, labels, embedder, run_id, device="cpu", motion=None) -> dict[str, list[dict]]
rules_rows(segments, predictions, pipeline, labels, model, run_id) -> list[dict]   # uncertain_mixed rows; shared by predict, mock, moments
#   returns {"videos": [...], "segments": [...], "predictions": [...], "moments": [...]}; writes nothing

# xclip.py  (only imported when models.motion.enabled)
XClipScorer(model_id, device, labels, other_prompt, frames_per_window=8)   # .name, .labels, .score(windows_frames) -> (windows, labels) softmax

# segment.py (changed)
segment_video(video, settings, sample_rate=16000, with_audio=True) -> Iterator[Window]

# moments.py (changed)
merge_moments(segments, predictions, pipeline, run_id, window_s=None, model=None) -> list[dict]
#   window_s defaults to pipeline["window_s"];
#   needs_review = conf < threshold(model, primary) * (1 + moments.needs_review_margin) or any segment uncertain_mixed

# config.py (new)
threshold(pipeline, model, label) -> float     # per-label (per model) with global fallback
# load_pipeline() merges config/thresholds.local.yaml (gitignored) over pipeline.yaml

# calibrate.py (new)
propose_thresholds(predictions, videos, model, top_pct, min_segments, min_threshold) -> dict[str, tuple[float | None, bool]]
#   dev only; value = max(percentile, min_threshold); bool = floor used
propose_idle_threshold(segments, videos, top_pct, change_threshold, min_segments) -> float | None   # dev, low-change only
dev_scores(predictions, videos, model) -> dict[str, np.ndarray]      # dev-split scores per label (used by the CLI table)
idle_candidates(segments, videos, change_threshold) -> np.ndarray    # dev low-change idle scores
write_local_thresholds(model, thresholds, idle_score_threshold=None, path=None) -> None

# config.py (new, with T031)
idle_threshold(pipeline, model) -> float   # idle_thresholds[model], else idle.score_threshold

# schema.py (new)
load_official(path=None) -> dict[str, list[str]]
read_official(table, stage=None, root=None) -> pa.Table   # result-facing code MUST use this, not read_table
#   stage None: moments and segments -> the official "moments" run for a video if listed, else "predict"; others -> "predict"
#   (rules rows from official moments runs replace predict's)
list_runs(root=None) -> list[dict]   # run_id, stage, video_ids, hardware, n_segments, n_predictions, runtime_s, peak_mem_gb
# OFFICIAL_RUNS path: TPA_OFFICIAL_RUNS overrides it (tests and checkpoints only)
```

## CLI

| command | does | exit codes |
|---|---|---|
| `tpa-index predict [--videos PATH] [--video-id ID …] [--kind KIND …] [--model ID]` | one predict run per video: segments with `is_idle`, predictions (model + rules [+ motion]), moments, runs | 0; 1 if any video failed (IDs listed); 2 guard; 3 config/model not found |
| `tpa-index runs` | lists complete runs, one line each; no filenames or paths | 0 / 2 |
| `tpa-index calibrate --stage predict --split dev [--top-pct N] [--idle-top-pct N] [--dry-run]` | official dev-split data only. Prints label, threshold, n_dev, clearing and floored; the labels at the floor; and the idle threshold. Writes `config/thresholds.local.yaml` (label and idle thresholds per model). Defaults come from `calibrate.top_pct` and `idle.calibrate_top_pct`. | 0; 3 if `--split eval` or no official dev runs |
| `tpa-index moments [--video-id ID …]` | rebuilds `is_idle` (from the stored `idle_score` and `visual_change`), `uncertain_mixed` rows and moments from official predict runs with the current thresholds; one `moments` run per video (videos, segments, rules predictions, moments); no model loaded | 0; 1 if any video failed |

`--model` defaults to `models.main.hpc` on CUDA and `models.main.mac` otherwise. The model is loaded once, before the first video. A missing model exits 3 before any video is processed.

## Scripts

| script | does |
|---|---|
| `scripts/slurm/predict.sbatch [predict args]` | 1× `gpu:6KSE`, 8 CPUs, 64 GB, 2 h; sets `HF_HOME=$HOME/hf-cache` and `HF_HUB_OFFLINE=1`; runs `uv sync --frozen`, then `uv run tpa-index predict "$@"` |
| `scripts/sync_code_to_hpc.sh [--dry-run]` | rsyncs the repo to `$HPC_USER@$HPC_HOST:~/tpa-work/repo/` with `scripts/sync.exclude` (excludes `config/*.local.yaml`; `config/official_runs.yaml` is synced); no `--delete` |
| `scripts/sync_outputs_from_hpc.sh [--dry-run]` | rsyncs `$HPC_OUTPUT_DIR/runs/` to the local `OUTPUT_DIR/runs/` (guarded) with `--ignore-existing`; never copies `reviews.duckdb` or media. Then fetches HPC `~/tpa-work/repo/config/thresholds.local.yaml` and **merges it per model** into the local file (HPC entries win for their models; the Mac's other models are kept) |

Both sync scripts read `HPC_USER`, `HPC_HOST` (and `HPC_OUTPUT_DIR`) from `.env` by parsing it, never by `source`-ing it. A missing key exits 2 and transfers nothing. `TPA_ENV_FILE` overrides the env-file path (for tests, so the real `.env` is never read).

Test-only overrides (never set in normal use): `TPA_SYNC_DEST` (code sync to a local folder), `TPA_SYNC_REMOTE=""` (output sync from a local folder), `TPA_THRESHOLDS_FILE` (merge target). Optional `.env` key `HPC_REPO_DIR` (default `~/tpa-work/repo`, expanded on HPC).

Output sync uses an allowlist (`--include='*/' --include='*.parquet' --exclude='*'`) instead of `sync.exclude`: only run-folder Parquet files can travel down.

`--kind` selects mapped videos by `kind` (for example `--kind promo` for a test run) without naming video IDs. A test run is simply not listed in `official_runs.yaml`.
