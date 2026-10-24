# Tasks: Visual Pipeline

**Input**: Design documents from `/specs/002-visual-pipeline/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ (interfaces.md, config.md), quickstart.md; 001 code on `main`

**Tests**: Requested. Each story phase includes its own tests. Run `uv run pytest` after every test task (slow tests are skipped by default).

**Organization**: Priority order: US1 (scoring), US2 (labels, uncertain, moments), US3 (idle), US4 (predict command), US5 (official runs), the rest of US2 (calibrate + moments rebuild, which needs US5), then US6 (sync). One developer works sequentially, so there are no `[P]` markers. Each task should take 15–30 minutes.

**Ground rules for every task**:
- Never use real filenames, video IDs, counts, or dataset details. Use only `mock*` and `t0*` IDs and names.
- Never read `.env`, `config/videos.local.yaml`, `config/thresholds.local.yaml` (when it holds real values), `DATA_DIR`, or `OUTPUT_DIR`. In tests and checkpoints, set `DATA_DIR` and `OUTPUT_DIR` to scratch paths through environment variables. They win over `.env`, so `.env` is never edited or read.
- Never download model weights. Every load uses `local_files_only=True`.
- Match the contracts. If code must differ from them, update the contract in the same task.

## Format: `[ID] [Story] Description`

## Phase 1: Setup

- [X] T001 Commit `.env.example` with `git add .env.example && git commit -m "chore: add .env.example"`. It must hold placeholders only; check its contents first. In `pyproject.toml`, under `[tool.pytest.ini_options]`, add `markers = ["slow: needs locally cached model weights"]` and `addopts = "-m 'not slow'"`. Run `uv run pytest` to confirm everything passes.
- [X] T002 Create `config/official_runs.yaml` from `contracts/config.md`, with empty `predict`, `moments`, and `segment` lists and its header comment.

---

## Phase 2: Foundational (config + 001 updates; blocks all stories)

- [X] T003 Edit `config/labels.yaml`:
  - Set `guideline_version: "0.2.0"`.
  - Delete the `uncertain` and `mixed` labels.
  - Add `uncertain_mixed` (group `status`, `source: rules`, color `#525252`, no prompts).
  - Change `needs_review` to `source: rules` and remove its prompts. Keep its color.
- [X] T004 In `src/tpa_index/config.py`, update `_check_labels`:
  - Accept an optional `source` with values in {`model`, `rules`}, defaulting to `model`.
  - Rules labels must have no `prompts`. Model labels still need 3–5.
  - Keep the other 001 rules.

  Add `rules_labels(labels)` and `model_labels(labels, groups)` helpers. In `tests/test_config.py`, add cases:
  - a rules label with prompts fails;
  - a model label without prompts fails;
  - the shipped file loads with guideline `0.2.0`.
- [X] T005 Edit `config/pipeline.yaml` to match `contracts/config.md`:
  - Add `visual_groups`, `uncertain_margin`, `label_thresholds: {}`, `models` and `idle`.
  - Remove `moments.needs_review_below` and `moments.conflict_margin`.

  In `src/tpa_index/config.py`, update `_check_pipeline` with every validation rule in `contracts/config.md`. Drop the checks for the removed keys. In `tests/test_config.py`, add one failing case each for:
  - a bad `visual_groups` entry;
  - a motion label that isn't an activity;
  - a threshold above 1;
  - a threshold for an unknown label or a rules label;
  - `batch_windows` of 0.
- [X] T006 In `src/tpa_index/config.py`:
  - Add a `THRESHOLDS_LOCAL = Path(os.environ.get("TPA_THRESHOLDS_FILE", CONFIG_DIR / "thresholds.local.yaml"))` constant. The env override lets checkpoints and script tests avoid the real file.
  - Make `load_pipeline()` deep-merge its `label_thresholds` over the pipeline's. The local file wins, per model and per label.
  - Add `threshold(pipeline, model, label)`, which falls back to `label_threshold`.

  In `tests/conftest.py`, monkeypatch `config.THRESHOLDS_LOCAL` to a missing tmp path. Add tests:
  - fallback;
  - per-model override;
  - local file wins;
  - a threshold for another model doesn't apply.
- [X] T007 Create `src/tpa_index/score.py` with pure rule functions:
  - `segment_labels(scores: dict[str, float], thresholds: dict[str, float], activity: list[str]) -> (primary | None, secondary: list[str])`. Primary is the highest-scoring activity that is ≥ its own threshold. Secondary is every other label ≥ its own threshold, sorted by score descending.
  - `uncertain_mixed(activity_scores, thresholds, margin) -> bool`, following research §7: no activity clears its threshold, or (top1 − next-highest activity) < margin.

  Write `tests/test_score.py`, covering a clear winner, nothing clearing, a close top two, and per-label thresholds changing the winner.
- [X] T008 Update `merge_moments(segments, predictions, pipeline, run_id, window_s=None, model=None)` in `src/tpa_index/moments.py`, following research §9 and §17:
  - `window_s` defaults to `pipeline["window_s"]`.
  - Use only rows where `model == model` (if given) or `model != "rules"`.
  - Primary candidates must clear `threshold(pipeline, model, label)`.
  - Skip segments with `is_idle is True`.
  - Set `needs_review = confidence < threshold(model, primary) or any segment in the moment has an uncertain_mixed rules row with score 1`.
  - Remove the `shortcut:` comment.

  Rewrite `tests/test_moments.py` for the new rules:
  - drop the `needs_review_below` and `conflict_margin` fixtures;
  - add `window_s=4` changing the gap and min length;
  - an idle segment breaks a moment;
  - an `uncertain_mixed` row sets `needs_review`;
  - a per-label threshold.
- [X] T009 Update `src/tpa_index/mock.py`:
  - `_predictions` skips `source: rules` labels.
  - After scoring, add one `uncertain_mixed` rules row per segment (model `"rules"`) using `score.uncertain_mixed` with mock thresholds.
  - `_videos_and_segments` uses `effective_settings(pipeline, v)` for windows.
  - `generate` calls `merge_moments` per video with that video's `window_s` and `model="mock-model"`.
  - Remove both `shortcut:` comments.

  In `_reviews`, pick review labels from model labels only. Update `tests/test_mock.py`:
  - rules rows exist for every segment;
  - review `guideline_version == "0.2.0"`;
  - the `clip`-kind mock video has 4 s windows (3 s stride);
  - reproducibility still holds.

**Checkpoint**: `uv run pytest` passes in full: all 001 tests under the new config, plus the new ones.

---

## Phase 3: User Story 1 — Score every visual label (P1) 🎯 MVP

**Goal**: For each segment, one 0–1 score per model-scored visual label, from an injectable embedder.

**Independent test**: With `FakeEmbedder` on synthetic clips, every segment gets exactly one row per model label in `visual_groups`. Every score is in [0, 1], the output is deterministic, and frames arrive whole and square.

- [X] T010 [US1] Create `src/tpa_index/embed.py` with the `Embedder` Protocol from `contracts/interfaces.md` (typing only, no torch import). Create `tests/fakes.py` with `FakeEmbedder(d=8, scale=10.0, bias=-5.0)`:
  - `embed_texts` maps each text to a unit vector seeded by `zlib.crc32(text)`.
  - `embed_images` uses each image's mean RGB plus fixed random projection columns, then L2-normalizes.
  - It records the image shapes it received in `self.seen_shapes`.
  - Add `steer(text, vec)` so a test can force a text's vector.
- [X] T011 [US1] In `src/tpa_index/score.py`, add:
  - `label_embeddings(embedder, labels)`: model labels only. Lowercase each prompt, embed all prompts in one call, average per label, then renormalize.
  - `window_embedding(frame_embeds)`: mean, then renormalize.
  - `sigmoid_scores(window_embeds, label_matrix, scale, bias)`: float32 `1/(1+exp(-(W·Lᵀ·scale + bias)))`.

  Add to `tests/test_score.py`:
  - averaging then renormalizing gives unit norm;
  - scores are in (0, 1);
  - a steered match scores higher than a mismatch;
  - prompt order doesn't change the label vector (to within 1e-6).
- [X] T012 [US1] In `src/tpa_index/segment.py`, add `with_audio: bool = True` to `segment_video`. When `False`, skip `_load_audio` and yield an empty float32 array for audio. Add to `tests/test_segment.py`: `with_audio=False` yields identical rows and frames and an empty audio array.
- [X] T013 [US1] In `src/tpa_index/score.py`, implement `predict_video(video, pipeline, labels, embedder, run_id, device="cpu", motion=None)`, with model scoring only for now:
  - Probe the video and build the videos row (same fields as the 001 segment CLI).
  - Iterate `segment_video(video, effective_settings(...), with_audio=False)` and buffer `pipeline["models"]["batch_windows"][device]` windows at a time.
  - For each batch, `embed_images` on the stacked frames (`B*N, S, S, 3`), reshape to `(B, N, d)`, then `window_embedding` per window.
  - Score against `label_embeddings` (computed once per call) and emit one prediction row per segment × model label in `visual_groups`, with `model=embedder.name` and `run_id`.
  - Return `{"videos", "segments", "predictions"}` (moments come in T016).
  - Write nothing to disk.
- [X] T014 [US1] Write `tests/test_predict.py`, using `FakeEmbedder` with the `clip_videos` fixture. Check that:
  - row count = segments × model labels in category, actor, and activity;
  - there are no audio, expression, indicator, or status model rows;
  - all scores are in [0, 1];
  - all rows have `model == "fake"` and the given `run_id`;
  - two calls give identical predictions;
  - `seen_shapes` are all square (S, S, 3), with S = max(w, h), for portrait `t02`;
  - with `batch_windows` of 1 and of 64 the results are the same;
  - rows pass `validate("predictions")` and `validate("segments")`.
- [X] T015 [US1] Implement `SiglipEmbedder(model_id, device, batch_images=256)` in `src/tpa_index/embed.py`, following research §1–§4:
  - Load with `AutoModel` and `AutoProcessor`, `local_files_only=True`, and `dtype=torch.bfloat16` on CUDA (float32 otherwise), moved to `device` and put in eval mode.
  - `embed_texts`: lowercase, `padding="max_length", max_length=64`, `get_text_features`, then L2-normalize in float32.
  - `embed_images`: processor on uint8 arrays in chunks of `batch_images` under `torch.inference_mode()`, `get_image_features`, then L2-normalize in float32 and return numpy.
  - `logit_scale = float(model.logit_scale.exp())` and `logit_bias = float(model.logit_bias)`. `name` is the model ID's last path part (for example `siglip2-base-patch16-224`).
  - If the model is missing from the local cache, raise `ConfigError(f"model not found locally: {model_id}")`.
- [X] T016 [US1] Write `tests/test_siglip_slow.py` (marked `@pytest.mark.slow`). Skip it when `huggingface_hub.try_to_load_from_cache("google/siglip2-base-patch16-224", "config.json")` is not a path. Run `predict_video` on `t01` with the real `SiglipEmbedder` on CPU and check:
  - the row count matches;
  - scores are in (0, 1);
  - a second run's scores are within 1e-3.

  Run it with `uv run pytest -m slow`.

**Checkpoint**: US1 is done. Scores come out for every visual label.

---

## Phase 4: User Story 2 — Primary, secondary, uncertain_mixed, and moments (P1)

**Goal**: `predict_video` adds `uncertain_mixed` rules rows and idle-aware moments built with the video's own window.

**Independent test**: Steered fake scores give the expected primary label and flag, and a clip-kind video's moments use 4 s windows.

- [X] T017 [US2] In `predict_video` (`src/tpa_index/score.py`), after scoring each segment, add one rules row: `label="uncertain_mixed"`, `label_group="status"`, `score=1.0 or 0.0` from `uncertain_mixed(...)` using `threshold(pipeline, embedder.name, label)` per activity label, `model="rules"`, same `run_id`. Then call `merge_moments(segments, predictions, pipeline, run_id, window_s=effective["window_s"], model=embedder.name)` and return `"moments"` too.
- [X] T018 [US2] Add to `tests/test_predict.py`, steering `FakeEmbedder` so one activity's prompts align with the clip's frame vectors. Check that:
  - that activity is primary and `uncertain_mixed` = 0;
  - with two activities steered equally, `uncertain_mixed` = 1 and the moment's `needs_review` is True;
  - with `label_thresholds` above every score, nothing is primary, every segment has `uncertain_mixed` = 1, and there are no moments;
  - `t02` (kind `clip`) gives moments whose bounds follow 4 s windows;
  - `validate_dataset`-style references hold (moments' `video_id` is in videos).

---

## Phase 5: User Story 3 — Skip idle time (P2)

**Goal**: A segment is idle only when the idle prompts match **and** the frames barely change.

**Independent test**: A static clip with the idle prompt steered high is idle. A moving clip with the same prompt score is not. A single-frame window is never idle.

- [X] T019 [US3] In `src/tpa_index/score.py`, add:
  - `visual_change(frames)`: `None` if there are fewer than 2 frames. Otherwise convert each frame to grayscale `(0.299R + 0.587G + 0.114B)`, downscale to 64×64 by block mean (`np.ndarray` reshape), and return the mean absolute difference between consecutive frames in 0–1.
  - `is_idle(idle_score, change, idle_cfg)`: `change is not None and idle_score >= idle_cfg["score_threshold"] and change < idle_cfg["change_threshold"]`.

  Add to `tests/test_score.py`: identical frames give 0, a black→white change gives 1, a single frame gives `None`, and the full AND truth table for `is_idle`.
- [X] T020 [US3] In `tests/clips.py`, add a `static=True` option to `make_clip` (one constant colour, no per-second change). In `tests/conftest.py`, add clip `t04`: static, 160×90, kind `live`. In `predict_video`:
  - embed `pipeline["idle"]["prompts"]` once with ensembling, as an "idle" vector;
  - compute each window's idle score with `sigmoid_scores`;
  - set `row["is_idle"] = is_idle(score, visual_change(frames), idle_cfg)`.

  No prediction rows are written for idle.
- [X] T021 [US3] Add to `tests/test_predict.py`, steering the idle prompts' vectors to match each clip's frame vectors. Check that:
  - `t04` (static) segments are all `is_idle=True` and give no moments;
  - `t01` (colour changes every second) has `is_idle=False` even with a high idle score;
  - with `frames_per_window: 1` via a `by_video` override, no segment is idle.

---

## Phase 6: User Story 4 — Predict command, one run per video (P2)

**Goal**: `tpa-index predict` works on CUDA, MPS, or CPU, with one tracked run per video. Failures don't lose other videos. There is an optional motion model and a Slurm wrapper.

**Independent test**: Three clips, one broken, give two complete runs, list the failed ID, and exit 1. The sbatch file has the required directives and `HF_HOME`.

- [X] T022 [US4] In `src/tpa_index/cli.py`, add `predict [--videos PATH] [--video-id ID …] [--kind KIND …] [--model ID]`:
  - Require `--videos` or `videos.local.yaml`, as `segment` does.
  - Filter by ID and kind. An unknown ID is a `ConfigError`.
  - `dev = runs.device()`. The model is `--model` or `models.main.hpc` if `dev == "cuda"`, else `models.main.mac`.
  - Build the embedder **once** through a module-level `_make_embedder(model_id, dev)`, which tests patch. A missing model exits 3 before any video.
  - For each video, inside `track_run("predict", model=embedder.name)`: `predict_video(...)`, then `write_table` for videos, segments, predictions, and moments, and set `ctx.n_segments`.
  - Report failed IDs and exit 1 if any failed, using the same redacted pattern as `segment`.
  - Set `os.environ.setdefault("HF_HUB_OFFLINE", "1")` before loading.
- [X] T023 [US4] Add to `tests/test_cli.py`, running in-process with `cli._make_embedder` patched to return `FakeEmbedder` and `install_redaction` patched to a no-op:
  - three clips plus one missing file give 3 complete predict runs, exit 1, and `failed videos: t09` with no filename shown;
  - every runs row has `stage == "predict"`, `model == "fake"`, a hardware name, and `n_segments > 0`;
  - `--kind clip` selects only `t02`;
  - a `_make_embedder` that raises `ConfigError` exits 3 with no runs written.
- [X] T024 [US4] Create `src/tpa_index/xclip.py` with `XClipScorer(model_id, device, labels, other_prompt)`:
  - Use `XCLIPModel` and `AutoProcessor`, with `local_files_only=True`.
  - `.name` is the last part of the model ID.
  - `.score(list_of_window_frames) -> np.ndarray (windows, len(labels))`: a softmax over `[label prompts (first prompt per label)…, other_prompt]`, keeping the label columns.

  In `predict_video`, when `motion` is given, run a second `segment_video` pass with `frames_per_window = models.motion.frames_per_window` and emit rows for the motion labels only, under `motion.name`. In `cli.py`, build it only when `models.motion.enabled`. Add to `tests/test_predict.py` a test with a fake motion scorer (`name="fake-motion"`, fixed scores): rows exist only for `dancing` and `acting` under `fake-motion`, and main-model rows are unchanged.
- [X] T025 [US4] Create `scripts/slurm/predict.sbatch`, following `contracts/interfaces.md`:
  - same header style as `job.sbatch`, with `--mem=64G` and `--job-name=tpa-predict`;
  - `export HF_HOME="$HOME/hf-cache" HF_HUB_OFFLINE=1`;
  - `uv sync --frozen`, then `uv run tpa-index predict "$@"`;
  - usage comments, including `--kind promo` for a test run.

  Create `tests/test_scripts.py`: `bash -n` passes, every required `#SBATCH` directive is present, and both the `HF_HOME` and `HF_HUB_OFFLINE` exports are present.
- [X] T026 [US4] **Checkpoint 1 (Mac, real model)**:
  1. `export DATA_DIR=<scratch>/data OUTPUT_DIR=<scratch>/out`, with the scratch folder outside the repo. Never touch `.env`.
  2. Run `uv run tpa-index mock --seed 0`.
  3. Make synthetic `t01` (landscape) and `t02` (portrait, kind `clip`) clips with `tests.clips.make_clip`, plus `<scratch>/videos.yaml`.
  4. Run `uv run tpa-index predict --videos <scratch>/videos.yaml --model google/siglip2-base-patch16-224`.
  5. Confirm two predict runs, rows = segments × model labels + rules rows, scores in (0, 1), and `is_idle` filled.
  6. Report runtime, peak memory, and hardware per run from `read_table("runs")`.
  7. Delete the scratch folder, and confirm `git status` is unchanged apart from the intended source edits.

---

## Phase 7: User Story 5 — Only official runs count (P2)

**Goal**: Result readers use only the listed runs, and a `runs` command helps fill in the list.

**Independent test**: With three runs and one official, the default read returns one run's rows. A bad or duplicate ID fails, naming the ID.

- [ ] T027 [US5] In `src/tpa_index/schema.py`, add:
  - `OFFICIAL_RUNS = repo_root()/"config/official_runs.yaml"` (patched in `tests/conftest.py`).
  - `load_official(path=None)`: returns `{stage: [ids]}` and ignores missing stages.
  - `read_official(table, stage=None, root=None)`, following `contracts/interfaces.md`:
    - With `stage` None, `moments` uses `"moments"` if listed, otherwise `"predict"`. Every other table uses `"predict"`.
    - Validate that each listed ID is a complete run folder (else `SchemaError("official_runs", stage, f"run {id} not found or incomplete")`). Validate that no two runs in a stage share a `video_id` from their `videos.parquet` (else an error naming both IDs).
    - For `predictions` with official `moments` runs, replace the predict runs' `rules` rows for those videos with the moments runs' rules rows.
- [ ] T028 [US5] Write `tests/test_official.py`, writing 3 mock-style runs with `write_table`. Check that:
  - the default read returns only the official run's rows;
  - an unknown ID raises, naming it;
  - an incomplete folder raises;
  - two official runs for one video raise, naming both;
  - an empty list returns an empty table;
  - moments precedence works, and so does rules-row replacement.
- [ ] T029 [US5] In `src/tpa_index/schema.py`, add `list_runs(root=None)`, following research §11:
  - The prediction count comes from `pq.ParquetFile(path).metadata.num_rows`.
  - `video_ids` come from that folder's `videos.parquet`, comma-joined.

  In `cli.py`, add `tpa-index runs`, which prints a fixed-width table of `run_id`, `stage`, `video_ids`, `hardware`, `n_segments`, `n_predictions`, `runtime_s` and `peak_mem_gb`. Add tests in `tests/test_official.py` (one line per complete run, incomplete runs skipped) and `tests/test_cli.py` (exit 0, no `OUTPUT_DIR` path in the output).

---

## Phase 8: User Story 2 (continued) — Calibrate thresholds and rebuild moments

**Goal**: Per-label thresholds (with an absolute floor) and the idle score threshold come from official **dev** data only. Idle flags and moments are then rebuilt without re-running the model.

**Independent test**: Eval rows never influence a threshold, `--split eval` is rejected, labels absent from the dev videos stop at the floor, the idle threshold comes from its own percentile, and a rebuild with new thresholds changes idle flags and moments while leaving predict runs untouched.

- [ ] T030 [US2] Store the idle inputs, so idle can be calibrated and rebuilt without the model. Add two nullable `float32` columns to `segments` in `src/tpa_index/schema.py`, after `is_idle`: `idle_score` (the ensembled idle-prompt sigmoid score) and `visual_change` (the `visual_change(frames)` value, null when there are fewer than 2 frames).
  - In `predict_video` (`src/tpa_index/score.py`), fill both on every segment row next to `is_idle`.
  - The 001 segmenter and the mock generator write them as `None`. Older run folders read back as null through `_conform`.
  - Update `specs/001-foundation/contracts/tables.md` (segments table) and `specs/001-foundation/data-model.md`.

  Add tests:
  - `tests/test_predict.py`: `idle_score` is in [0, 1] and `visual_change` ≥ 0 on every segment. `t04` (static) has `visual_change == 0`. With `frames_per_window: 1`, `visual_change` is null.
  - `tests/test_schema.py`: an older segments file without the two columns reads back with nulls.
- [ ] T031 [US2] Create `src/tpa_index/calibrate.py` with:
  - Config first: add `calibrate: {top_pct: 10, min_threshold: 0.05, min_segments: 20}` and `idle.calibrate_top_pct: 20` to `config/pipeline.yaml`, following `contracts/config.md`. Validate them in `_check_pipeline`: both percentiles in (0, 100), `min_threshold` in [0, 1], `min_segments` ≥ 1. Add one failing case each to `tests/test_config.py`.
  - `propose_thresholds(predictions, videos, model, top_pct, min_segments, min_threshold)`:
    - keep only rows whose segment belongs to a video with `split == "dev"` (join via `segment_id` prefix → `video_id`) and `model == model`;
    - per model label, return `(value, floored)`, where `value = max(np.percentile(scores, 100 - top_pct), min_threshold)` and `floored` is True when the floor was used. Return `(None, False)` if there are fewer than `min_segments` scores.
    - The floor stops a label that never appears in the dev videos from getting `top_pct`% of segments above threshold, which would produce fake moments.
  - `propose_idle_threshold(segments, videos, top_pct, change_threshold, min_segments)`:
    - use dev-split segments only, with `visual_change` not null and `< change_threshold`;
    - return the `(100 - top_pct)` percentile of their `idle_score`, or `None` if there are fewer than `min_segments`.
    - `change_threshold` stays hand-set and is never proposed.
  - `write_local_thresholds(model, thresholds, idle_score_threshold=None, path=None)`:
    - load the existing `config.THRESHOLDS_LOCAL` if present;
    - replace only `label_thresholds[model]` (skipping `None` values) and, when given, `idle_thresholds[model]`;
    - keep other models;
    - write with `yaml.safe_dump(sort_keys=True)` and a header comment `# written by tpa-index calibrate; gitignored, data-derived`.
  - In `src/tpa_index/config.py`:
    - `load_pipeline()` also merges `idle_thresholds` from the local file;
    - add `idle_threshold(pipeline, model)`, which returns `idle_thresholds[model]` and falls back to `idle.score_threshold`;
    - `predict_video` and the moments rebuild use it.

  Add `tests/test_calibrate.py` covering:
  - the percentile on known scores;
  - changing eval-split scores leaves the thresholds unchanged;
  - a label scoring near 0 on every dev segment returns `(min_threshold, True)`;
  - too few segments gives `None`;
  - the idle proposal ignores high-change segments and eval segments;
  - writing keeps another model's label and idle blocks;
  - `idle_threshold()` falls back and overrides correctly.
- [ ] T032 [US2] In `cli.py`, add `tpa-index calibrate --stage predict --split dev [--top-pct N] [--idle-top-pct N] [--dry-run]`:
  - The defaults come from `pipeline.yaml`: `calibrate.top_pct` for labels and `idle.calibrate_top_pct` for idle. They are separate settings, because the share of idle footage differs from any label's share.
  - `--split` other than `dev` raises `ConfigError("calibration uses the dev split only")`, exiting 3.
  - Read `read_official("predictions", stage)`, `read_official("segments", stage)` and `read_official("videos", stage)`. If any is empty, raise `ConfigError("no official dev runs for stage …")`.
  - The model is the most common non-`rules` model in the official predictions.
  - Print a `label | threshold | n_dev | clearing | floored` table, a line `labels at the floor (min_threshold): a, b, …` (or `none`), and a line `idle score_threshold: <value> (n_dev low-change segments)`. All of these are aggregate numbers only.
  - Write via `write_local_thresholds` unless `--dry-run`.

  Add tests to `tests/test_cli.py`:
  - `--split eval` exits 3;
  - `--dry-run` writes nothing;
  - a real run writes the patched `THRESHOLDS_LOCAL`, with both `label_thresholds` and `idle_thresholds`;
  - a label that never scores above 0.01 on mock dev data appears in the "labels at the floor" line;
  - `--idle-top-pct` changes only the idle value.
- [ ] T033 [US2] In `cli.py`, add `tpa-index moments [--video-id ID …]`. For each official predict run (filtered by video), inside `track_run("moments", model=<main model of that run>)`:
  - read that run's `videos`, `segments` and `predictions` via `read_table(..., run_ids=[id])`;
  - recompute `is_idle` from the stored `idle_score` and `visual_change`, using `idle_threshold(pipeline, model)` and the hand-set `idle.change_threshold`. Segments whose `visual_change` is null are never idle;
  - recompute the `uncertain_mixed` rows with the current `load_pipeline()` thresholds;
  - run `merge_moments(..., window_s=effective, model=main)`;
  - write `videos` (copied), `segments` (with the recomputed `is_idle`), `predictions` (the rules rows only), and `moments`.

  In `read_official`, `segments` now prefers an official `moments` run for that video, the same way `moments` does. Failures are listed and exit 1. Add tests to `tests/test_cli.py`:
  - after raising one label's threshold in the patched local file, the rebuilt moments differ from predict's;
  - after setting a model's idle threshold to 0 in the patched local file, the low-change segments become idle and their moments disappear;
  - the predict run's files are byte-identical before and after;
  - with the moments run listed under `moments:`, `read_official("moments")` and `read_official("segments")` return the rebuilt rows.

---

## Phase 9: User Story 6 — Move code to HPC and results back (P3)

**Goal**: One command each way. Media, secrets, local metadata, and the review log never travel.

**Independent test**: With a fake `rsync` on `PATH`, the scripts pass the right excludes, stop when keys are missing, and merge thresholds without losing the Mac's models.

- [ ] T034 [US6] Create `scripts/sync.exclude` (one pattern per line):
  - `.venv/`, `.env`, `.env.*`, `!.env.example`, `models/`, `__pycache__/`, `outputs/`;
  - `*.mp4`, `*.mov`, `*.m4v`, `*.mkv`, `*.avi`, `*.wav`, `*.mp3`, `*.m4a`, `*.parquet`, `*.duckdb`;
  - `config/*.local.yaml` (covers `videos.local.yaml` and `thresholds.local.yaml`) and `slurm-*.out`.

  Do not exclude `config/official_runs.yaml`. Create `scripts/sync_code_to_hpc.sh`:
  - `set -euo pipefail`;
  - parse `${TPA_ENV_FILE:-<repo>/.env}` line by line for `HPC_USER` and `HPC_HOST` with `grep`/`cut` (never `source`), exiting 2 with `missing HPC_USER in .env` (or the matching key) when absent;
  - `--dry-run` adds `-n -v`;
  - `rsync -az --exclude-from=scripts/sync.exclude "$REPO/" "$HPC_USER@$HPC_HOST:~/tpa-work/repo/"`, with no `--delete`;
  - run from any directory (`REPO=$(cd "$(dirname "$0")/.." && pwd)`).
- [ ] T035 [US6] Create `scripts/sync_outputs_from_hpc.sh`:
  1. Parse `HPC_USER`, `HPC_HOST`, and `HPC_OUTPUT_DIR` the same way.
  2. Resolve `LOCAL=$(uv run --project "$REPO" python -c "from tpa_index.paths import output_dir; print(output_dir())")`. Guard failure exits 2.
  3. Run `rsync -az --ignore-existing --exclude-from="$REPO/scripts/sync.exclude" --include='*.parquet' --exclude='reviews.duckdb' "$HPC_USER@$HPC_HOST:$HPC_OUTPUT_DIR/runs/" "$LOCAL/runs/"`. Parquet must be included here, because these are output folders.
  4. Fetch `~/tpa-work/repo/config/thresholds.local.yaml` with rsync into a `mktemp` file. If it exists, merge it per model into `$REPO/config/thresholds.local.yaml` through `uv run python -c`. This covers both `label_thresholds` and `idle_thresholds`: HPC models replace their own blocks, and other models are kept.
  5. `--dry-run` passes `-n -v` to the runs rsync and prints the models that would be merged, without writing.
- [ ] T036 [US6] Add to `tests/test_scripts.py`, putting a fake `rsync` (a bash script in `tmp_path` that appends its args to a log and, for the thresholds fetch, copies a fixture YAML to its last argument) first on `PATH` and setting `TPA_ENV_FILE` to a tmp env file. Check that:
  - the code sync passes `--exclude-from`, there is no `--delete`, and `sync.exclude` lists `.env`, `*.mp4`, `config/*.local.yaml`, and `models/` but not `official_runs.yaml`;
  - a missing `HPC_HOST` exits 2 and the log is empty;
  - the output sync passes `--ignore-existing` and `--exclude=reviews.duckdb`, and a guard failure (`OUTPUT_DIR` inside the repo) exits 2 before any rsync call;
  - the thresholds merge keeps a pre-existing Mac-model block and replaces the HPC model's block, using a patched local thresholds path via `TPA_THRESHOLDS_FILE`;
  - `--dry-run` writes nothing.

  Add `TPA_THRESHOLDS_FILE` support to the script, for tests only.

---

## Phase 10: Polish & Final Checkpoint

- [ ] T037 Update `README.md` with `predict`, `runs`, `calibrate`, `moments`, the sync scripts, and `predict.sbatch`. Note that HPC keeps its own `.env` and `config/videos.local.yaml`, which are never synced. Cross-check against the contracts:
  - the signatures in `contracts/interfaces.md`;
  - the config keys in `contracts/config.md`;
  - the run-folder contents in `data-model.md`.

  Fix any drift. Grep `src/` for leftover `shortcut:` comments about windows (there should be none) and for `needs_review_below` or `conflict_margin` (there should be none).
- [ ] T038 **Final checkpoint (Mac)**:
  1. Run `uv run pytest`, then `uv run pytest -m slow`. Record the counts.
  2. Quickstart in scratch, with `DATA_DIR` and `OUTPUT_DIR` exported to scratch paths and `.env` untouched:
     - `tpa-index check`;
     - `tpa-index mock`;
     - `predict` on synthetic `t01` and `t02` with siglip2-base;
     - `tpa-index runs`.
  3. Back up `config/official_runs.yaml` and list the two predict runs plus `mock-seed0` under `predict`. Mock and real must not share video IDs; if `read_official` reports a duplicate, list only the predict runs.
  4. Exercise `calibrate`: first `--dry-run`, then a real write, with `config.THRESHOLDS_LOCAL` pointed at scratch through `TPA_THRESHOLDS_FILE`, so the repo file is untouched. Then run `tpa-index moments`.
  5. Restore `config/official_runs.yaml` with `git checkout`, and confirm `git status` shows only the intended source changes.
  6. Delete the scratch folder.
  7. **Print** (do not run) the exact HPC commands for a promo-only **test** run:
     - `scripts/sync_code_to_hpc.sh --dry-run`, then `scripts/sync_code_to_hpc.sh`;
     - `ssh $HPC_USER@$HPC_HOST`, then `cd ~/tpa-work/repo`;
     - `sbatch scripts/slurm/predict.sbatch --kind promo`;
     - `squeue -u $USER` and `tail -f slurm-tpa-predict-<jobid>.out`;
     - back on the Mac, `scripts/sync_outputs_from_hpc.sh --dry-run`, then the real sync, then `uv run tpa-index runs`.

     Note: a test run is not listed in `official_runs.yaml`.

---

## Dependencies & Execution Order

- **Setup (T001–T002)**, then **Foundational (T003–T009)**, which blocks everything. It changes config that 001 code and tests depend on.
- **US1 (T010–T016)** needs Foundational.
- **US2, part 1 (T017–T018)** needs US1 (`predict_video`) and T007–T008.
- **US3 (T019–T021)** needs US1.
- **US4 (T022–T026)** needs US1–US3. Checkpoint 1 (T026) needs a working `predict`.
- **US5 (T027–T029)** needs only Foundational, but comes after US4 by priority.
- **US2, part 2 (T030–T033)** needs US5 (`read_official`). T030 changes the segments table, and T031–T033 use its new columns.
- **US6 (T034–T036)** needs only `paths.output_dir` (001) and the thresholds file format (T006).
- **Polish (T037–T038)** needs everything else.

## Parallel Opportunities

None are marked. One developer works through the tasks in ID order.

## Implementation Strategy

1. **MVP**: T001–T018 (config updates, scoring, and moments from `predict_video`). Moments can be demonstrated on synthetic clips with the fake embedder.
2. **Real-model proof**: T019–T026, ending at Checkpoint 1 with siglip2-base on the Mac.
3. **Trustworthy results**: T027–T033 (official runs, idle inputs, calibration, moments rebuild).
4. **HPC workflow**: T034–T036 (sync scripts).
5. **Wrap-up**: T037–T038, including the printed promo test-run commands.
