# Tasks: Shared Data Foundation

**Input**: Design documents from `/specs/001-foundation/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ (tables.md, config.md, interfaces.md), quickstart.md

**Tests**: Requested. Each story phase includes its own tests. Run `uv run pytest` after every test task.

**Organization**: Phases follow the requested order: US1 (contract + mock), then US2 (data guard), US3 (segmenter), US4 (run tracking), US5 (environment/Slurm), US6 (config validation). One developer works sequentially, so there are no `[P]` markers. Each task should take 15–30 minutes.

**Ground rules for every task**:
- Never use real filenames or dataset details. Use only `mock*` and `t0*` IDs and names.
- Never read `config/videos.local.yaml`, `.env`, or anything under `DATA_DIR` or `OUTPUT_DIR` outside of tests' `tmp_path`.
- Match the contracts exactly. If code must differ from them, update the contract in the same task.

## Format: `[ID] [Story] Description`

## Phase 1: Setup

- [X] T001 Rename the package:
  - Move the folder with `git mv src/2026_ai4good_hackathon src/tpa_index` (if it is untracked, use a plain `mv`).
  - In `pyproject.toml`, set `name = "tpa-index"` and description `"Searchable, staff-verified timeline of program videos"`, and replace `[project.scripts]` with `tpa-index = "tpa_index.cli:main"`.
  - Replace `src/tpa_index/__init__.py` with a one-line module docstring.
  - Create `src/tpa_index/cli.py` with an argparse `main()` that has subparsers `check`, `mock`, and `segment`, each printing "not implemented" for now.
- [X] T002 In `pyproject.toml`, add the pytest dev dependency with `uv add --dev pytest`. Add `[tool.pytest.ini_options]` with `testpaths = ["tests"]`. Run `uv sync` and verify that `uv run python -c "import tpa_index"` and `uv run tpa-index --help` both work.
- [X] T003 Append `slurm-*.out` to `.gitignore`. Create the empty directories `config/`, `scripts/slurm/`, and `tests/`.

---

## Phase 2: Foundational (blocks all stories)

- [X] T004 Create `config/labels.yaml` following `contracts/config.md`:
  - Set `guideline_version: "0.1.0"`.
  - Include the full starter vocabulary table, plus the expression group `confident, upset, positive, neutral, negative`.
  - Give every label 3 observable prompt phrasings (no mental-state wording).
  - Give every label a distinct `#RRGGBB` color that is readable on white.
- [X] T005 Create `config/pipeline.yaml` exactly as in `contracts/config.md`. Create `config/videos.example.yaml` with 4 mock videos, `mock01` to `mock04`:
  - filenames `mock/mock_<kind>_0N.mp4`;
  - orientations: 3 landscape and 1 portrait;
  - kinds: live, edited, clip, promo;
  - phases: early, mid, showcase, unknown;
  - splits: `mock04` is `eval`, the rest are `dev`.
- [X] T006 Create `src/tpa_index/paths.py` with:
  - `repo_root()`, returning `Path(__file__).resolve().parents[2]`;
  - `load_dotenv(repo_root()/".env", override=False)` at import;
  - `class GuardError(Exception)`;
  - `data_dir()` and `output_dir()`, which read `DATA_DIR` and `OUTPUT_DIR`, return `Path(v).expanduser().resolve()`, and raise `GuardError("DATA_DIR is not set")` (or the `OUTPUT_DIR` equivalent) when unset.

  The inside-repo check comes in US2. Do not import torch here.
- [X] T007 Create `src/tpa_index/config.py` with:
  - `class ConfigError(Exception)`;
  - `load_labels(path=None)`, `load_pipeline(path=None)`, and `load_videos(path=None)`, each using `yaml.safe_load`. Defaults are under `repo_root()/"config"`. `load_videos` returns `videos.local.yaml` when it exists, else `videos.example.yaml`, as a list of dicts;
  - `config_hash(labels, pipeline)`, per research §6.

  Full validation comes in US6.
- [X] T008 Add `windows(duration_s, window_s, overlap_s) -> list[tuple[float, float]]` to `src/tpa_index/segment.py`, following research §4 (stride, trailing window under half a window folded into the previous one, short video gives one segment). Write `tests/test_windows.py` covering:
  - 60 s at 8/2: starts `0,6,…,54`, last end `60`;
  - 51 s: the trailing 3 s window is folded, so the last segment is `(42, 51)`;
  - 5 s: `[(0, 5)]`;
  - 8/2 with `window_s=4, overlap_s=1`: stride 3.
- [X] T009 Create `tests/conftest.py` with:
  - an autouse fixture that uses `monkeypatch.setenv` to point `DATA_DIR` at `tmp_path/"data"` and `OUTPUT_DIR` at `tmp_path/"out"` (creating both);
  - fixtures `labels`, `pipeline`, and `example_videos` that load the shipped configs (`load_videos(repo_root()/"config/videos.example.yaml")`).

**Checkpoint**: `uv run pytest` passes (window tests).

---

## Phase 3: User Story 1 — Build against a fixed contract with mock data (P1) 🎯 MVP

**Goal**: The six tables can be validated, written, read, and generated as mock data.

**Independent test**: `tpa-index mock` writes a dataset in which every table passes validation, all cross-table references resolve, there are no identity columns, and the output is reproducible by seed.

- [X] T010 [US1] In `src/tpa_index/schema.py`, define `SCHEMAS` for all six tables, with exact column names, types, and nullability from `contracts/tables.md`. Use `pa.list_(pa.string())` for list columns and `pa.timestamp("us", tz="UTC")` for timestamps. Next to it, add a `RULES` dict per table with:
  - `pk` (tuple of columns);
  - `enums` (column → allowed set; `phase` is filled from the `phases` argument);
  - `unit` (columns in 0–1);
  - `positive` (> 0);
  - `nonneg` (≥ 0).

  Use the data-model.md lists.
- [X] T011 [US1] In `src/tpa_index/schema.py`, implement `class SchemaError(Exception)`, which takes `(table, column, reason)`, and `validate(table, data, phases=None) -> pa.Table`. `data` is a list of dicts or a `pa.Table`. Apply data-model.md rules 1–7 in order:
  1. exact column set;
  2. `pa.Table.from_pylist(rows, schema)` or `.cast(schema)`, wrapping `pa.ArrowInvalid` or `TypeError` in `SchemaError`;
  3. no nulls in non-nullable fields;
  4. enum values;
  5. unit, positive, and nonneg ranges;
  6. `end_s > start_s` and `start_s >= 0`;
  7. unique pk, plus the reviews `new_label` rule.

  Use pyarrow.compute only. If `phases` is None, load them from `load_pipeline()["video_phases"]`.
- [X] T012 [US1] Write `tests/test_schema.py`. Start with one valid row per table, kept in a helper dict `VALID`, and check it passes. Then add one failing case per rule, each asserting that `SchemaError` names the right table and column:
  - missing column;
  - extra `person_id`, `track_id`, or `embedding`;
  - a string in `score`;
  - null `video_id`;
  - `split="train"`;
  - `phase="finale"`;
  - `score=1.2`;
  - `end_s <= start_s`;
  - duplicate `segment_id`;
  - review `action="edit"` with null `new_label`;
  - review `action="accept"` with a non-null `new_label`.
- [X] T013 [US1] In `src/tpa_index/schema.py`, implement:
  - `write_table(table, data, run_id, root=None) -> Path`: validate first, then write to `(root or output_dir())/"runs"/run_id/f"{table}.parquet"`. Raise `FileExistsError` if the target already exists. Write to a `.tmp` file, then `os.replace`.
  - `read_table(table, root=None) -> pa.Table`: use DuckDB `read_parquet([...files], union_by_name=true)` over the `runs/*/{table}.parquet` files whose run folder also contains `runs.parquet`, and return `.arrow()` cast to `SCHEMAS[table]`. If there are no files, return `SCHEMAS[table].empty_table()`.
- [X] T014 [US1] In `src/tpa_index/schema.py`, implement `append_reviews(rows, root=None) -> int` and `read_reviews(root=None)`, using `(root or output_dir())/"reviews.duckdb"`:
  - Fill a missing `review_id` with `uuid4().hex` and a missing `created_at` with UTC now, then validate.
  - Create the table with `CREATE TABLE IF NOT EXISTS reviews AS SELECT * FROM empty_arrow`, then `INSERT INTO reviews SELECT * FROM arrow_tbl`.
  - Guard writes with a module-level `threading.Lock`, and open and close the connection on each call.
  - Provide no update or delete function.
- [X] T015 [US1] Add these to `tests/test_schema.py`:
  - writing the same table and run twice raises `FileExistsError`;
  - an invalid write leaves no file;
  - `read_table` concatenates two complete runs and skips a run folder without `runs.parquet`;
  - `append_reviews` twice gives `read_reviews().num_rows` equal to the sum;
  - `schema` has no attribute matching `update|delete`.
- [X] T016 [US1] Implement `merge_moments(segments, predictions, pipeline, run_id) -> list[dict]` in `src/tpa_index/moments.py`, following research §12:
  - Group by video, ordered by `start_s`.
  - Primary candidate: the top `primary_group` label with score ≥ `label_threshold`.
  - Merge consecutive segments with the same label when the gap is ≤ `max_gap_windows * window_s`.
  - Drop moments shorter than `min_windows * window_s`.
  - `secondary_labels`, `indicators`, `confidence`, and `needs_review` follow research §12, using `conflict_margin` against the runner-up primary label.
  - `moment_id = f"{video_id}_{round(start_s*1000)}_{label}"`.
  - Return dicts that match the moments schema.
- [X] T017 [US1] Write `tests/test_moments.py` with hand-built segments and predictions (8 s windows) checking that:
  - 3 adjacent "singing" segments become 1 moment;
  - a 1-window gap merges and a 2-window gap splits;
  - a lone 5 s trailing segment is dropped;
  - mean score 0.55 gives `needs_review=True`;
  - a runner-up within 0.1 gives `needs_review=True`;
  - indicator and secondary labels above threshold are listed;
  - output passes `validate("moments", …)`.
- [X] T018 [US1] In `src/tpa_index/mock.py`, implement `_videos_and_segments(rng, videos_cfg, pipeline)`:
  - For each example video, set `duration_s = rng.uniform(300, 1500)` and `fps = 30.0`.
  - Build segments with `windows()`, using `segment_id = f"{video_id}_{round(start*1000)}"`.
  - Set signals per segment: `person_count` 0–15, `faces_detected` ≤ `person_count`, `has_speech`, `transcript_chars` (0 when no speech), and `is_idle`. Leave about 5% of the signals null.
  - With 4 videos, the total should land at 200–1000 segments.
- [X] T019 [US1] In `src/tpa_index/mock.py`, implement `_predictions(rng, segments, labels, run_id)`:
  - Per video, walk segments and keep a latent activity that persists for `rng.integers(2, 8)` segments, so moments form.
  - Score the true label with `beta(5, 2)` and the others with `beta(1.5, 6)`. Score the other groups the same way, with their own latent label.
  - Use model `"mock-model"`. Cast scores to float32 in the range 0–1.
- [X] T020 [US1] In `src/tpa_index/mock.py`, implement `_reviews(rng, moments, segments, labels)`:
  - Review about 20% of moments and about 5% of segments.
  - Include every `action`, `source`, and `mode` value at least once, with both agreeing (accept) and disagreeing (edit or reject with `new_label`) cases.
  - Use reviewers `mock-staff-a` and `mock-staff-b` and `guideline_version` from labels.
  - Base `created_at` on `2026-01-01T00:00Z + i minutes`, and derive `review_id` from the rng (`rng.bytes(16).hex()`) so output is reproducible.
- [X] T021 [US1] In `src/tpa_index/mock.py`, implement `generate(out_root=None, seed=0, n_videos=4) -> str`:
  - Set `run_id = f"mock-seed{seed}"` and use `np.random.default_rng(seed)`. Take the first `n_videos` example videos.
  - Build videos, segments, predictions, and moments (via `merge_moments`).
  - Build one runs row per mock stage: `segment` and `predict`, both with hardware `"mock-hardware"`, a fixed `started_at`, `runtime_s > 0`, and `config_hash` from `config_hash()`.
  - Call `write_table` for each table, writing `runs` last, then `append_reviews`. Return the `run_id`.
- [X] T022 [US1] Add `validate_dataset(root=None)` to `src/tpa_index/schema.py`. It reads all tables and raises `SchemaError` when:
  - `segments.video_id` or `moments.video_id` is not in videos;
  - `predictions.segment_id` is not in segments;
  - `predictions.run_id` or `moments.run_id` is not in runs.

  Wire `tpa-index mock --seed N --n-videos N` in `src/tpa_index/cli.py` to call `generate` and then `validate_dataset`, and print the row count per table.
- [X] T023 [US1] Write `tests/test_mock.py` checking that:
  - `generate()` finishes in under 60 s;
  - all six tables are non-empty and `validate_dataset` passes;
  - there are at least 3 videos and 200–1000 segments;
  - all five actions and both sources and modes are present;
  - some moments have `needs_review=True` and some `False`;
  - no column in any schema matches `person|track|face_id|embedding|crop`;
  - two generates with seed 0 into different roots give equal tables (`pa.Table.equals`), and seed 1 differs.

**Checkpoint**: US1 is complete. Later feature work can start against the mock data.

---

## Phase 4: User Story 2 — Keep data out of the repo and out of assistant context (P1)

**Goal**: Every entry point refuses paths inside the repo, and no path or filename reaches logs or errors.

**Independent test**: A guard failure exits with code 2 and a message that names only the setting. A forced error on a data file prints only placeholders.

- [X] T024 [US2] In `src/tpa_index/paths.py`, add `check_guard()`. It resolves both settings (`expanduser().resolve()`, which follows symlinks) and raises `GuardError(f"{name} must point outside the repo")` when `p.is_relative_to(repo_root())`. Make `data_dir()` and `output_dir()` call the check. Never put the path in a message.
- [X] T025 [US2] Write `tests/test_paths.py` with:
  - Failure cases, each with `monkeypatch.setenv`: inside the repo (`repo_root()/"data"`), a relative path `"data"` with `monkeypatch.chdir(repo_root())`, `~`-relative to a home-symlinked dir, a symlink in `tmp_path` pointing into the repo, and an unset setting. Each must raise `GuardError`, and the path string must not be in `str(err)`.
  - Passing cases: paths outside the repo.
  - A subprocess check that `import tpa_index.paths, tpa_index.cli` does not load `torch` (`'torch' not in sys.modules`).
- [X] T026 [US2] In `src/tpa_index/paths.py`, implement `redact(text)`, following research §9:
  - Replace the raw and resolved `DATA_DIR` and `OUTPUT_DIR` strings with `<DATA_DIR>` and `<OUTPUT_DIR>`.
  - For each video in `load_videos()`, replace the relative filename, the basename, and the stem (stems of 4+ characters only) with `<video:{video_id}>`. Longest strings go first.
  - Use a regex to replace `<DATA_DIR>/\S+` with `<DATA_DIR>/<file>`.
  - Cache the replacement list, and add `_reset_redaction_cache()` for tests.
- [X] T027 [US2] In `src/tpa_index/paths.py`, implement `install_redaction()`. It is idempotent and does two things:
  - Adds a `logging.Filter` subclass to the root logger's handlers (creating a basic `StreamHandler` if there are none). The filter sets `record.msg = redact(record.getMessage())` and `record.args = ()`.
  - Sets `sys.excepthook` to print `redact("".join(traceback.format_exception(...)))` to stderr.
- [X] T028 [US2] Add these to `tests/test_paths.py`:
  - `redact` removes the data root, the mock filename, the basename, and the stem;
  - `av.open(str(data_dir()/"mock/mock_live_01.mp4"))` raises an error whose `redact(str(e))` contains no part of the real path;
  - a `caplog` log line holding a data path is redacted;
  - a subprocess script that calls `install_redaction()` and raises `FileNotFoundError(<data path>)` has stderr containing `<DATA_DIR>` and not the path.
- [X] T029 [US2] In `src/tpa_index/cli.py`, make `main()` call `install_redaction()` and then `check_guard()` before dispatch. Map `GuardError` to exit 2 and `ConfigError` to exit 3, printing the redacted message to stderr. Add `tests/test_cli.py`, which runs `uv run tpa-index mock` with `OUTPUT_DIR` inside the repo and checks it exits 2 and that the path is not in stdout or stderr.
- [X] T030 [US2] Write `tests/test_repo_hygiene.py` checking that:
  - `git check-ignore config/videos.local.yaml .env` succeeds;
  - `.claude/settings.json` `permissions.deny` contains `Read(./config/videos.local.yaml)`;
  - `git ls-files` contains no `.mp4`, `.mov`, `.mkv`, `.avi`, `.wav`, `.parquet`, or `.duckdb` files;
  - if `config/videos.local.yaml` exists, no tracked file contains any of its filenames (on failure, report only a count, never the names).

**Checkpoint**: P1 is complete. Both guards and the mock dataset work.

---

## Phase 5: User Story 3 — Segment a video into fixed windows (P2)

**Goal**: For each window, produce segment rows, N letterboxed frames, and 16 kHz mono audio, kept in memory.

**Independent test**: On synthetic landscape, portrait, and no-audio clips, the segment count, IDs, frame shapes, letterbox, and audio length match the config.

- [ ] T031 [US3] Create `tests/clips.py` with `make_clip(path, w, h, seconds=20, fps=10, audio=True)`:
  - Encode with PyAV using codec `mpeg4` and `pix_fmt yuv420p`.
  - Make each frame a solid colour that changes every second (so sampling is checkable), with a 4-pixel white border so letterbox padding is detectable.
  - When `audio=True`, add an `aac` track with a 440 Hz sine at 16 kHz mono.

  In `tests/conftest.py`, add:
  - a session fixture `clips` that builds `t01` (160×90), `t02` (90×160), and `t03` (160×90, no audio) under `tmp_path_factory`;
  - a fixture `clip_videos` that points `DATA_DIR` at that folder and returns the matching video dicts (kinds live, clip, live).
- [ ] T032 [US3] In `src/tpa_index/segment.py`, implement:
  - `discover_videos(root=None)`: `rglob` under `data_dir()`, keeping `.mp4 .mov .m4v .mkv .avi` and skipping any path with a part starting with `.`.
  - `probe(path) -> dict`: use PyAV to get `duration_s` (stream duration × time_base, else container duration / 1e6), `fps` (`average_rate`), `orientation` (portrait if h > w, applying the display-matrix rotation if present), and `has_audio`.
- [ ] T033 [US3] In `src/tpa_index/segment.py`, implement `_letterbox(img)`, which pads an HxWx3 uint8 image with black to a centred square of side max(H, W) and never crops. Also implement `_sample_frames(container, stream, start, end, n)`:
  - Timestamps are `start + (i+0.5)*(end-start)/n`.
  - For each, `container.seek(int(t/stream.time_base), stream=stream, backward=True)`, then decode to the first frame with `frame.time >= t` (`to_ndarray(format="rgb24")`).
  - On a decode error, reuse the nearest good frame in the window. If no frame decodes, raise `RuntimeError("no decodable frames")`.
  - Return an array of shape `(n, S, S, 3)`.
- [ ] T034 [US3] In `src/tpa_index/segment.py`, implement `_load_audio(path, rate)`. Use `av.AudioResampler(format="flt", layout="mono", rate=rate)` over all audio frames, concatenated into a float32 array. If there is no audio stream, return `None`, and have the caller return `np.zeros(int((end-start)*rate), float32)` per window and log `"video %s has no audio track"` with the video_id only.
- [ ] T035 [US3] In `src/tpa_index/config.py`, add `effective_settings(pipeline, video)`: defaults, then `overrides.by_kind[video.kind]`, then `overrides.by_video[video.video_id]`, limited to `window_s`, `overlap_s`, and `frames_per_window`. In `src/tpa_index/segment.py`, add:
  - `Window = NamedTuple("Window", row=dict, frames=np.ndarray, audio=np.ndarray)`;
  - `segment_video(video, settings, pipeline)`: open `data_dir()/video["filename"]`, probe it, load audio once, then for each `windows(...)` yield a `Window`. The row has all segments columns, with signal columns `None`. Audio is sliced as `audio[int(s*rate):int(e*rate)]`, padded with zeros to the exact length. Close the container in `finally`.
- [ ] T036 [US3] Write `tests/test_segment.py` checking that:
  - `t01` (20 s, 8/2) gives 3 segments with IDs `t01_0`, `t01_6000`, and `t01_12000`, and the last `end_s` is 20;
  - each window's frames have shape `(4,160,160,3)`;
  - for `t02` (portrait), the left and right padding columns are all 0 and the centre column is non-zero;
  - for `t01`, the top and bottom rows are padding and the image is not cropped (the white border is present);
  - audio length is `(end-start)*16000`, with RMS > 0 for `t01` and RMS == 0 for `t03`;
  - the `clip` kind override (4/1) changes the segment count for `t02`;
  - `discover_videos` skips `._t01.mp4` and `.hidden/x.mp4` created in the data dir;
  - rows pass `validate("segments", …)` with null signal columns;
  - `git status --porcelain` is unchanged after segmenting.

**Checkpoint**: The segmenter works on the Mac with synthetic clips.

---

## Phase 6: User Story 4 — Record every stage run with cost (P2)

**Goal**: Every stage writes a runs row with runtime, peak memory, hardware, and config fingerprint. A failed run leaves no runs row.

**Independent test**: A dummy stage that allocates memory records plausible numbers. A failing stage re-raises and its run is ignored by `read_table`.

- [ ] T037 [US4] In `src/tpa_index/runs.py`, implement the following, importing torch lazily inside each function:
  - `device() -> str`: `"cuda"`, `"mps"`, or `"cpu"`.
  - `hardware_name()`: on CUDA, `torch.cuda.get_device_name(0)`; on darwin, `subprocess.run(["sysctl","-n","machdep.cpu.brand_string"])`; otherwise `platform.processor() or platform.machine()`.
  - `_peak_rss_gb()`: `resource.getrusage(RUSAGE_SELF).ru_maxrss`, in bytes on darwin and KiB on Linux, converted to GB.
  - `_peak_mem_gb(dev)`: on CUDA, `max_memory_allocated()/1e9`; on MPS, `max(torch.mps.driver_allocated_memory()/1e9, _peak_rss_gb())`; otherwise `_peak_rss_gb()`. Return `None` and log the reason on any exception.
- [ ] T038 [US4] In `src/tpa_index/runs.py`, implement a `track_run(stage, model=None, cfg_hash=None, root=None)` contextmanager yielding a `RunContext` dataclass (`run_id`, `n_segments=0`):
  - `run_id = f"{stage}-{utcnow:%Y%m%dT%H%M%S}-{secrets.token_hex(2)}"`.
  - On CUDA, `reset_peak_memory_stats()` first. Use `perf_counter`.
  - On normal exit, `write_table("runs", [row], run_id, root)`, where `cfg_hash` defaults to `config_hash(load_labels(), load_pipeline())`.
  - On exception, write nothing, log `"run %s failed"` (redacted via logging), and re-raise.
- [ ] T039 [US4] Write `tests/test_runs.py` checking that:
  - a stage allocating `np.ones(50_000_000)` and sleeping 0.1 s records `runtime_s >= 0.1`, `peak_mem_gb > 0.3` (or `None` only if unmeasurable), non-empty `hardware`, and a 12-hex `config_hash`;
  - `config_hash` changes when `pipeline["window_s"]` changes and is equal for identical dicts;
  - a stage that raises `ValueError` re-raises, has no `runs.parquet`, and is skipped by `read_table` even though it wrote `segments.parquet`.
- [ ] T040 [US4] Wire `tpa-index segment [--videos PATH] [--video-id ID ...]` in `src/tpa_index/cli.py`:
  - Load videos (from `--videos`, else `load_videos()`) and filter by ID; an unknown ID raises `ConfigError`.
  - Inside `track_run("segment")`, for each video, probe it to build a videos row (from config fields plus probed `duration_s`, `fps`, and `orientation`), then consume `segment_video` and collect the rows, discarding frames and audio.
  - Write `videos` and `segments` via `write_table` with `ctx.run_id`, and set `ctx.n_segments`.
  - A processing error exits 1 with a redacted message.
- [ ] T041 [US4] Add to `tests/test_cli.py`: write a temporary videos YAML for `t01`–`t03` in `tmp_path`, then run `uv run tpa-index segment --videos <yaml>` twice. Check that:
  - both exit 0;
  - `read_table("segments")` holds 2 identical sets of rows (excluding `run_id`), as a determinism proxy for Mac vs HPC;
  - `read_table("runs")` has 2 rows with `stage == "segment"` and `n_segments` equal to the segment count.

**Checkpoint**: P2 is complete.

---

## Phase 7: User Story 5 — Same code on the Mac and UNF HPC (P3)

**Goal**: One `pyproject.toml` works on both machines, there is a working `check` command, and there is a Slurm template.

**Independent test**: `tpa-index check` passes on the Mac. The sbatch file passes a syntax check and contains the required directives.

- [ ] T042 [US5] In `pyproject.toml`, add an explicit `[[tool.uv.index]]` with `name = "pytorch-cu130"`, `url = "https://download.pytorch.org/whl/cu130"`, and `explicit = true`. In `[tool.uv.sources]`, map `torch`, `torchvision`, and `torchaudio` to `{ index = "pytorch-cu130", marker = "sys_platform == 'linux'" }`. Run `uv lock`, then verify:
  - `uv.lock` lists `+cu130` wheels for linux and pypi wheels for darwin (grep);
  - `uv sync` still works on the Mac.
- [ ] T043 [US5] Implement `tpa-index check` in `src/tpa_index/cli.py`. It prints:
  - `guard: ok`;
  - `configs: ok (N labels, guideline X)`, after loading and validating all three configs;
  - `device: <cuda|mps|cpu>`;
  - `hardware: <name>`;
  - `torch: <version>`.

  It never prints paths.
- [ ] T044 [US5] Create `scripts/slurm/job.sbatch`, following research §15:
  - `#SBATCH` lines for `--job-name=tpa-index`, `--partition=gpu`, `--gres=gpu:6KSE:1`, `--cpus-per-task=8`, `--mem=32G`, `--time=02:00:00`, and `--output=slurm-%x-%j.out`;
  - a body with `set -euo pipefail`, `cd "$SLURM_SUBMIT_DIR"`, `nvidia-smi -L`, `uv sync --frozen`, and `uv run "$@"`;
  - a usage comment at the top: `sbatch scripts/slurm/job.sbatch tpa-index check`.
- [ ] T045 [US5] Add to `tests/test_cli.py`:
  - `tpa-index check` exits 0, its output has `device:` in {cuda, mps, cpu}, and contains neither the `DATA_DIR` nor the `OUTPUT_DIR` string;
  - `bash -n scripts/slurm/job.sbatch` succeeds, and the file contains each required `#SBATCH` directive (each checked by an assertion).
- [ ] T046 [US5] Write `README.md`, condensed from `quickstart.md`: purpose (one paragraph), install, `.env`, check, mock, segment, HPC sbatch usage, and a data-rules note pointing to the constitution and `.claude/settings.json`. Use no real paths or filenames.

---

## Phase 8: User Story 6 — Change labels and pipeline settings without code (P3)

**Goal**: Malformed config fails at load, with messages that name the entry.

**Independent test**: The shipped configs load, and each malformed variant raises `ConfigError` naming the label or key.

- [ ] T047 [US6] In `src/tpa_index/config.py`, add `_check_labels(d)`, called by `load_labels`, which checks:
  - `guideline_version` is a string;
  - names are unique;
  - `group` is in the allowed set;
  - there are 3–5 prompts, each a string;
  - `color` matches `^#[0-9a-fA-F]{6}$`;
  - every group is present;
  - the expression set is exactly the five allowed.

  Each failure raises `ConfigError(f"label {name}: {problem}")`.
- [ ] T048 [US6] In `src/tpa_index/config.py`, add `_check_pipeline(d)`, called by `load_pipeline`. It checks:
  - required keys and types;
  - `0 < overlap_s < window_s` and `frames_per_window >= 1`;
  - `0 <= label_threshold <= 1`;
  - `video_phases` is a non-empty list;
  - moments keys are valid and their groups are known;
  - overrides contain only the 3 allowed keys, and every effective combination still has `overlap_s < window_s`.

  Also add `_check_videos(list, phases)`, called by `load_videos`, which checks:
  - IDs are unique and match `^[a-z0-9]+$`;
  - filenames are relative with no `..`;
  - the enums are valid and `phase` is in `video_phases`.
- [ ] T049 [US6] Write `tests/test_config.py` checking that:
  - the shipped labels, pipeline, and example videos load;
  - each malformed variant (built by deep-copying the loaded dict and mutating it) raises `ConfigError` naming the entry: 2 prompts, 6 prompts, an unknown group, a bad color, a missing `guideline_version`, `overlap_s >= window_s`, an unknown override key, `phase` not in the list, a filename with `..`;
  - a banned-term check passes: no label name or prompt contains any of `anxious, depressed, adhd, autism, autistic, disabled, disability, trauma, mental, diagnosis, race, ethnicity, gender, age, religion, iq, risk`. Match whole words, case-insensitively.

---

## Phase 9: Polish & Checkpoint

- [ ] T050 Cross-check the code against the contracts. Make sure `SCHEMAS` in `src/tpa_index/schema.py` matches `contracts/tables.md`, and that the function names and signatures in `src/tpa_index/*.py` match `contracts/interfaces.md`. Fix whichever side is wrong (keep the contract unless it is a typo). Delete any unused code.
- [ ] T051 **Checkpoint (Mac)**:
  1. Run `uv run pytest`. All tests must pass. Record the pass count and duration.
  2. Run the quickstart in a scratch folder outside the repo:
     - Create a temporary `.env` with `DATA_DIR=<scratch>/data` and `OUTPUT_DIR=<scratch>/out`. Back up any existing `.env` first, and restore it after.
     - Run `uv run tpa-index check`.
     - Run `uv run tpa-index mock --seed 0`, then `uv run python -c "from tpa_index.schema import read_table, read_reviews; print({t: read_table(t).num_rows for t in ['videos','segments','predictions','moments','runs']}, read_reviews().num_rows)"`.
     - Generate a synthetic clip with `uv run python -c "from tests.clips import make_clip; make_clip('<scratch>/data/t01.mp4', 160, 90)"`. Write `<scratch>/videos.yaml` mapping it to `t01`, then run `uv run tpa-index segment --videos <scratch>/videos.yaml`.
  3. Confirm that `git status --porcelain` shows no new files except the intended source changes.
  4. Report runtime, peak memory, and hardware from the segment run's runs row.
  5. Delete the scratch folder.

---

## Dependencies & Execution Order

- **Setup (T001–T003)**, then **Foundational (T004–T009)**, which blocks every story.
- **US1 (T010–T023)** needs only Foundational. It is the MVP and unblocks the review app, insights, and eval work.
- **US2 (T024–T030)** needs Foundational. The tests in T028 use the mock filenames from T005.
- **US3 (T031–T036)** needs Foundational and US1 (`validate`). It uses US2's guard through `data_dir()`.
- **US4 (T037–T041)** needs US1 (`write_table`). T040 and T041 also need US3.
- **US5 (T042–T046)** needs US4 for `device()` and `hardware_name()` in `check`.
- **US6 (T047–T049)** needs Foundational only. It comes last by request; until then, configs load unvalidated.
- **Polish (T050–T051)** needs everything else.

Within each story, implement first and then write its tests in the next task. Run `uv run pytest` after each test task.

## Parallel Opportunities

None are marked. One developer on one machine works through the tasks in ID order.

## Implementation Strategy

1. **MVP**: T001–T023 (Setup, Foundational, and US1). Stop there and demo `tpa-index mock`, so teammates can start building on mock data.
2. **Safety**: T024–T030 (US2). Finish this before any real footage is touched.
3. **Pipeline core**: T031–T041 (US3 and US4).
4. **Environment and config hardening**: T042–T049.
5. **Checkpoint**: T050–T051.
