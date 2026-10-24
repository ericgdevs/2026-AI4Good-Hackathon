---
description: "Task list for 006-export-demo"
---

# Tasks: Official Export, Judge Demo and Mac Benchmark

**Input**: Design documents from `/specs/006-export-demo/` (plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md)

**Tests**: Required. The spec's FR-033 and every success criterion SC-001 to SC-010 have their own test task, marked **SC-00x test**.

**Rules for every task**:
- Mock data, synthetic clips and temporary dirs only (the existing `tests/conftest.py` isolation).
- Never read DATA_DIR, `~/tpa-data`, `/Volumes/T7`, the real reviews database, `config/videos.local.yaml` or `config/*.local.yaml`. Never start the app on real data. No ssh or rsync.
- No commits.
- Changes to `cli.py` stay in one contiguous block. `pipeline.yaml` is not changed.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency on an unfinished task)
- **[Story]**: US1 export · US2 timeline · US3 insights · US4 staff-defined labels · US5 Mac benchmark · US6 UI polish

## Decisions from the plan review (apply throughout)

- **Benchmark video**: chosen at run time from v08–v12 (all have every official stage), never stored in the repo. The summary separates **model-loading time** from **processing time**. `runs.runtime_s` already measures processing only, because every command loads its models before `track_run`. Loading = the step's wall time minus the runtime of the runs it wrote. So the four signal stages run as four separate calls.
- **Staff-led rule**: restated in `demo.py` for now. Phase 6 switches to 005's function if one exists.
- **Eval lock**: eval videos stay hidden in Timeline, the same lock as Review. The demo uses a dev video.
- **Readable names**: optional `display:` keys in `labels.yaml`, used **only** in the Timeline and Insights tabs. The Review tab's UI and its tests are not changed. The blind checklist is not changed.
- **Headline indicator IDs**: never assumed. `results.md` (005) lists the indicator IDs but does not pin which ones `headline.csv` contains (for example `solo_moments_count` vs `solo_moments_minutes`). The insights page charts every indicator present in `headline.csv`, in file order, and Phase 5/6 confirm the IDs against merged 005.

---

## Phase 1: Export + Devpost mock sample (US1, P1) 🎯 MVP

**Goal**: `tpa-index export` writes the case's annotation export (JSON + CSV, one record per official moment, 13 fields), with `--verified-only`. `tpa-index export --sample` writes the checked-in Devpost sample from mock data only.

**Independent test**: `uv run pytest tests/test_export.py` passes on mock moments and in-memory reviews covering every action.

- [X] T001 Confirm the baseline: on branch `006-export-demo`, run `uv run pytest` and note the pass count in the PR notes. Read `src/tpa_index/review.py` (`latest_state`, `ai_labels`, `review_row`, `observation_row`, `role_of`) and `src/tpa_index/schema.py` (`read_official`, `read_reviews`) before writing code.
- [X] T002 [P] [US1] Create `tests/test_export.py` with helpers:
  - `moment(mid, video_id, start, end, primary, secondaries, indicators, confidence)` returning a moments-table dict;
  - `rows(*specs)`, which builds review rows with `review.review_row` / `review.observation_row` using the `labels` fixture, and sets `created_at` explicitly (base time + n minutes) so order is deterministic.
- [X] T003 [P] [US1] **SC-001 test** in `tests/test_export.py`: `test_fields_exact_order`. For records built from at least 5 varied mock moments, `list(record)` equals `export.FIELDS` == `["video_id","start_s","end_s","primary_label","secondary_labels","indicators","contextual_observation","confidence","source","review_decision","guideline_version","reviewer_note","review_date"]`. After `write_export`, the JSON objects' key order and the CSV header row are identical to `FIELDS`, with no extra field.
- [X] T004 [P] [US1] **SC-002 test** in `tests/test_export.py`: `test_decision_table`, one moment per case:
  - **no review** → `ai` / `pending`, empty note and date;
  - **accept** → `accepted`;
  - **edit** → `edited`, `primary_label` = new label, `source` `staff`;
  - **reject** → `rejected`;
  - **defer** → `deferred`;
  - **reviewed twice** (accept, then reject): only the latest counts → `rejected`;
  - **equal `created_at`**: the tie is broken by `review_id`, the same on every run;
  - **rejected secondary**: dropped;
  - **edited secondary**: replaced;
  - **deferred secondary**: kept;
  - **staff-added secondary**: appended, `source` `staff`;
  - **added then rejected**: not present;
  - **observation only**: `source` stays `ai`.

  Also: `verified_only=True` removes exactly the `rejected` records. `guideline_version` comes from the primary's deciding review (use a 0.4.0 row), else the current config. `reviewer_note` / `review_date` come from the newest non-reject review on the moment.
- [X] T005 [P] [US1] Test file writing in `tests/test_export.py`: `test_files_never_overwritten`:
  - two `write_export` calls in the same second give two distinct timestamped pairs, and `annotations-latest.*` matches the second;
  - names match `annotations-\d{8}T\d{6}-[0-9a-f]{4}-g0\.5\.0\.(json|csv)`;
  - CSV round-trips through `csv.DictReader` with a note containing a comma, a quote, a line break and non-English text;
  - lists are joined with `;`, an empty list is an empty cell, empty text is `""` in JSON (never `null`);
  - `start_s` / `end_s` have 2 decimals and `confidence` has 3.
- [X] T006 [US1] Create `src/tpa_index/export.py` with `FIELDS`, `DECISIONS = {"accept": "accepted", "edit": "edited", "reject": "rejected", "defer": "deferred"}` and the pure function `build_records(moments, state, labels, verified_only=False) -> list[dict]`, following research R2 and contracts/export-format.md:
  - **state**: the `review.latest_state(rows, "review")` dict;
  - **staff adds**: state rows with action `add` and role secondary/indicator on that moment;
  - **observations**: role `observation` rows in `created_at` order, as `type` or `type: note`, joined with `; `, exactly as stored;
  - **`review_date`**: `created_at` UTC date as `YYYY-MM-DD`.

  No I/O and no label names in the code.
- [X] T007 [US1] In `src/tpa_index/export.py`, add `write_export(records, out_dir, guideline) -> tuple[Path, Path]`:
  - create `out_dir`;
  - timestamped names `annotations-{YYYYmmddTHHMMSS}-{secrets.token_hex(2)}-g{guideline}`, opened with mode `"x"` (retry the hex on `FileExistsError`);
  - JSON is a plain array (`ensure_ascii=False`, indent 2), CSV via `csv.writer` with list fields joined by `;`;
  - then write `annotations-latest.json` / `.csv` through a temp file and `os.replace`.

  It returns the timestamped paths.
- [X] T008 [US1] In `src/tpa_index/export.py`, add `export_official(root=None, verified_only=False) -> tuple[dict, tuple[Path, Path]]`:
  - `read_official("moments", root=root)`; if empty, raise `ConfigError("no official moments: list a moments or predict run in config/official_runs.yaml")`;
  - `review.latest_state(read_reviews(root), "review")`, `load_labels()`;
  - `build_records`, then `write_export` to `Path(root or output_dir()) / "exports"`.

  It returns per-decision counts and the paths.
- [X] T009 [US1] In `src/tpa_index/export.py`, add `sample_reviews(moments, labels) -> list[dict]` and `write_sample(dest=None)`:
  - In a `tempfile.TemporaryDirectory`, run `mock.generate(out_root=tmp, seed=0)`, then `read_table("moments", tmp)` sorted by (`video_id`, `start_s`), and take the first 20 moments (raise if fewer).
  - Script the reviews in memory with reviewers `mock-staff-a` / `mock-staff-b`:
    - every decision value at least twice;
    - one rejected, one edited and one deferred secondary;
    - one staff-added label;
    - one observation with a short note;
    - one label reviewed twice.
  - `build_records`, then write `docs/samples/annotation_export_sample.json` / `.csv` with fixed names. It never reads OUTPUT_DIR or `reviews.duckdb`.
- [X] T010 [US1] In `src/tpa_index/cli.py`, add the `export` subparser (`--verified-only`, `--sample`) in one block just before `args = parser.parse_args(argv)`, and a `_export(args)` handler next to the other handlers. It prints the per-decision counts and the two **file names only** (`path.name`), and returns 0. Errors flow through the existing exit-3 path.
- [X] T011 [US1] Run `uv run tpa-index export --sample` (mock only) to create `docs/samples/annotation_export_sample.json` and `.csv`. Open both and check:
  - about 20 records;
  - five decision values;
  - a staff-added label;
  - an observation;
  - `mock0N` IDs only.
- [X] T012 [P] [US1] **SC-003 test** in `tests/test_export.py`: `test_sample_is_mock_only_and_current`:
  - `write_sample(dest=tmp_path)` reproduces the checked-in `docs/samples/` files byte for byte. The generation must be deterministic: seed 0, fixed `created_at`, and review IDs derived from the moment ID and step, not random;
  - the sample has all five `review_decision` values, at least one staff-sourced record with an added label, and one non-empty `contextual_observation`;
  - a scan of both files finds none of the `filename` values from `config/videos.example.yaml`, no `/`-rooted path, no `.mp4`, no `mock-staff` reviewer name, and no transcript sentinel (the test plants `TRANSCRIPT_SENTINEL_006` in a `transcript.json` inside the temporary mock root by monkeypatching `mock.generate` to also write one).
- [X] T013 [US1] End-to-end test in `tests/test_export.py`: `test_export_official_cli`:
  - build a mock dataset in the test OUTPUT_DIR and list its run in a temporary official runs file (as `tests/test_official.py` does);
  - append mock reviews with `append_reviews`, and plant a transcript sentinel file in a run folder;
  - `cli.main(["export"])` returns 0 and writes the pair plus latest, and the printed text contains no directory path;
  - `cli.main(["export", "--verified-only"])` has no `rejected` records;
  - with no official runs, `cli.main(["export"])` returns 3;
  - neither file contains the sentinel, a reviewer name or a mock filename.
- [X] T014 [P] [US1] In `README.md`, under commands, add `tpa-index export [--verified-only]` (output folder, file names, fields link to `specs/006-export-demo/contracts/export-format.md`) and `tpa-index export --sample` (mock only, regenerates `docs/samples/`).
- [X] T015 **Checkpoint 1**: `uv run pytest` is green (baseline plus `tests/test_export.py`). `docs/samples/` exists and matches the code. US1 is demoable on mock data.

---

## Phase 2: Mac benchmark command (US5, P2)

**Goal**: `tpa-index benchmark --video-id X` runs segment, predict, transcript, audio, people, expression and moments on this machine for one video. Each stage writes its runs row. The command writes a Mac vs HPC summary per stage and end to end, separating model-loading time from processing time.

**Independent test**: `uv run pytest tests/test_benchmark.py` passes with a fake stage runner (no models, no real video).

- [X] T016 [P] [US5] Create `tests/test_benchmark.py` with a fake runner. `fake_runner(argv)` reads the stage and `--video-id` from argv, sleeps nothing, and writes a complete run to the test OUTPUT_DIR with `schema.write_table`:
  - `videos` (`duration_s` = 300 for the "Mac" video);
  - `runs` (`hardware` = "Apple M-test", `runtime_s` set per stage);
  - an optional failure for a named stage (raises).

  A fake clock (an iterator of `perf_counter` values) makes the wall time per step exact. Also write mock "HPC" official runs (`hardware` = "NVIDIA test GPU") for some stages, plus a temporary official runs file listing them.
- [X] T017 [US5] **SC-007 test** in `tests/test_benchmark.py`: `test_benchmark_summary`, using `run_benchmark(video_id, runner=fake_runner, clock=fake_clock)`.
  - **Steps**: seven calls in order, each with `--video-id X`. The signals calls are `["signals","--stages",s,"--video-id",X,"--force"]` for s in transcript, audio, people, expression.
  - **Mac per-stage values**:
    - `mac_processing_s` = the run's `runtime_s`;
    - `mac_load_s` = wall − runtime (never negative);
    - `mac_s_per_video_hour` = `runtime_s / (300/3600)`;
    - peak memory, model and hardware are present.
  - **HPC values**: summed runtime ÷ summed video-hours of the official runs. "no HPC run" where none is listed.
  - **Total row**: sums processing per video-hour, sums load time, and takes the maximum peak memory.
  - **Failure**: a failing stage gives status `failed`, the later steps still run, the total says `incomplete: missing …`, and `cli.main(["benchmark", ...])` returns 1.
  - **Repo files untouched**: `config/official_runs.yaml` (and the temporary official file) are byte-identical before and after.
  - **No filenames**: the JSON and MD summaries contain no filename, no `.mp4` and no absolute path.
- [X] T018 [US5] Create `src/tpa_index/benchmark.py` with `STAGES = ("segment","predict","transcript","audio","people","expression","moments")` and `steps(video_id, videos=None) -> list[tuple[str, list[str]]]` (argv per stage; `--videos` is passed to segment, predict and signals when given). Add `run_benchmark(video_id, videos=None, runner=None, clock=time.perf_counter, root=None) -> list[dict]`. For each step it:
  1. records the run IDs from `schema.list_runs(root)`;
  2. times `runner(argv)` (default `cli.main`, imported lazily), catching any exception or non-zero return as `failed` with a redacted message;
  3. collects the new runs of that stage covering the video.
- [X] T019 [US5] In `src/tpa_index/benchmark.py`, add `summarize(results, root=None) -> dict`.
  - **Duration** comes from the run folder's `videos` table (`read_table("videos", root, run_ids=[rid])`).
  - **Mac fields**: `mac_processing_s`, `mac_load_s = max(wall - sum(runtime), 0)`, `mac_s_per_video_hour` (processing only), `mac_peak_mem_gb`, `mac_model`, `mac_hardware`.
  - **HPC fields**: from `schema.load_official()` and `list_runs()` for the same stage, as sum(runtime) / sum(video-hours), plus the maximum peak memory and the set of models and hardware.
  - **Status**: `ok` / `failed: …` / `no HPC run`.
  - **Total row**: as in T017.
  - **Notes**:
    - "Processing time excludes model loading; loading is shown once per stage";
    - "Mac uses the smaller Mac models from pipeline.yaml";
    - "Benchmark runs are not official".
- [X] T020 [US5] In `src/tpa_index/benchmark.py`, add `write_summary(summary, root=None) -> tuple[Path, Path]`. It writes `{OUTPUT_DIR}/benchmarks/benchmark-{YYYYmmddTHHMMSS}-{hex}.json` and `.md` (exclusive create). The MD table has columns stage, Mac s/video-hour (processing), Mac load s, Mac peak GB, HPC s/video-hour, HPC peak GB, models and status, followed by the notes. There is no path in the content.
- [X] T021 [US5] In `src/tpa_index/cli.py`, in the same block as `export`, add a `benchmark` subparser (`--video-id` required, a single value; `--videos`) and a `_benchmark(args)` handler. It runs the benchmark, prints the MD table and the summary file **names**, and returns 0 if every step is ok, else 1. It never writes `config/official_runs.yaml`.
- [X] T022 [P] [US5] In `README.md`, add a "Mac benchmark" section:
  - `tpa-index benchmark --video-id <id>` with the id typed at run time from the promos that have every official stage, never committed;
  - what the summary shows (processing vs loading);
  - benchmark runs never go into `config/official_runs.yaml`.
- [ ] T023 [US5] **Team only (not an assistant)**: on the Mac, run `uv run tpa-index benchmark --video-id <one of v08–v12>`, check that `tpa-index runs` shows seven new rows with runtime and peak memory, and copy only the aggregate per-video-hour table (no video ID) into the deck notes. This closes the real-data half of SC-007.
- [X] T024 **Checkpoint 2**: `uv run pytest` is green (including `tests/test_benchmark.py`). The benchmark command is ready for T023.

---

## Phase 3: Theme + guideline-history check (US6 palette, US4 staff-defined labels, P2/P3)

**Goal**: the TPA palette and credit live in `config/theme.yaml` (validated, WCAG AA) and drive the Gradio theme. A new observation type is offered in Review after a guideline bump; without a bump, `check` and the app refuse. The blind checklist never changes.

**Independent test**: `uv run pytest tests/test_theme.py tests/test_guideline.py` passes.

- [X] T025 [P] [US6] Create `config/theme.yaml` with:
  - `background: "#F8F7F8"`, `surface: "#FFFFFF"`;
  - `text: "#201C25"`, `text_secondary: "#6F717A"`;
  - `accent: "#7851B9"`, `accent_deep: "#4100AD"`;
  - `chart_series: ["#7851B9", "#0099B3", "#FF4255"]`;
  - `credit: "Built for The Performance Academy"`;
  - a header comment: "The only place app colours are defined. Teal and coral are chart fills and borders only (they fail AA as text)."
- [X] T026 [US6] In `src/tpa_index/config.py`, add `contrast_ratio(a, b) -> float` (WCAG relative luminance) and `load_theme(path=None) -> dict` (default `CONFIG_DIR / "theme.yaml"`). It:
  - validates `#RRGGBB` for every colour, at least 3 `chart_series` entries and a non-empty `credit`;
  - requires ≥ 4.5 for `text`, `text_secondary`, `accent` and `accent_deep` on both `background` and `surface`, and for `surface` on `accent` (button text);
  - raises `ConfigError` naming the failing pair and its ratio.
- [X] T027 [US6] In `src/tpa_index/app.py`, change `theme()` to build `gr.themes.Default(...)` and `.set(...)` from `load_theme()`: body background, body/block text, subdued text, primary button background, hover and text, selected-tab and link colour (accent), block titles and headers (accent_deep). No colour literal is left in `app.py`. Keep `PAGE_JS` forcing light.
- [X] T028 [US6] In `src/tpa_index/app.py` `build_app`, change the header to `TPA video index (local only)`, followed by a secondary-text line with `theme["credit"]` (`elem_id="app-credit-header"`), and add a footer `gr.Markdown(theme["credit"], elem_id="app-credit-footer")` after the tabs. No image or logo component. The Blind labeling and Review tab contents are untouched. If an existing `tests/test_app.py` assertion checks the old header text, update only that assertion.
- [X] T029 [P] [US6] **SC-010 test** in `tests/test_theme.py`: `test_palette_meets_wcag_aa`:
  - the shipped `load_theme()` passes;
  - the computed ratios match SC-010 to 2 decimals (15.67, 4.55, 5.35, 10.75 on `#F8F7F8`; 16.74, 4.86, 5.71, 11.49 on white; 5.71 for white on accent);
  - a palette with `text_secondary: "#9A9CA3"` raises `ConfigError` naming the pair;
  - teal and coral never appear in any text slot of `theme()`;
  - `git ls-files src config` content search (case-insensitive) finds each palette hex only in `config/theme.yaml`.
- [X] T030 [P] [US6] Test the light theme and credit in `tests/test_theme.py`: `test_light_theme_and_credit`:
  - the `theme()` variables equal the palette values;
  - `build_app` (mock env as in `tests/test_app.py`) has the credit in `app-credit-header` and `app-credit-footer`, and no `Image` component;
  - `launch_kwargs` still passes `js=PAGE_JS`, and `PAGE_JS` contains `__theme` / `light`.
- [X] T031 [US4] In `src/tpa_index/config.py`, add:
  - `GUIDELINE_HISTORY = Path(os.environ.get("TPA_GUIDELINE_HISTORY", CONFIG_DIR / "guideline_history.yaml"))`;
  - `guideline_fingerprint(labels) -> str`: sha256[:12] of JSON over sorted (name, group, source) for every label plus sorted observation-type names; prompts, colours, descriptions and `display` are excluded;
  - `check_guideline(labels, path=None) -> str`, which returns a status line:
    - recorded and matching → `guideline: ok (0.5.0)`;
    - not recorded → `guideline: 0.6.0 not recorded; add to config/guideline_history.yaml:  "0.6.0": "<fp>"`;
    - recorded and different → raise `ConfigError("labels: vocabulary changed without bumping guideline_version (0.5.0)")`.

  It is not called from `load_labels()`.
- [X] T032 [US4] Create `config/guideline_history.yaml` with a header comment ("one line per released guideline version; written by hand from `tpa-index check`") and `"0.5.0": "<fingerprint>"`, computed with `uv run python -c "from tpa_index.config import load_labels, guideline_fingerprint; print(guideline_fingerprint(load_labels()))"`.
- [X] T033 [US4] Wire the check:
  - in `src/tpa_index/cli.py` `_check`, print `check_guideline(labels)` after the configs line (one added line);
  - in `src/tpa_index/app.py` `launch()`, call `check_guideline(load_labels())` before `build_app`, so a `ConfigError` stops the launch with the message;
  - in `tests/conftest.py`, add `TPA_GUIDELINE_HISTORY` / `config.GUIDELINE_HISTORY` isolation pointing at the real tracked file (it is not local data), so tests can override it.
- [X] T034 [P] [US4] **SC-006 test** in `tests/test_guideline.py`: `test_new_observation_type`:
  - copy `labels.yaml` to `tmp_path` and add `{name: interview, description: "a youth speaking to camera about their work"}` under `observation_types`;
  - **without a bump**: `check_guideline(load_labels(copy))` raises with "without bumping";
  - **with `guideline_version: "0.6.0"`**: it returns the "not recorded" line;
  - **checklist**: `json.dumps(review.checklist(new, pipeline), sort_keys=True)` equals the same for the shipped labels (byte-identical);
  - **Review picker**: `ReviewController(labels=new, pipeline=...)` lists `interview` in `.observations`, and `build_app(labels=new, ...)` has `interview` in the `review-add-obs` dropdown choices.
- [X] T035 [P] [US4] Test the fingerprint and history in `tests/test_guideline.py`: `test_fingerprint_rules`:
  - the fingerprint is unchanged by prompt, colour, description or `display` edits;
  - it changes on a label rename, a group change or an added or removed observation type;
  - the shipped labels match the `0.5.0` entry in `config/guideline_history.yaml` (this guards future edits);
  - `cli.main(["check"])` output includes `guideline: ok (0.5.0)` (monkeypatch out the torch/device lines if slow).
- [X] T036 [P] [US4] Test the US4 export cases in `tests/test_export.py`: `test_observation_types_in_export`:
  - an observation saved under 0.6.0 with type `interview` exports as `interview: …`, and the record keeps its guideline version when it is the deciding review;
  - an observation whose type was later removed from config exports exactly as stored (Clarification 8).
- [X] T037 **Checkpoint 3**: `uv run pytest` is green (including `tests/test_theme.py` and `tests/test_guideline.py`). Phases 1–3 are complete and do not depend on 005.

---

## Phase 4: Timeline tab (US2, P1) ⏳ AFTER THE 005 MERGE

**Prerequisite**: 005-insights-eval is merged into `main` and this branch is rebased onto it.

**Goal**: a Timeline tab in the same app. Per video ("Video N · phase"), it shows moments plus per-segment people, speech yes/no, audio cues, the expression marked "AI suggestion, unvalidated" and who led. Staff-led segments are hidden by default. No transcript text or filenames.

**Independent test**: `uv run pytest tests/test_demo.py` passes on mock data with a planted transcript.

- [ ] T038 Rebase `006-export-demo` onto `main` after the 005 merge and resolve the expected small conflict in `cli.main()` (005's `evaluate` / `insights` subparsers next to `export` / `benchmark`). Run `uv run pytest` to green. Read 005's merged staff-led code (expected in `src/tpa_index/indicators.py`) and note whether it exposes a reusable per-segment function and its signature, for T059.
- [ ] T039 [P] [US2] In `config/labels.yaml`, add optional `display:` keys where the automatic form reads badly:
  - `lyricism_spoken_word: "Lyricism / spoken word"`;
  - `acting_theater: "Acting / theater"`;
  - `podcast_interview: "Podcast / interview"`;
  - `modeling_runway: "Modeling / runway"`;
  - `actor_participant: "Youth-led"`;
  - `actor_staff_facilitation: "Staff-led"`;
  - `singing_voice: "Singing voice"`.

  Add a comment that `display` is presentation only, not guideline content (no bump). In `src/tpa_index/config.py` `_check_labels`, allow an optional non-empty string `display`. Then confirm that `guideline_fingerprint` still matches `0.5.0`, the blind checklist snapshot test passes, and `tests/test_review.py` / `tests/test_app.py` pass **unchanged**.
- [ ] T040 [US2] Create `src/tpa_index/demo.py` with:
  - `display_label(name, labels)`: the `display` key, else `name.replace("_", " ")` with the first letter capitalised;
  - `video_name(video, shown_videos)`: `f"Video {int(trailing digits)} · {phase}"`, with `phase` `unknown` → "phase not set", and ` · {kind}` appended only when two shown videos would get the same name. It never reads `filename`.
- [ ] T041 [US2] In `src/tpa_index/demo.py`, add the constants `STAFF_ACTOR = "actor_staff_facilitation"`, `INSTRUCTION = "instruction"`, `PARTICIPANT = "actor_participant"`, plus `check_rule_labels(labels)`, which raises `ConfigError` with a plain message if any is missing.

  Also add `staff_led(segment_id, verified: set[str] | None, scores: dict[str, float], thresholds: dict[str, float]) -> tuple[bool, str]`, returning (staff_led, tag), with tag "staff-verified" or "model estimate":
  - **verified actor labels**: staff-led = STAFF_ACTOR in the set and PARTICIPANT not in it;
  - **model only**: (STAFF_ACTOR ≥ t or INSTRUCTION ≥ t) and PARTICIPANT < t.

  Add a comment: `# shortcut: restates 005's rule (spec FR-018); switch to 005's function after the merge (T059)`.
- [ ] T042 [US2] In `src/tpa_index/demo.py`, add `DemoController(root=None, labels=None, pipeline=None)`. `reload()` reads:
  - `read_official("moments")`;
  - `schema.read_signals()` with `transcript_chars` dropped immediately (never stored on the controller);
  - main predictions (the most common non-rules model, as in `ReviewController`);
  - `read_official("predictions", stage="audio")` and `stage="expression"`;
  - thresholds via `config.threshold`;
  - blind state (`latest_state(..., "blind")`, actor-group labels per segment, not rejected) and review state (actor labels per moment).

  Its methods:
  - **`video_choices()`**: from `review.review_videos()` (eval lock kept), as `(video_name, video_id)` pairs, plus the lock reason.
  - **`moments(video_id)`**: rows with time, readable primary, readable other labels, confidence and review status (`review.moment_status`).
  - **`segments(video_id, moment=None, show_staff=False)`**: returns (rows, hidden_count). Rows come from an allow-list only:
    - time;
    - people (or "not available");
    - speech (Yes / No / not available);
    - audio cues (readable names at or above threshold, or "none" / "not available");
    - expression (`AI suggestion, unvalidated: {readable}`, "none", "no faces" or "not available");
    - led by (`Youth` / `Staff-led (staff-verified)` / `Staff-led (model estimate)`).
- [ ] T043 [US2] In `src/tpa_index/app.py` `build_app`, append a `Timeline` tab after the Review tab, containing:
  - a video dropdown (`elem_id="timeline-video"`) and a lock note;
  - a moments dataframe (`timeline-moments`) and a segments dataframe (`timeline-segments`);
  - a "Show staff-led segments" checkbox, default off (`timeline-show-staff`), plus a hidden-count line;
  - a clip (`timeline-clip`) via the existing `_cut`;
  - the note "Signals are observable cues from AI models, unvalidated. Staff-led time is not counted in youth indicators."

  The existing Reload button also calls `demo.reload()`. The Review tab code is unchanged.
- [ ] T044 [P] [US2] Test the timeline signals in `tests/test_demo.py`: `test_timeline_signals`, on a mock dataset plus official signal runs written for one mock video (people, transcript, audio and expression stages, built like `tests/test_signals.py` fixtures):
  - each segment row matches the mock values;
  - a stage with no official run shows "not available";
  - `faces_detected = 0` gives "no faces";
  - the expression text starts with `AI suggestion, unvalidated:`;
  - audio cues use display names (`Singing voice`);
  - moments show readable labels (`Lyricism / spoken word`).
- [ ] T045 [P] [US2] **SC-005 test** in `tests/test_demo.py`: `test_staff_led_excluded_by_default`.
  - **Rule table** (mirrors 005's cases):
    - verified {staff} → staff-led, staff-verified;
    - verified {staff, participant} → youth;
    - model: facilitation ≥ t, participant < t → staff-led, model estimate;
    - instruction ≥ t, participant < t → staff-led;
    - facilitation ≥ t, participant ≥ t → youth;
    - all below → youth.
  - **Precedence**: a blind segment label beats a moment review label, which beats model scores.
  - **Default view**: `segments(...)` with defaults returns zero staff-led rows and the right `hidden_count`; `show_staff=True` returns them with their tag.
  - **Startup check**: `check_rule_labels` raises when a rule label is renamed.
- [ ] T046 [P] [US2] **SC-004 test (Timeline part)** in `tests/test_demo.py`: `test_timeline_never_shows_transcript`:
  - plant `transcript.json` containing `TRANSCRIPT_SENTINEL_006 hello world` in the mock transcript run folder, and set `transcript_chars = 987654`;
  - the outputs of `video_choices`, `moments` and `segments` (both toggle states), and `json.dumps` of the `build_app(...).config`, contain neither the sentinel nor `987654`;
  - no row key or header mentions `transcript`.
- [ ] T047 [P] [US2] Test the eval lock and names in `tests/test_demo.py`: `test_names_and_eval_lock`:
  - an eval-split mock video is absent from `video_choices()` while blind labeling is unfinished, and the lock reason matches the Review tab's;
  - names read `Video 1 · early`, with kind appended on a forced name collision;
  - no `filename` value appears in any choice;
  - `tests/test_app.py` and `tests/test_review.py` pass unmodified.
- [ ] T048 **Checkpoint 4**: `uv run pytest` is green (including `tests/test_demo.py`). The Timeline is demoable on mock data with a dev video.

---

## Phase 5: Insights tab (US3, P2) ⏳ AFTER THE 005 MERGE

**Goal**: an Insights tab that reads the newest complete `results/insights-*` folder (005 contract `results.md`, `schema_version` 1). It charts the headline indicators early vs showcase (mid only if present), shows every tag verbatim, adds a staff-led and idle context line, and has a clear empty state.

**Independent test**: `uv run pytest tests/test_insights.py` passes against fixture folders.

- [ ] T049 [US3] Read the merged `specs/005-insights-eval/contracts/results.md` and 005's headline code and tests (read-only), and write down in a comment at the top of `tests/test_insights.py`:
  - the exact indicator IDs 005 writes to `headline.csv` (for example whether solo is `solo_moments_count`, `solo_moments_minutes` or both);
  - the column names;
  - the tag strings;
  - `schema_version`.

  Do not hardcode headline IDs in `demo.py`: the page charts every indicator present in `headline.csv`, in file order.
- [ ] T050 [P] [US3] Create fixtures under `tests/fixtures/results/`:
  - **`insights-20261024T213040-3c4d/`** (complete):
    - `manifest.json`: `schema_version` 1, `kind` insights, `created_at`, `limitations`;
    - `headline.csv`: the four headline indicators from T049, with early and showcase, using all three tag forms (`staff-verified`, `model estimate (unvalidated)`, `mixed (40% staff-verified)`), plus one `status` = `no videos in this phase` row;
    - `time_breakdown.csv`, including scope `all`;
    - `indicators.csv`.
  - **`insights-20261024T220000-aaaa/`**: newer, without `manifest.json` (incomplete).
  - **`evaluate-20261024T230000-bbbb/`**: complete, but must be ignored.
  - **`with_mid/insights-20261024T213041-cccc/`**: a variant with mid rows.

  Aggregates only. No filenames.
- [ ] T051 [US3] In `src/tpa_index/demo.py`, add `InsightsController(root=None)`:
  - **`latest_folder()`**: the last by name of the `results/insights-*` folders that contain `manifest.json`.
  - **`load()`** returns a state dict: available / not available / unsupported. It is unsupported when `schema_version != 1` or `headline.csv` lacks `indicator, phase, value, unit, tag, verified_share, change_vs_early` (columns are selected by name with `csv.DictReader`).
  - **`table()`**: rows of indicator (display name via a readable form of the ID), phase, value with unit (share → `%`, minutes → `min`, count, ratio), tag verbatim and change vs early.
  - **`figures()`**: one matplotlib figure per headline indicator:
    - white background, `load_theme()["chart_series"]` colours, legend;
    - phases in order early, mid (only if present), showcase;
    - each bar labelled `value · tag` in `theme["text"]`;
    - `status != ok` rows drawn as text instead of a bar.
  - **`context()`**: staff-led (verified + model) and idle minutes for scope `all` from `time_breakdown.csv`, labelled "excluded from youth indicators", plus the manifest's `limitations`.
  - **`source_line()`**: built from `manifest.created_at`, never the folder path.

  Empty state: "Insights are not available yet. Run the insights step first." (plus "(unsupported results format)" when unsupported).
- [ ] T052 [US3] In `src/tpa_index/app.py` `build_app`, append an `Insights` tab after Timeline, containing:
  - a source line (`insights-source`);
  - one `gr.Plot` per figure (`insights-chart-{i}`);
  - a table (`insights-table`);
  - context and limitations Markdown (`insights-context`);
  - an empty-state Markdown.

  The Reload button refreshes it. The rest of the app works when results are missing.
- [ ] T053 [P] [US3] Test the insights page in `tests/test_insights.py`: `test_insights_reads_newest_complete`, with the fixtures copied into the test `OUTPUT_DIR/results/`:
  - the complete insights folder is chosen (the newer folder without a manifest and the evaluate folder are ignored);
  - every number in `table()` carries exactly the CSV's tag string;
  - mid is absent for the base fixture and present for `with_mid`;
  - the status row shows `no videos in this phase` and no bar;
  - `schema_version` 2 → the unsupported message;
  - a missing `results/` → the not-available message, and `build_app` still builds;
  - figure backgrounds are white, and bar colours come from `chart_series`.
- [ ] T054 [P] [US3] **SC-004 test (Insights part)** in `tests/test_insights.py`: `test_insights_never_shows_transcript`. With a planted `TRANSCRIPT_SENTINEL_006` transcript in an OUTPUT_DIR run folder and the fixtures in place, none of `source_line`, `table`, `context`, figure texts or `build_app(...).config` contains the sentinel or a transcript character count.
- [ ] T055 [P] [US3] **SC-008 test** in `tests/test_insights.py`: `test_judge_flow_is_fast`, on the 4-video mock dataset plus fixtures:
  - `DemoController` construction plus `segments()` for one video takes under 3 s;
  - `InsightsController.load()` plus `figures()` takes under 2 s (`time.perf_counter`);
  - each tab's first screen needs no raw ID: choices use `Video N · phase`, and no `segment_id` appears in any table.
- [ ] T056 [P] **SC-009 test** in a new `tests/test_no_filenames.py`: `test_no_filenames_anywhere`. Scan all of the following for every `filename` in `config/videos.example.yaml`, `.mp4`, and the absolute DATA_DIR and OUTPUT_DIR paths:
  - the export pair and latest (from T013's setup);
  - the benchmark JSON and MD (from T016's fake runner);
  - every `DemoController` and `InsightsController` output;
  - `json.dumps(build_app(...).config)`.

  Zero occurrences.
- [ ] T057 [US3] Manual walkthrough on mock data only (`tpa-index mock --clips` in a temp OUTPUT_DIR, fixtures copied in). Time a fresh open:
  - to a dev video's timeline with signals: under 1 min;
  - then to the insights headline: under 30 s more.

  Check light mode with the OS in dark mode, the credit in the header and footer, and no logo. Take mock-only screenshots for the deck. This is the manual half of SC-008.
- [ ] T058 **Checkpoint 5**: `uv run pytest` is green (including `tests/test_insights.py` and `tests/test_no_filenames.py`). All six user stories are demoable on mock data.

---

## Phase 6: Switch-to-005 follow-ups ⏳ AFTER THE 005 MERGE

**Goal**: remove the temporary duplication with 005 and prove the two features fit together on mock data.

- [ ] T059 [US2] If T038 found a reusable per-segment staff-led function in 005 (expected in `src/tpa_index/indicators.py`), replace the body of `demo.staff_led` in `src/tpa_index/demo.py` with a call to it (adapting arguments), and delete the `shortcut:` comment. `tests/test_demo.py::test_staff_led_excluded_by_default` must pass unchanged. If no such function exists, leave the copy and update the comment to name the 005 function or file it mirrors.
- [ ] T060 [US2] If merged `config/pipeline.yaml` has an `insights:` block naming the staff-led rule's labels, read them from there in `src/tpa_index/demo.py` (in place of the three constants), and keep `check_rule_labels` as the validation. If not, leave the constants as they are.
- [ ] T061 [US3] Re-check the merged `results.md` against `tests/fixtures/results/` (headline IDs, columns, tag strings, `schema_version`), and update the fixtures and the T049 comment if 005 changed anything. If `schema_version` moved past 1, make `InsightsController` accept the new version only after confirming the columns it reads are unchanged.
- [ ] T062 [US3] Add an integration test in `tests/test_insights.py`: `test_reads_real_005_output_on_mock`. Generate a mock dataset with official runs in the test OUTPUT_DIR, run 005's `cli.main(["insights"])` on it, and assert `InsightsController().load()` is available, that every `headline.csv` row appears in `table()` with its tag verbatim, and that the staff-led context matches `time_breakdown.csv`. Mock data only.
- [ ] T063 **Checkpoint 6**:
  - `README.md` gains a short "Demo" section (Timeline and Insights tabs, the eval lock, mock-only screenshots);
  - `CLAUDE.md` is still correct;
  - `uv run pytest` is fully green;
  - `git status` shows only intended files (no data, no `.local` files, nothing staged).

---

## Dependencies & Execution Order

- **Phase 1 → Phase 2 → Phase 3** can run now, before 005 merges. Phases 2 and 3 don't depend on Phase 1's code and could run in parallel, but the requested order is sequential with checkpoints.
- **Phases 4, 5 and 6 wait for the 005 merge** (T038 rebases).
  - Phase 5 uses `demo.py` and the theme from Phases 3–4.
  - Phase 6 uses both.
- **Within each phase**: write the test tasks first (they should fail), then the implementation, then the checkpoint.
- **`cli.py`**: T010 and T021 edit the same block in sequence. T033 is a one-line edit in `_check`.
- **`app.py`**: edited in order T027 → T028 → T033 → T043 → T052. Never in parallel.
- **`demo.py`**: edited in order T040 → T041 → T042 → T051 → T059/T060.

## Parallel Opportunities

- **Phase 1**: T002–T005 (test scaffolding and SC-001/SC-002 tests, same new file but independent functions; write them in one sitting), plus T014.
- **Phase 2**: T016 with T022.
- **Phase 3**: T025, T029, T030, T034, T035 and T036 once T026 and T031 exist.
- **Phase 4**: T039 alongside T040–T042. T044–T047 in parallel after T042.
- **Phase 5**: T050 alongside T051. T053–T056 in parallel after T052.

## Implementation Strategy

1. **MVP = Phase 1.** The case-required export and the Devpost sample, mock-verified.
2. Phase 2 gives the "fits a Mac" proof (the team runs T023 on a real promo).
3. Phase 3 gives judge-ready visuals and the staff-owned vocabulary demo.
4. After 005 merges: Phase 4 (Timeline), Phase 5 (Insights), Phase 6 (de-duplication and integration).

## Success-criterion test map

| SC | Test task |
|---|---|
| SC-001 | T003 |
| SC-002 | T004 |
| SC-003 | T012 |
| SC-004 | T046 (Timeline), T054 (Insights) |
| SC-005 | T045 |
| SC-006 | T034 |
| SC-007 | T017 (and T023, team run) |
| SC-008 | T055 (and T057, manual) |
| SC-009 | T056 |
| SC-010 | T029 |
