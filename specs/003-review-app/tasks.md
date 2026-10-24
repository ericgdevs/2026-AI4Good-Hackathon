# Tasks: Review App

**Input**: Design documents from `/specs/003-review-app/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ (interfaces.md, config.md), quickstart.md; 001 and 002 code on `main`

**Tests**: Requested. Each phase has its own tests. Run `uv run pytest` after every test task.

**Organization**: Priority order:
1. **US1**: blind labeling (P1)
2. **US5**: safety (P1)
3. **US2**: overlap set and top-up (P1)
4. **Checkpoint 1**: blind mode end to end on mock data (the Saturday-morning target)
5. **US3**: review timeline (P2)
6. **US4**: review actions (P2)

There is one developer and no `[P]` markers. Each task takes 15–30 minutes.

**Ground rules for every task**:
- **No hardcoded label names or groups.** The label vocabulary is about to change. Every checklist item, key binding, label-role decision, timeline column and filter value MUST come from `load_labels()` and `load_pipeline()` at runtime. `src/tpa_index/review.py`, `clips.py` and `app.py` must not contain any label name or group name as a string literal. A test enforces this (T008).
- Never read `.env`, `config/videos.local.yaml`, `config/thresholds.local.yaml` or anything under the real `DATA_DIR`/`OUTPUT_DIR`. Tests and checkpoints use mock data and synthetic clips in scratch folders, through `DATA_DIR`, `OUTPUT_DIR`, `TPA_LOCAL_VIDEOS`, `TPA_THRESHOLDS_FILE` and `TPA_OFFICIAL_RUNS`.
- The app binds `127.0.0.1` only, never sets `share=True`, and is never pointed at real footage by the assistant.
- Match the contracts. If code must differ from them, update the contract in the same task.

## Format: `[ID] [Story] Description`

## Phase 1: Setup

- [X] T001 Move `make_clip` from `tests/clips.py` to `src/tpa_index/synthetic.py`, unchanged, with its module docstring "Tiny synthetic test clips (no real footage)". Turn `tests/clips.py` into a one-line re-export: `from tpa_index.synthetic import COLORS, make_clip  # noqa: F401`. Run `uv run pytest` to confirm everything still passes.
- [X] T002 Add the `app:` section to `config/pipeline.yaml`, exactly as in `contracts/config.md`, including `blind_groups`, `indicator_groups`, `uncertain_label` and `skip_observation`. In `src/tpa_index/config.py`, add `_check_app(app, pipeline, labels)`, called from `_check_pipeline_labels`, applying every validation rule in `contracts/config.md`. Errors look like `ConfigError("pipeline.app: …")`. In `tests/test_config.py`, add one failing case per rule:
  - an `overlap_size` larger than `sample_size`;
  - an unknown group in `blind_groups`;
  - a primary group missing from `blind_groups`;
  - an `uncertain_label` that is not a rules label;
  - a `skip_observation` that is not an observation type;
  - a `padding_s` of 31;
  - a `port` of 80.
- [X] T003 Add `observation_types` to `config/labels.yaml`, exactly as in `contracts/config.md`. Bump `guideline_version` to `"0.4.0"` and add the header comment that adding a type (or a label) requires a bump. In `_check_labels` (`src/tpa_index/config.py`), validate:
  - names are unique, match `^[a-z][a-z0-9_]*$`, and differ from every label name;
  - each description is a non-empty string.

  Add `observation_types(labels) -> list[str]`. Add tests: a duplicate name, a name clashing with a label, and a bad name pattern each fail; the shipped file loads with guideline `0.4.0`.
- [X] T004 In `src/tpa_index/schema.py`:
  - Add `"observation"` to the `reviews` `label_role` enum.
  - Add the `blind_queue` pyarrow schema (columns and nullability from `data-model.md`), with `RULES`: primary key (`queue`, `position`), `queue` enum {main, overlap, topup}, `position` ≥ 0.
  - Add `append_blind_queue(rows, root=None)`. It validates, refuses to write if that queue already has rows (append-only, written once per queue), and inserts into the `blind_queue` table of `reviews.duckdb` under the existing write lock.
  - Add `read_blind_queue(queue=None, root=None) -> pa.Table`, ordered by queue, then position.

  Update `specs/001-foundation/contracts/tables.md` (`label_role` values). Add tests to `tests/test_schema.py`:
  - an observation review row validates;
  - a second write of the same queue raises;
  - reads come back in position order;
  - `(queue, segment_id)` duplicates are rejected.

---

## Phase 2: Foundational (blocks all stories)

- [X] T005 Create `src/tpa_index/review.py` with these pure helpers. Every label and group comes from the `labels` and `pipeline` dicts passed in.
  - `checklist(labels, pipeline) -> dict` returns `{"primary": [names in pipeline.moments.primary_group], "groups": [(group, [model and staff label names]) for group in app.blind_groups if group != primary_group], "uncertain": app.uncertain_label}`.
    - Labels keep their `labels.yaml` order. The checklist includes model **and** staff labels (staff-only indicators are exactly what blind labeling needs). It excludes rules labels and never shows the null group (`pipeline.calibrate.null_group`).
    - Labels from `labels.yaml` that aren't in `blind_groups` are not shown, and that is not an error.
  - `role_of(label, labels, pipeline, chosen_primary=None) -> str` returns `primary` for the chosen primary label, `indicator` for labels in `app.indicator_groups`, and `secondary` otherwise.
  - `key_map(checklist) -> dict[str, str]` maps keys to labels:
    - `"1"`–`"9"` go to the primary labels, in order;
    - letters `a`–`z` go to every other checklist label, in order, skipping the reserved keys `u` (the uncertain label), `Enter`, `Backspace` and `Space`;
    - if there are more labels than keys, the extra labels get no key and are listed in a returned warning.
  - `latest_state(rows, mode=None) -> dict[(target_id, label), row]` keeps the newest row by `created_at`, then `review_id`.
- [X] T006 Write `tests/test_review.py` with tests for T005, using the fixtures `labels` and `pipeline`. Include the vocabulary-swap test: write an alternate `labels.yaml` to `tmp_path`. In it:
  - the `activity` group has only `juggling` and `drumming`;
  - there is a new group `venue` with `outdoor` and `indoor`;
  - there are new `indicator` labels;
  - the rules label is renamed `unsure`.

  Also write an alternate `pipeline.yaml` whose `app.blind_groups` is `[activity, venue, indicator]` and whose `app.uncertain_label` is `unsure`. Load both with `load_labels(path)` and `load_pipeline(path, labels=…)`, then assert:
  - `checklist()` lists exactly those labels, in order;
  - `key_map()` binds `1`→juggling and `2`→drumming;
  - `role_of()` follows `indicator_groups`;
  - no shipped label name appears anywhere in the result.
- [X] T007 Create `src/tpa_index/clips.py`:
  - `session_dir()`: `tempfile.mkdtemp(prefix="tpa-clips-")`. Assert it is not inside `repo_root()`. Remove leftover `tpa-clips-*` folders in the same temp root that are older than 12 h. Return the path.
  - `ClipCutter(session_dir, clip_height=480)` with these methods:
    - `.cut(video_row, start_s, end_s, pad_s=0) -> Path`: clip the bounds to `[0, video_row["duration_s"]]`, then run ffmpeg as in research §4 with `subprocess.run(…, timeout=30, capture_output=True)`. On failure, raise `RuntimeError(redact(stderr tail))`. Output goes to `session_dir/<uuid>.mp4`, and the previous clip from this cutter is deleted.
    - `.prefetch(video_row, start_s, end_s)`: cut in a single background thread and cache by `(video_id, start, end)`.
    - `.cleanup()`: remove the session folder.
  - `register_cleanup(cutter)`: `atexit`, plus SIGINT and SIGTERM handlers that call `cleanup()` and then the previous handler.
- [X] T008 Write `tests/test_clips.py` with synthetic clips only, using the `clips`/`clip_videos` fixtures and `DATA_DIR` set to the clips folder. Check that:
  - `cut()` on `t01` gives an H.264 file (`probe` the codec through PyAV) with duration ≈ end − start;
  - `pad_s=3` at `start=0` is clipped to the video start, and at the end is clipped to `duration_s`;
  - the previous clip is deleted on the next cut;
  - `prefetch()` then `cut()` reuses the prefetched file;
  - `cleanup()` removes the folder;
  - `session_dir()` is outside the repo;
  - a missing file raises `RuntimeError` with no data path in the message;
  - `git status --porcelain` is unchanged.

  Also add the hardcode guard to `tests/test_review.py`. It reads every `name` from the shipped `config/labels.yaml` labels and every group name, then asserts none appears as a quoted string literal (`'name'` or `"name"`) in `src/tpa_index/review.py`, `clips.py` or `app.py`. Each file is checked once it exists; files that don't exist yet are skipped.
- [X] T009 Add `--clips` to `tpa-index mock` in `src/tpa_index/cli.py` and `src/tpa_index/mock.py`. After `generate()`, for each mock video row, call `synthetic.make_clip(DATA_DIR/<filename>, 160, 90, seconds=ceil(duration_s), fps=4)`, creating parent folders. Add a test to `tests/test_cli.py`: after `mock --clips --n-videos 1`, the file exists and its probed duration covers the last mock segment's `end_s`.

**Checkpoint**: `uv run pytest` passes in full.

---

## Phase 3: User Story 1 — Blind-label a fixed random sample quickly (P1) 🎯 MVP

**Goal**: A seeded, saved main queue of eval segments, a fast keyboard checklist built from config, append-only blind rows, and resume.

**Independent test**: On mock data with `--clips`, label 5 segments with the keyboard, restart, and resume at segment 6 with the same order. No model value appears on the blind screen.

- [X] T010 [US1] In `src/tpa_index/review.py`, implement the following. The app config is `load_pipeline()["app"]`.
  - `build_main_queues(app_cfg, root=None) -> int`: if queue `main` exists, return its size. Otherwise:
    - take eval videos from `read_official("videos")` (falling back to `stage="segment"` when it's empty), keeping `split == "eval"`;
    - take segments from `read_official("segments")` with the same fallback, keeping eval videos only;
    - sort by `segment_id`, apply `np.random.default_rng(seed).permutation`, and take `sample_size`;
    - append queue `main`, plus queue `overlap` (its first `overlap_size` items, renumbered 0…).

    If there are no eval segments, raise `ConfigError("no eval-split segments in official runs")`.
  - `load_queue(queue, root=None) -> list[dict]`.
  - `next_position(queue, reviewer, root=None)`: the first position whose segment has no `mode=blind` row from that reviewer, or `None`.
  - `progress(queue, reviewer, root=None) -> (done, total)`.
  - `queue_missing(queue, root=None) -> list[str]`: queue segment IDs that are no longer in the official segments, reported and never replaced.
- [X] T011 [US1] In `src/tpa_index/review.py`, add `blind_rows(segment, primary, chosen, uncertain, reviewer, labels, pipeline, previous=None) -> list[dict]`:
  - Require `primary` or `uncertain`, else raise `ValueError("choose a primary activity or the uncertain option")`.
  - Write one `add` row per chosen label (the primary, each checked label, and the uncertain label when set). The role comes from `role_of`, `new_label` equals the label, `source="staff"`, `mode="blind"`, `target_type="segment"`, and `guideline_version` comes from labels.
  - Write one `reject` row for each label in `previous` (this reviewer's latest blind adds on the segment) that is no longer chosen.

  Also add `skip_row(segment, reviewer, labels, pipeline)`, an observation row with `label=app.skip_observation`. Add tests to `tests/test_review.py`:
  - roles, actions and `new_label` are correct;
  - saving with neither primary nor uncertain raises;
  - re-labeling with one label unticked writes exactly one reject;
  - the rows pass `validate("reviews", …)`.
- [X] T012 [US1] Add queue tests to `tests/test_review.py`. Make mock data with an official runs file listing `mock-seed0`, through the patched `schema.OFFICIAL_RUNS`. Check that:
  - two `build_main_queues` calls give the same queue, and the second call doesn't write;
  - a fresh database with the same official runs and seed gives an identical order;
  - all segments are from eval videos;
  - `sample_size` larger than the number of eval segments gives all of them;
  - overlap is the first 60;
  - `next_position` and `progress` follow the saved rows for one reviewer and ignore another reviewer's rows;
  - with only a segment-stage official run, the queue still builds (the fallback);
  - removing a run from the official list after building leaves the queue unchanged, and `queue_missing` reports the gone IDs.
- [X] T013 [US1] Create `src/tpa_index/app.py`:
  - Before `import gradio`, set `os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"` and `os.environ["GRADIO_TEMP_DIR"]` to a `gradio` subfolder of the cutter's session folder.
  - `THEME`: `gr.themes.Default(font=["system-ui", "sans-serif"], font_mono=["ui-monospace", "monospace"])`.
  - `LIGHT_JS`: the force-light snippet (research §5).
  - `build_app(root=None, cutter=None, labels=None, pipeline=None) -> gr.Blocks`: `gr.Blocks(analytics_enabled=False, title="TPA review")`, with:
    - a header: the reviewer `gr.Textbox` with the placeholder "staff first name or role", and a **Reload data** button;
    - `gr.Tab("Blind labeling")`: queue picker (`gr.Radio`), progress `gr.Markdown`, `gr.Video(autoplay=True, loop=True)`, the primary `gr.Radio`, one `gr.CheckboxGroup` per checklist group, the uncertain `gr.Checkbox` and the buttons;
    - every label component is built from `review.checklist()` and shows its key hint (`"singing [1]"`-style choice text from `key_map`), with `elem_id`s from `key_map` for the shortcuts;
    - `gr.Tab("Review")` stays a placeholder until T024.
  - `launch(port=None)`: `build_app().launch(server_name="127.0.0.1", share=False, enable_monitoring=False, inbrowser=False, allowed_paths=[str(session)], theme=THEME, js=LIGHT_JS + KEYS_JS, show_api=False)`.
- [X] T014 [US1] Wire blind mode in `src/tpa_index/app.py`:
  - On reviewer and queue selection: call `build_main_queues`, jump to `next_position`, show the clip (`cutter.cut`), prefetch the next item, and show `progress`.
  - **Save & next**: `blind_rows` → `append_reviews` → next item. A `ValueError` shows as a `gr.Warning`, and nothing is saved.
  - **Back**: the previous position, with that reviewer's previous choices pre-filled from `latest_state` (blind rows only).
  - **Replay**: re-sends the same clip path.
  - **Skip: clip unavailable**: `skip_row` → next.
  - A cut error shows a redacted `gr.Warning` and enables the Skip button.
  - No blind-tab function may read predictions or moments.
- [X] T015 [US1] Add `KEYS_JS` to `src/tpa_index/app.py`: one `keydown` listener, ignored when the focus is in an `input`/`textarea`.
  - The digit and letter keys click the matching choice or checkbox, found by `elem_id` and the choice value from the `key_map` passed into the page as JSON (no label strings written into the JS source).
  - `u` toggles the uncertain checkbox.
  - `Enter` clicks Save & next, `Backspace` clicks Back, `Space` replays.

  Add `tpa-index app [--port N]` to `src/tpa_index/cli.py`:
  - `install_redaction`, then `check_guard`, then load configs;
  - if the port is already answering on 127.0.0.1, exit 3 with "the review app is already running on port N";
  - otherwise `app.launch(port)`, printing only `http://127.0.0.1:<port>`.
- [X] T016 [US1] Write `tests/test_app.py` on mock data with `--clips`-style synthetic videos and a scratch official file. Check that:
  - `build_app()` builds without error;
  - walking the Blocks config (`demo.config["components"]`) finds the blind tab's checklist choices equal to `review.checklist()`, and no component label or value in the blind tab contains "AI", "score", "confidence" or a model name from `pipeline.models`;
  - with `schema.read_official` monkeypatched to raise for `predictions` and `moments`, a full blind flow (select a queue, save 3 segments, go back, re-save, skip one) calls the blind handlers directly and succeeds, so blind mode never touches model output;
  - the saved rows match `blind_rows` expectations;
  - with the alternate `labels.yaml` and `pipeline.yaml` from T006 passed into `build_app(labels=…, pipeline=…)`, the blind checklist components show exactly the alternate labels.

**Checkpoint (code)**: blind mode works through direct handler calls. `uv run pytest` passes.

---

## Phase 4: User Story 5 — Keep footage and people safe (P1)

**Goal**: Loopback only, no outbound requests, no lasting footage copies, redacted errors, and the staff-only reviewer prompt.

**Independent test**: Launching on loopback with sharing off leaves no external sockets, and the session folder is gone after shutdown.

- [X] T017 [US5] Add these to `tests/test_app.py`:
  - `demo.launch(server_name="127.0.0.1", prevent_thread_lock=True, …)` with the same settings as `app.launch`: the server address is `127.0.0.1`, `demo.share` is falsy, and an HTTP GET of the page from `127.0.0.1` returns 200.
  - After a blind save driven through the handlers, `psutil.Process().net_connections(kind="inet")` lists no connection whose remote address is outside `127.0.0.0/8` or `::1`.
  - After `demo.close()` and `cutter.cleanup()`, the session folder and the Gradio temp subfolder no longer exist.
  - The reviewer textbox placeholder says "staff first name or role".
  - A grep test: `share=True` does not appear anywhere in `src/`.
- [X] T018 [US5] In `src/tpa_index/app.py`, make every user-visible error message pass through `redact()`, including Gradio warnings from cut errors and load errors. Add a test that forces a cut error on a missing synthetic file and checks that the warning text has no `DATA_DIR` path or filename.

---

## Phase 5: User Story 2 — Second labeler and rare-label top-up (P1)

**Goal**: The overlap queue for the second labeler, and a model-picked, shuffled, capped and tagged top-up queue that is shown blind and hidden until official predict runs cover the eval videos.

**Independent test**: The second reviewer sees exactly the 60 overlap segments. The top-up has at most 20 segments per rare label, never repeats a labeled segment, shows no target label, and is hidden with a message when there are no predict runs.

- [X] T019 [US2] In `src/tpa_index/review.py`:
  - `topup_available(root=None) -> (bool, reason)`: True only if official predict runs (`load_official()["predict"]`) cover every eval video in the main queue. The reason names what's missing ("official predict runs are needed for the eval videos").
  - `build_topup_queue(app_cfg, labels, pipeline, root=None) -> int`: if a top-up queue exists, return its size. Otherwise:
    - rare labels are the model labels in `app.blind_groups` (excluding rules labels) with fewer than `rare_min_positives` blind positives, where a positive is a latest blind row with action add, from any reviewer, on main-queue segments;
    - for each rare label, take up to `topup_per_label` eval segments not in `main`, `overlap` or an earlier top-up, ranked by that label's official main-model score (`read_official("predictions")`, non-rules model rows), highest first;
    - remove duplicates (the first label wins), shuffle with `default_rng(seed + 1)`, and append queue `topup` with `target_label`.
- [X] T020 [US2] Add tests to `tests/test_review.py`, building official predict runs from mock data with crafted scores:
  - no predict runs means `topup_available` is false with the reason, and building raises;
  - with predict runs: at most 20 per rare label, no segment from main or overlap, a common label (many blind positives) gets none, the order differs from the score order but is stable across rebuilds, and every row has `target_label`;
  - a second build writes nothing.
- [X] T021 [US2] In `src/tpa_index/app.py`, add the queue picker entries:
  - **Main sample**.
  - **Overlap set (second labeler)**.
  - **Top-up**: shown only when `topup_available()` is true, otherwise a `gr.Markdown` shows the reason. Building happens on first selection.

  Top-up items use exactly the same blind screen; `target_label` never reaches any component. Add tests to `tests/test_app.py`:
  - a second reviewer on the overlap queue sees 60 items, in overlap order;
  - the top-up option is hidden before predict runs and shown after;
  - no top-up `target_label` value appears in any blind component value while a top-up item is shown.

---

## Phase 6: Checkpoint 1 — blind mode end to end (Saturday-morning target)

- [X] T022 **Checkpoint 1 (Mac, mock data only)**:
  1. Export `DATA_DIR`, `OUTPUT_DIR`, `TPA_LOCAL_VIDEOS` (a missing file), `TPA_THRESHOLDS_FILE` and `TPA_OFFICIAL_RUNS` to scratch paths. Never touch `.env`.
  2. Run `uv run tpa-index mock --seed 0 --clips`, and write `predict: [mock-seed0]` to the scratch official file.
  3. Run `uv run tpa-index app --port 7861` in the background.
  4. With the Playwright browser tool, against `http://127.0.0.1:7861` only:
     - enter the reviewer `mock-staff-a` and choose Main sample;
     - label 5 segments using only the keyboard (a digit, a letter, `Enter`), recording the time per segment;
     - stop the app and start it again, and confirm it resumes at segment 6 with the same clip order (compare segment IDs through `review.load_queue`);
     - as `mock-staff-b`, open the Overlap set and label 2;
     - confirm the Top-up option is visible (predict runs exist) and that its screen shows no target label;
     - record **all** network requests during the session and assert every host is `127.0.0.1`;
     - take one screenshot and confirm the light theme.
  5. Stop the app. Confirm that the session clip folder is gone, no `tpa-clips-*` folder remains, and `git status --porcelain` is unchanged. Delete the scratch folders.
  6. Report: the seconds per segment, the rows written (counts by action, role and mode), the request hosts seen, and anything that blocked the flow.

---

## Phase 7: User Story 3 — Review AI moments on a timeline (P2)

**Goal**: A per-video moment timeline with filters built from config, padded playback, "AI suggestion, unvalidated" markers, and eval videos hidden until the blind sets are done.

**Independent test**: On mock data, filter to needs-review, play a moment with its padding, and see the unvalidated markers. Eval videos are hidden until unlock.

- [X] T023 [US3] In `src/tpa_index/review.py`:
  - `ai_labels(moment) -> list[(label, role)]`: `primary_label`→primary, each `secondary_labels`→secondary, each `indicators`→indicator.
  - `moment_status(moment, state)`, following research §9.
  - `filter_moments(moments, state, label=None, needs_review=None, status=None)`.
  - `eval_unlocked(root=None) -> (bool, reason)`: every main segment has at least one blind row, and every overlap segment has blind rows from at least 2 distinct reviewers.
  - `review_videos(root=None) -> list[dict]`: the official videos, minus eval videos while locked.

  Add tests to `tests/test_review.py`:
  - each status case;
  - combined filters;
  - unlock is false until both conditions hold, including the case where the overlap is labeled by the same reviewer twice, which must still be locked;
  - eval videos are excluded while locked;
  - labels in a moment that are absent from `labels.yaml` still show and are flagged as retired.
- [X] T024 [US3] Build the Review tab in `src/tpa_index/app.py`:
  - a video picker from `review_videos` (with the lock reason shown when eval videos are hidden);
  - filters:
    - the label dropdown, with choices from `labels.yaml` (every model label and observation type, whatever the groups);
    - needs-review (any, yes or no);
    - status;
    - a show-idle toggle, off by default (when on, idle segments are listed as rows of type "idle");
  - a moments `gr.Dataframe` with columns time, primary, status, needs_review and confidence, sorted by time;
  - on selection, `cutter.cut(video, start - padding_s, end + padding_s)`, plus a per-label panel showing label, role, confidence and the badge "AI suggestion, unvalidated" (or the latest state with reviewer and time).
- [X] T025 [US3] Add tests to `tests/test_app.py`:
  - review handlers on mock data return moments sorted by time;
  - a filter change narrows the list;
  - the clip request covers start − 3 to end + 3, clipped to the video bounds;
  - unreviewed labels carry the badge text;
  - with the alternate `labels.yaml` from T006, the label filter choices are exactly the alternate vocabulary plus the observation types.

---

## Phase 8: User Story 4 — Accept, edit, reject, defer or add (P2)

**Goal**: Every review action is one append-only row with mode review. Observations come from the config list, with the no-names, no-judgments reminder.

**Independent test**: Perform all five actions and an observation on one moment, and get one row each, with the moment state following the latest action.

- [X] T026 [US4] In `src/tpa_index/review.py`, add:
  - `review_row(moment, label, role, action, reviewer, labels, new_label=None, note=None, source="ai")`: the target is the moment and `mode="review"`. `edit` requires `new_label`; accept, reject and defer forbid it. `source="staff"` and `action="add"` are used for staff-added labels.
  - `observation_row(moment, obs_type, reviewer, labels, note=None)`: raises if `obs_type` is not in `observation_types(labels)`.

  Add tests: every action validates against the reviews schema, edit without `new_label` raises, an unknown observation type raises, and a sequence of accept → defer → edit leaves the latest state as edit.
- [X] T027 [US4] Wire the review actions in `src/tpa_index/app.py`:
  - per AI label: Accept / Reject / Defer buttons and an Edit dropdown (other labels from the same group, read from config);
  - add-label: a dropdown of every model label, with its role from `role_of`;
  - add-observation: a dropdown from `observation_types`;
  - a note `gr.Textbox` with a visible `gr.Markdown` reminder: "No names. No judgments about a person.";
  - each action calls `append_reviews`, then refreshes the label panel, the moment status and a read-only history table (time, reviewer, label, action and note).

  Add tests to `tests/test_app.py`:
  - driving each action through the handlers writes exactly one row;
  - the panel badge changes from unvalidated to the action;
  - the reminder text is present;
  - the observation dropdown choices equal `observation_types(labels)`.

---

## Phase 9: Polish and Final Checkpoint

- [X] T028 Update `README.md` with a "Review app" section, from `quickstart.md`: launching, blind labeling keys, the overlap set and top-up, review actions, the local-only guarantee and clip cleanup. Cross-check `src/` against `contracts/interfaces.md`, `contracts/config.md` and `data-model.md` (signatures, config keys, the `blind_queue` columns and the review-row table), fixing drift on whichever side is wrong. Rerun the hardcode guard from T008 over all three modules, which must now exist.
- [X] T029 **Final checkpoint (Mac, mock data only)**:
  1. Run `uv run pytest`, then `uv run pytest -m slow`, and record the counts.
  2. Repeat T022 steps 1–3.
  3. With the Playwright tool:
     - blind-label the whole overlap set for two reviewers, and the main sample through the handlers (fast path), so that review unlocks;
     - in Review, confirm eval videos appear only after the unlock;
     - filter to needs-review and play one moment, checking the padding;
     - accept one label, edit one, reject one, defer one, add a label and add an observation with a note;
     - confirm the badges and the history;
     - record all network requests (every host must be `127.0.0.1`);
     - take a light-theme screenshot.
  4. Stop the app. Confirm no clip folders remain and `git status` is unchanged, then delete the scratch folders.
  5. Report: the test counts, the rows by mode, action and role, the request hosts, and the seconds per blind segment from T022.

---

## Dependencies & Execution Order

- **Setup (T001–T004)**, then **Foundational (T005–T009)**, which blocks everything. T008's hardcode guard applies from the moment each module exists.
- **US1 (T010–T016)** needs Foundational.
- **US5 (T017–T018)** needs the US1 app.
- **US2 (T019–T021)** needs US1's queues and app.
- **Checkpoint 1 (T022)** needs US1, US5 and US2. It is the Saturday-morning gate.
- **US3 (T023–T025)** needs Foundational. It comes after the checkpoint by priority, and its unlock rule depends on US1/US2 queues.
- **US4 (T026–T027)** needs US3.
- **Polish (T028–T029)** needs everything else.

## Parallel Opportunities

None are marked. One developer works through the tasks in ID order.

## Implementation Strategy

1. **Saturday-morning MVP**: T001–T022. Blind labeling (main sample, overlap set and top-up) runs locally on the Mac, with safety verified at Checkpoint 1. Staff can start producing ground truth.
2. **Review**: T023–T027.
3. **Wrap-up**: T028–T029.
