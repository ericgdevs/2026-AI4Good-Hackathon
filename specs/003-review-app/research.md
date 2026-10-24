# Research: Review App

Checked on 2026-10-24 against the locked environment: Gradio 6.30.0 and Homebrew ffmpeg with `libx264`, `h264_videotoolbox` and `aac`.

## 1. App framework settings (Gradio 6)

- In Gradio 6, `theme`, `css`, `js` and `head` go to **`launch()`**, not to `gr.Blocks()`. `launch` also takes `server_name`, `share`, `allowed_paths` and `enable_monitoring`, and `gr.Blocks` takes `analytics_enabled`.
- **Decision**:
  - Set `os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"` before importing gradio, and pass `gr.Blocks(analytics_enabled=False)`.
  - Launch with `launch(server_name="127.0.0.1", share=False, enable_monitoring=False, allowed_paths=[clip_dir], inbrowser=False)`.
  - The port comes from config (default 7860) with a fallback to the next free port.
  - `share` is hard-coded to `False`. There is no CLI flag or config key for it.

## 2. No outbound calls (FR-018)

- **Finding**:
  - The default theme's CSS references no external URL (checked: only the SVG namespace), and its fonts are not Google-font objects.
  - The frontend bundle does contain external hosts (Google Fonts, unpkg, jsdelivr, Hugging Face). Most belong to components we don't use: maps, 3D models, plots and docs links.
  - A static scan cannot prove that no request goes out.
- **Decision**: Make the theme use system fonts only (`font=["system-ui", "sans-serif"]`, `font_mono=["ui-monospace", "monospace"]`) and verify at runtime in two layers:
  1. A test: launch the app with `prevent_thread_lock`, use it, then list the process's open sockets with `psutil` (already a transitive dependency) or `lsof -p`. Assert there are no non-loopback remote addresses.
  2. A checkpoint step: load the page in a real browser with the Playwright tool, record every network request during a blind save and a review action, and assert every host is `127.0.0.1`.
- **Alternatives**: An offline firewall rule. Rejected: not portable to TPA machines and not testable in CI.

## 3. Where footage copies live (FR-019)

- **Finding**: `gr.Video` copies served files into Gradio's cache (`GRADIO_TEMP_DIR`, which defaults to the system temp folder under `gradio/`). Without control, footage clips would pile up there after the app exits.
- **Decision**: At startup, create one session folder, `tempfile.mkdtemp(prefix="tpa-clips-")`, which is outside the repo (asserted). Set `GRADIO_TEMP_DIR` to a subfolder of it before importing gradio.
  - Cut clips go into the session folder.
  - Each newly shown clip deletes the previous one.
  - `atexit` plus SIGINT and SIGTERM handlers remove the whole folder.
  - On startup, any leftover `tpa-clips-*` folders older than 12 h are removed (crash recovery).
- **Test**: after a session, the folder is gone and nothing new exists in the repo (`git status` unchanged).

## 4. Cutting clips

- **Decision**: `ffmpeg -nostdin -loglevel error -ss {start} -i {src} -t {dur} -vf scale=-2:{clip_height} -c:v libx264 -preset veryfast -crf 26 -c:a aac -movflags +faststart {out}`.
  - Input seeking with a re-encode is frame-accurate, and the output is H.264/AAC, which every browser plays. The synthetic test clips are `mpeg4`, which browsers can't play, so they get re-encoded too.
  - `clip_height` defaults to 480, for speed.
  - The cut runs through `subprocess.run(..., timeout=30)`. Errors are redacted (001 `redact`).
  - **Prefetch**: in blind mode, the next queue item's clip is cut in a background thread while the current one is labeled, to keep about 25 s per segment (SC-001).
- **Video path**: `data_dir() / videos.filename`, from the official `videos` table (written from `videos.local.yaml` at predict or segment time). The app never reads `videos.local.yaml` directly.
- **Alternatives**:
  - PyAV re-encode: more code and slower to write.
  - Serving the original file with a media fragment (`#t=`): this exposes the whole video to the browser cache and isn't frame-exact.

## 5. Light theme forced (FR-022)

- **Decision**: Pass `launch(js=...)` with a snippet that, on load, adds `?__theme=light` if the URL lacks it and reloads (Gradio honours `__theme`). Also remove any `dark` class from the page body.
- **Test**: the page HTML has no `dark` class, and the checkpoint takes a screenshot in light mode.

## 6. Keyboard shortcuts (FR-009)

- **Decision**: `launch(js=...)` installs one `keydown` handler, ignored while typing in a text field. The key map is generated from `labels.yaml` at startup, so new labels get keys automatically:
  - `1`–`9`: the primary group's labels (from `pipeline.moments.primary_group`), in config order.
  - Letters: every other checklist item in config order, skipping the reserved keys.
  - Reserved: `U` for `app.uncertain_label`, `Enter` for save-and-next, `Backspace` for back, `Space` for replay.

  Each label's key is shown next to it. A handler "presses" the matching component through its `elem_id`.
- **Alternatives**: Gradio's built-in `.key_up` events. Rejected: they're per-component, and focus would have to stay inside one box.

## 7. Blind queues (FR-006, FR-007, FR-012, FR-013)

- **Storage**: a new table, `blind_queue`, in the same local `reviews.duckdb`, which is outside the run folders and never synced. Columns:
  - `queue`: `main`, `overlap` or `topup`
  - `position` (int)
  - `segment_id`, `video_id`, `start_s`, `end_s`
  - `target_label`: top-up only, never shown
  - `seed` and `created_at`

  It's written once per queue and never modified.
- **Main sample**:
  - Eval videos come from the official `videos` table (`split == "eval"`).
  - Eval segments come from `read_official("segments")`, falling back to `read_official("segments", stage="segment")` when no predict runs are official yet, because blind mode needs only segments.
  - Sort by `segment_id`, apply `np.random.default_rng(seed).permutation`, and take `sample_size` (300, or all of them if there are fewer).
- **Overlap**: the first `overlap_size` (60) entries of `main`, copied as queue `overlap` with positions 0–59.
- **Top-up**:
  - Built on request, and only if official predict runs cover every eval video; otherwise it's hidden with a message.
  - A label is rare when it has fewer than `rare_min_positives` (10) blind positives (latest blind row with action add, any reviewer) in the main sample. Rules labels are excluded; `uncertain_mixed` is not a target.
  - For each rare label, take up to `topup_per_label` (20) eval segments not in `main`, `overlap` or any earlier top-up, with the highest official main-model score for that label.
  - Remove duplicates (first label wins), shuffle with the seed, and store them with `target_label`.
- **Resume**: the first position in the reviewer's queue that has no blind row from that reviewer for that segment.

## 8. Blind save rows (FR-010)

- One row per chosen label: `target_type=segment`, `target_id=segment_id`, `action=add`, `new_label=label` (the 001 rule requires `new_label` for add), `source=staff`, `mode=blind`.
- Label role mapping, all from config (no label or group names in code):
  - the chosen label of `pipeline.moments.primary_group` → `primary`;
  - labels in `app.indicator_groups` → `indicator`;
  - everything else, including `app.uncertain_label` → `secondary`.
- **Re-label**: compare with that reviewer's latest blind state on the segment, and write `action=reject` (with `new_label` null) for labels that were unticked.
- **Skip a missing file**: an observation row, `label_role=observation`, `label=<app.skip_observation>`, `action=add`. It counts as handled for resume.

## 9. Review state and filters (FR-014 to FR-017)

- **AI labels of a moment**: `primary_label` (role primary), `secondary_labels` (secondary) and `indicators` (indicator).
- **Current state** of (`moment_id`, `label`): the latest `mode=review` row, by `created_at` then `review_id`.
- **Moment status**:
  - `unreviewed`: no AI label has a row.
  - `deferred`: any latest action is `defer`.
  - `done`: every AI label's latest action is accept, edit or reject.
  - `partly`: anything else.
- **Eval unlock** (FR-014a): every `main` segment has at least one blind row, and every `overlap` segment has blind rows from at least 2 distinct reviewers.
- **Idle**: segments with `is_idle` are hidden from the timeline unless the toggle is on. Moments never contain idle segments (002).
- **Data loading**: tables are read once at startup and on a "Reload" button. Review state is recomputed from the reviews table after each save, which is cheap at these sizes (thousands of rows).

## 10. Observations (FR-017, FR-017a)

- `labels.yaml` gets `observation_types: [{name, description}]`, validated as unique names in snake_case. It ships with `staging_issue`, `audio_problem`, `camera_problem`, `off_task`, `worth_showcasing` and `clip_unavailable`.
- Adding the list bumps the guideline version to `0.3.0`. The rule that adding a type bumps the version is documented in the file header, because it can't be enforced in code without a history.
- `label_role` gains `observation` in the 001 `reviews` schema (`RULES`). Older rows stay valid.

## 11. Concurrency

- One app process serves every browser tab on the machine. `append_reviews` already serializes writes with an in-process lock.
- A second app process on the same database fails DuckDB's write lock. The CLI detects an already-running instance on the port and refuses to start, with a clear message.

## 12. Mock clips for end-to-end use without footage

- Move `make_clip` from `tests/clips.py` to `src/tpa_index/synthetic.py`; tests import it from there.
- Add `tpa-index mock --clips`, which writes a synthetic video for each mock video (`DATA_DIR/mock/<name>.mp4`) at 4 fps and 160×90, with the mock video's duration. The app then runs fully on mock data.
- Mock `videos.duration_s` comes from the generator, so the files cover every mock segment.
