# Contract: Python API, CLI and UI (003)

This extends the 001 and 002 contracts, which still hold.

## Python (`tpa_index`)

```python
# review.py: pure logic, no UI, no ffmpeg (fully unit-tested)
load_queue(queue, root=None) -> list[dict]                        # [] if not built yet
build_main_queues(app_cfg, root=None) -> int                      # main + overlap; no-op if they exist; returns size
topup_available(root=None) -> tuple[bool, str]                    # (ok, reason shown when hidden)
build_topup_queue(app_cfg, labels, pipeline, root=None) -> int    # appends a 'topup' queue once; returns count
next_position(queue, reviewer, root=None) -> int | None           # resume point; None when finished
progress(queue, reviewer, root=None) -> tuple[int, int]           # (done, total)
checklist(labels, pipeline) -> dict                              # primary / groups / uncertain, from config (model + staff labels)
key_map(checklist) -> tuple[dict[str, str], list[str]]          # (key -> label, unkeyed); keys 1-9, 0, a-z without u
role_of(label, labels, pipeline, chosen_primary=None) -> str
blind_rows(segment, primary, chosen, uncertain, reviewer, labels, pipeline, previous=None) -> list[dict]
#   chosen: label -> role; adds reject rows for labels in `previous` that are no longer chosen
latest_state(rows, mode) -> dict[tuple[str, str], dict]          # (target_id, label) -> latest row
ai_labels(moment) -> list[tuple[str, str]]                       # (label, role): primary, secondaries, indicators
moment_status(moment, state) -> str                               # unreviewed | partly | done | deferred
eval_unlocked(root=None) -> tuple[bool, str]
review_videos(root=None) -> list[dict]                            # official videos; eval hidden while locked
badge(row) -> str                                                 # "AI suggestion, unvalidated" or the latest state
label_panel(moment, state, labels, confidences, thresholds) -> list[dict]
#   primary first, then each role by strength (score / threshold); primary confidence = moment confidence;
#   other labels: mean raw score over the moment's segments; labels missing from labels.yaml flagged retired
review_row(moment, label, role, action, reviewer, labels, new_label=None, note=None, source="ai") -> dict
observation_row(moment, obs_type, reviewer, labels, note=None) -> dict
filter_moments(moments, state, label=None, needs_review=None, status=None) -> list[dict]

# clips.py: the only module that touches footage
ClipCutter(session_dir, clip_height=480)        # .cut(video_row, start_s, end_s, pad_s=0) -> Path; .prefetch(...); .cleanup()
session_dir() -> Path                           # mkdtemp('tpa-clips-'), asserted outside the repo; old leftovers purged

# synthetic.py: moved from tests/clips.py
make_clip(path, w, h, seconds=20, fps=10, audio=True, rate=16000, static=False) -> Path

# app.py: Gradio UI
build_app(root=None, cutter=None, labels=None, pipeline=None) -> gr.Blocks  # no launch; testable
#   demo.blind / demo.review: BlindController / ReviewController, plain-method handlers for tests
launch(port=None) -> None                       # 127.0.0.1, share=False, analytics and monitoring off, light theme forced
```

## CLI

| Command | What it does | Exit codes |
|---|---|---|
| `tpa-index app [--port N]` | starts the local app on 127.0.0.1 (prints the local URL only) | 0; 2 guard failure; 3 config error or app already running |
| `tpa-index mock [--clips]` | `--clips` also writes synthetic videos for the mock videos into `DATA_DIR/mock/` | as in 001 |

There is no `--share` flag and no host flag.

## UI

**Header**, on every screen:
- reviewer box: required before any action; hint text "staff first name or role";
- mode tabs: **Blind labeling** and **Review**;
- a **Reload data** button.

**Blind labeling**:
- queue picker: Main sample, Overlap set (second labeler) and Top-up (hidden with its reason when unavailable);
- the clip player, which autoplays and loops the 8 s clip;
- the progress counter, "done / total";
- the checklist, generated from `labels.yaml`: primary activity as a radio group (keys 1–9), every other model label grouped by group as checkboxes (letter keys), and uncertain/mixed (`U`);
- buttons: **Save & next** (`Enter`), **Back** (`Backspace`), **Replay** (`Space`), and **Skip: clip unavailable**.

Nothing derived from predictions appears on this screen.

**Review**:
- video picker: eval videos are hidden until unlocked, with the reason shown;
- filters: label, needs-review, status (unreviewed, partly, done, deferred), and a show-idle toggle;
- a moment timeline table: time, primary label, status, needs-review and confidence;
- a player for the moment with padding;
- for each AI label: name, role, confidence and the badge "AI suggestion, unvalidated" (or the latest review state), in a label table; pick one label, then Accept / Edit (dropdown of the same group) / Reject / Defer;
- add-label (vocabulary dropdown) and add-observation (observation-type dropdown);
- a note box with the visible reminder "No names. No judgments about a person.";
- the moment's review history, read-only.
