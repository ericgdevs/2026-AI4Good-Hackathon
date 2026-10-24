# Implementation Plan: Review App

**Branch**: `003-review-app` | **Date**: 2026-10-24 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/003-review-app/spec.md`

## Summary

A local Gradio app (`tpa-index app`) with two tabs.

**Blind labeling**:
- Saved, seeded queues: a 300-segment main sample of eval segments, the first 60 as the overlap set, and a model-picked top-up queue capped at 20 per rare label, shuffled and tagged `topup`.
- A keyboard-driven checklist built from `labels.yaml`, with no model output on screen.
- Each label is saved as a `reviews` row (mode blind).

**Review**:
- A per-video moment timeline with filters (label, needs-review, status, idle).
- Padded playback, and accept/edit/reject/defer/add/observation actions, each saved as a row (mode review).
- Eval videos are hidden until the blind sets are done.

**How footage is handled**: footage is cut on demand with ffmpeg into one session temp folder outside the repo. Gradio's own cache is pointed into that folder, and the whole folder is deleted on exit. The app binds `127.0.0.1` only, with sharing, analytics and monitoring off, system fonts only and a forced light theme.

## Technical Context

**Language/Version**: Python 3.12 and uv. The package is `tpa_index` (001/002).

**Primary Dependencies**: Gradio 6.30 (already locked), ffmpeg (system, macOS Homebrew, with libx264 and aac), and the 001/002 stack. No new Python dependencies. `psutil` (already locked) is used by one test.

**Storage**: the existing local `reviews.duckdb`, plus a new `blind_queue` table in the same file. Inputs are read through `read_official`.

**Testing**: pytest, with unit tests for `review.py` and `clips.py` on mock data and synthetic clips, an app build-and-launch test on loopback with a socket check, and a browser network check at the checkpoint (Playwright tool, no new dependency).

**Target Platform**: macOS laptop with footage (the TPA Mac), in the browser on the same machine.

**Project Type**: local web UI on top of the library and CLI.

**Performance Goals**:
- A blind segment takes 25 s or less (median) to label, with the next clip prefetched.
- A moment clip starts within 3 s.

**Constraints**:
- Loopback only, with 0 outbound requests.
- No footage copies outside the session temp folder.
- No model output in blind mode.
- Append-only rows.
- Light theme.

**Scale/Scope**: 1–3 staff on one machine, about 400 blind segments in total, thousands of moments, and tens of thousands of review rows at most.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| # | Principle | Status | How |
|---|---|---|---|
| I | Data local only | PASS | Loopback bind, `share=False` hard-coded, and no network calls (verified at runtime). Clips live in a session temp folder outside the repo and are deleted on exit. Gradio's cache is redirected there. The reviews database is never synced. |
| II | Efficient, measured | PASS | No model runs in the app. Clip cuts are small (480p); prefetch keeps labeling fast. |
| III | Assistants never see data | PASS | Tests and checkpoints use mock data and synthetic clips only. Errors are redacted. |
| IV | No identity | PASS | The reviewer is a staff name or role. No person-level rows, faces or tracks. |
| V | No judgments | PASS | Observations come from a fixed type list, with a visible no-names, no-judgments reminder. No rankings or individual scores on any screen. |
| VI | Humans verify | PASS | Every AI label is marked "AI suggestion, unvalidated" until reviewed. Every action is a logged row with reviewer, guideline version and timestamp. |
| VII | Contracts first | PASS | The `observation` role, `observation_types` and the `blind_queue` table are recorded in contracts/data-model. |
| VIII | Measure everything shown | PASS | Blind ground truth comes from a seeded random sample. Top-up is tagged and precision-only. Eval videos are hidden from review until the blind sets are done. |
| IX | Simplest thing | PASS | One Gradio app, logic in plain functions, ffmpeg subprocess calls, and no new dependencies. |
| X | Light mode | PASS | Light theme forced through `__theme=light`, with system fonts. |
| XI | Config over code | PASS | Sample size, seed, overlap, top-up, padding, clip size, observation types and the checklist all come from config. |
| XII | One app, two modes | PASS | One app serves both modes over the same tables. The 006 demo view builds on it. |
| XIII | Deliverables | PASS | This is the review interface deliverable. |

**Post-design re-check**: PASS.

## Project Structure

### Documentation (this feature)

```text
specs/003-review-app/
├── plan.md  research.md  data-model.md  quickstart.md
├── contracts/{interfaces.md, config.md}
└── tasks.md            # /speckit-tasks
```

### Source Code

```text
config/labels.yaml           # + observation_types; guideline_version 0.4.0
config/pipeline.yaml         # + app: section
src/tpa_index/
├── schema.py                # label_role += observation; blind_queue schema + append/read helpers
├── config.py                # validate observation_types and the app: section
├── synthetic.py             # make_clip (moved from tests/clips.py)
├── mock.py                  # --clips: synthetic videos for the mock videos
├── review.py                # queues, blind rows, latest state, status, filters, eval unlock (pure)
├── clips.py                 # session temp dir, ffmpeg cut, prefetch, cleanup
├── app.py                   # Gradio Blocks + launch settings (loopback, no share, light, keys)
└── cli.py                   # `app`, `mock --clips`
tests/
├── clips.py                 # re-exports synthetic.make_clip
├── test_review.py           # sampling stability, overlap, top-up (cap, shuffle, hidden), resume, rows, status, unlock
├── test_clips.py            # cut synthetic clip (H.264), padding clipped to bounds, prefetch, cleanup, outside repo
└── test_app.py              # build_app on mock data; blind tab has no model fields; launch on loopback, no share, no external sockets
```

**Structure Decision**: Logic is in `review.py`, footage handling in `clips.py`, and UI in `app.py`. Everything testable sits outside Gradio, and only `clips.py` ever touches footage.

## Complexity Tracking

No violations.
