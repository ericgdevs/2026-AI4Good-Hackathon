# Data Model: Review App

## Changes to existing contracts

| Where | Change |
|---|---|
| 001 `reviews.label_role` | adds `observation` (allowed set: primary, secondary, indicator, observation) |
| `config/labels.yaml` | adds `observation_types: [{name, description}]`; `guideline_version` moves to `0.3.0` |
| `config/pipeline.yaml` | adds the `app:` section (see contracts/config.md) |

No pipeline table changes.

## New table: `blind_queue` (in the local `reviews.duckdb`, never synced)

| column | type | nullable | notes |
|---|---|---|---|
| queue | string | no | `main`, `overlap` or `topup` |
| position | int32 | no | 0-based order within the queue |
| segment_id | string | no | |
| video_id | string | no | eval-split video |
| start_s | float64 | no | |
| end_s | float64 | no | |
| target_label | string | yes | top-up only; never shown in the UI; used by 005 metrics |
| seed | int64 | no | |
| created_at | timestamp[us, UTC] | no | |

Primary key: (`queue`, `position`), and (`queue`, `segment_id`) is unique too. Rows are written once per queue and never modified (append-only, like `reviews`).

## Review rows written by the app

| Action | target_type | label_role | action | new_label | source | mode |
|---|---|---|---|---|---|---|
| blind: label chosen | segment | primary / secondary / indicator | add | = label | staff | blind |
| blind: label unticked on re-label | segment | same as before | reject | null | staff | blind |
| blind: clip unavailable | segment | observation | add | `clip_unavailable` | staff | blind |
| review: accept / reject / defer an AI label | moment | role of that AI label | accept / reject / defer | null | ai | review |
| review: edit an AI label | moment | role of that AI label | edit | the new label | ai | review |
| review: add a vocabulary label | moment | primary / secondary / indicator (by group) | add | = label | staff | review |
| review: add an observation | moment | observation | add | = observation type | staff | review |

Every row also has `reviewer`, `guideline_version` (from `labels.yaml` at save time), an optional `note`, and `created_at`.

## Derived state (computed, never stored)

- **Latest state** of (`target_id`, `label`): the newest row by `created_at`, then `review_id`.
- **Blind progress** of a reviewer in a queue: the queue positions whose segment has at least one blind row from that reviewer.
- **Moment status**: `unreviewed`, `partly`, `done` or `deferred` (research §9).
- **Eval unlock**: every `main` segment has a blind row, and every `overlap` segment has blind rows from at least 2 reviewers.

## Invariants

- Blind mode never shows, and never orders by, model output. Only the top-up queue uses model scores, and only to choose segments, whose order is then shuffled.
- No youth names, no per-person rows, no individual scores or rankings anywhere.
- Clip files exist only in the session temp folder, outside the repo, during the session.
