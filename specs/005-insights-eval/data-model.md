# Data Model: Evaluation and Program Indicators (005)

**No schema change** to any 001–004 table or to the review log. 005 only reads them, and writes result folders described in `contracts/results.md`.

## In-memory tables

### segment view (one row per official segment)

| field | from | notes |
|---|---|---|
| segment_id, video_id, start_s, end_s, is_idle | official segments | |
| split | official videos | `filename` is dropped on read |
| phase, phase_source | local videos config if present, else official videos | `local config` / `stored runs` |
| slice_s | computed | `min(end_s, next.start_s) - start_s`; the last segment uses its own `end_s`. Sums to the video's duration |
| person_count, faces_detected, has_speech | `read_signals()` | null when the stage has no official run for the video |
| model_cleared | main-model predictions + `required_threshold` | sets per group: activity, category, actor |
| model_primary, model_strength | 002 rule | `uncertain_mixed` if nothing clears or no moment contains the segment |
| uncertain_flag | rules row `uncertain_mixed` ≥ 0.5 | |
| needs_review, no_call | containing moment | no_call = in no moment |
| audio_ok | AST predictions ≥ `insights.audio_threshold` (fixed, default 0.5) | label → bool, or null if no audio run |
| expr_top | expression predictions | argmax label, or null without faces / expression run |

### human labels (blind, main sample)

`segment_id, primary, activities (set), categories (set), actors (set), indicators (set), queue (main|topup), skipped (bool)`

The latest state per (segment, label); a label counts when its latest action is `add`. Primary is the `primary`-role label, or `uncertain_mixed`.

### verified labels (for indicators)

Per segment and group: the verified label set, or none. Precedence: blind label → review decision on the containing moment → none (then the model call is used).

### staff-led and time classes (per segment)

| class | rule |
|---|---|
| idle | `is_idle` |
| staff-led (verified) | verified actors ∋ `actor_staff_facilitation` and ∌ `actor_participant`, or verified categories ∋ `instruction` and verified actors ∌ `actor_participant` |
| staff-led (model) | no verified actor or category, and (`actor_staff_facilitation` or `instruction` clears) and `actor_participant` does not clear |
| youth | everything else |

Idle wins over staff-led. Youth time is the denominator of every TPA-facing share.

## Output records

The CSV and JSON columns and tag strings are in `contracts/results.md`.

- **Metric record**: system, scope, group, label, metric, value, CI, support, n_segments, status, tag (`metric`).
- **Indicator record**: indicator, scope, value, unit, numerator, denominator, base, tag, verified_share, status.
- **Manifest**: written last; it marks the result folder complete.

## Lifecycle

```
check_inputs ─fail→ exit 3 (nothing written)
     │ok
build views → compute → --dry-run? print, stop
                         │no
            write files into new results/{kind}-… folder → manifest.json last (complete)
```
