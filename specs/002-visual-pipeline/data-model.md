# Data Model: Visual Pipeline

**No table schema changes.** 002 writes the existing 001 tables. See `specs/001-foundation/contracts/tables.md`.

## What one predict run writes (one run per video)

`{OUTPUT_DIR}/runs/{predict-…}/` contains:

| file | rows | notes |
|---|---|---|
| `videos.parquet` | 1 | same fields as the segment stage (probed) |
| `segments.parquet` | 1 per window | `is_idle` filled; other signals null |
| `predictions.parquet` | segments × (visual labels + 1) | main model: one row per visual label scored by the model. `rules`: one `uncertain_mixed` row. Optional motion model: one row per motion label. |
| `moments.parquet` | merged | from main-model rows only, non-idle segments only, using the video's effective `window_s` |
| `runs.parquet` | 1 | written last (completion marker); `stage = "predict"`, `model` = main model id |

## Label vocabulary change (guideline 0.1.0 → 0.2.0)

- The `status` group becomes two rules labels with no prompts:
  - `uncertain_mixed`: written per segment as a rules prediction row;
  - `needs_review`: carried by the existing `moments.needs_review` column. No prediction rows.
- `uncertain` and `mixed` are removed.
- A label's `source` is `model` (the default) or `rules`. Rules labels need no prompts. Every other label rule from 001 still applies.
- Visual groups (`pipeline.yaml: visual_groups`): category, actor, activity. Status is rules-only. Audio, expression and indicator labels are not scored in 002.

## Derived per-segment fields (computed, not stored as columns)

| field | rule |
|---|---|
| primary activity | highest activity score among labels ≥ their own threshold `threshold(model, label)` |
| secondary labels | other visual labels ≥ their own threshold |
| `uncertain_mixed` | stored as a rules row: 1 if no activity clears its threshold, or top1 − top2 < `uncertain_margin` |
| moment `needs_review` | `confidence < threshold(model, primary_label) × (1 + moments.needs_review_margin)` or any segment `uncertain_mixed = 1` |
| `is_idle` | stored on the segment: idle-prompt score ≥ `idle.score_threshold` AND frame change < `idle.change_threshold` (needs ≥ 2 frames) |

## Official runs

```yaml
# config/official_runs.yaml
predict: []        # run_ids that count for results, one per video
moments: []        # optional: rebuilt moments (tpa-index moments) take precedence over predict's
segment: []
```

Validation, applied on read:
- every listed ID must be a complete run folder;
- within a stage, no two runs may share a `video_id`.

Errors name the run ID, never a path.

## Invariants

- No identity data: embeddings live only in memory during a run and are never written (Constitution IV).
- A missing score is never stored as 0. Labels a model does not cover have no rows.
- Mac and HPC runs are separate runs. `read_official` decides which one counts.

## Moments rebuild run (stage `moments`, one per video)

| file | rows | notes |
|---|---|---|
| `videos.parquet` | 1 | copied from the source predict run |
| `predictions.parquet` | segments | `uncertain_mixed` rules rows only, recomputed with the current thresholds |
| `moments.parquet` | merged | current thresholds and the effective window |
| `runs.parquet` | 1 | `stage = "moments"`, `model` = main model id |

## Local thresholds file (gitignored)

`config/thresholds.local.yaml` is written by `tpa-index calibrate`. It holds data-derived values, so it never goes into git (Constitution I).

```yaml
label_thresholds:
  siglip2-so400m-patch14-384: {singing: 0.14, dancing: 0.11}   # key = predictions.model
```
