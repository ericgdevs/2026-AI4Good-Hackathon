# Data Model: Visual Pipeline

**One schema change:** `segments` gains two nullable `float32` columns, `idle_score` and `visual_change`. Older runs read them back as null. Otherwise 002 writes the existing 001 tables; see `specs/001-foundation/contracts/tables.md`.

## What one predict run writes (one run per video)

`{OUTPUT_DIR}/runs/{predict-…}/` contains:

| file | rows | notes |
|---|---|---|
| `videos.parquet` | 1 | same fields as the segment stage (probed) |
| `segments.parquet` | 1 per window | `is_idle`, `idle_score` and `visual_change` filled; other signals null |
| `predictions.parquet` | segments × (visual + null labels + 1) | main model: one row per visual label and per null-group label (`label_group` "null"; noise floor only, never a result). `rules`: one `uncertain_mixed` row. Optional motion model: one row per motion label. |
| `moments.parquet` | merged | from main-model rows only, non-idle segments only, using the video's effective `window_s` |
| `runs.parquet` | 1 | written last (completion marker); `stage = "predict"`, `model` = main model id |

## Label vocabulary change (guideline 0.1.0 → 0.2.0)

- The `status` group becomes two rules labels with no prompts:
  - `uncertain_mixed`: written per segment as a rules prediction row;
  - `needs_review`: carried by the existing `moments.needs_review` column. No prediction rows.
- `uncertain` and `mixed` are removed.
- A label's `source` is `model` (the default) or `rules`. Rules labels need no prompts. Every other label rule from 001 still applies.
- Visual groups (`pipeline.yaml: visual_groups`): category, actor, activity. Status is rules-only. Indicators are `source: staff` (no prompts, never scored). Audio and expression labels are not scored in 002. The `"null"` group (`calibrate.null_group`) is scored by predict for the noise floor and excluded from every result.
- Guideline 0.3.0 replaces the 002 starter vocabulary with the approved one (`specs/001-foundation/contracts/config.md`).

## Derived per-segment fields (computed, not stored as columns)

| field | rule |
|---|---|
| primary activity | highest strength = score / `threshold(model, label)` among activity labels with strength ≥ 1 |
| secondary labels | other visual labels ≥ their own threshold |
| `uncertain_mixed` | stored as a rules row: 1 if no activity clears its threshold, or the second-highest cleared strength ≥ `moments.close_ratio` × the top strength |
| moment `needs_review` | `confidence < threshold(model, primary_label) × (1 + moments.needs_review_margin)` or any segment `uncertain_mixed = 1` |
| `is_idle` | stored on the segment: `idle_score` ≥ `idle_threshold(model)` (calibrated per model, falling back to `idle.score_threshold`) AND `visual_change` < `idle.change_threshold` (hand-set; null when there are fewer than 2 frames, and then never idle) |

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
| `segments.parquet` | 1 per window | copied, with `is_idle` recomputed from the stored `idle_score` and `visual_change` |
| `predictions.parquet` | segments | `uncertain_mixed` rules rows only, recomputed with the current thresholds |
| `moments.parquet` | merged | current thresholds and the effective window |
| `runs.parquet` | 1 | `stage = "moments"`, `model` = main model id |

## Local thresholds file (gitignored)

`config/thresholds.local.yaml` is written by `tpa-index calibrate`. It holds data-derived values, so it never goes into git (Constitution I).

```yaml
label_thresholds:
  siglip2-so400m-patch14-384: {singing: 0.14, dance: 0.11}   # key = predictions.model
idle_thresholds:
  siglip2-so400m-patch14-384: 0.03
```

Calibration rules:
- null floor = percentile `calibrate.null_percentile` (default 99) of each dev segment's max null-group score;
- label threshold = max(percentile `(100 − calibrate.top_pct)` of dev scores, null floor, `calibrate.min_threshold` (default 0.001));
- idle threshold = max(percentile `(100 − idle.calibrate_top_pct)` of `idle_score` over dev segments with `visual_change < idle.change_threshold`, `calibrate.min_threshold`);
- labels with fewer than `calibrate.min_segments` dev segments, and an idle set that small, keep the fallback.
