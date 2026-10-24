# Data Model: Shared Data Foundation

Exact column types are in [contracts/tables.md](contracts/tables.md). This file covers entities, relationships, validation and lifecycle.

## Relationships

```text
videos 1 ──< segments 1 ──< predictions >── 1 runs
   │                                          │
   └──< moments >─────────────────────────────┘
reviews ── target_id → segments.segment_id | moments.moment_id   (by target_type)
```

- `segments.video_id`, `moments.video_id` → `videos.video_id`
- `predictions.segment_id` → `segments.segment_id`
- `predictions.run_id`, `moments.run_id` → `runs.run_id`
- `reviews.target_id` → segment or moment, chosen by `target_type`; it is not enforced at write time because reviews may outlive a run.

## Entities

| Entity | Primary key | Lifecycle |
|---|---|---|
| Video | `video_id` (opaque, e.g. `v01`; mock: `mock01`) | Written once per segmenter or mock run from video metadata plus a probe. |
| Segment | `segment_id` = `{video_id}_{start_ms}` | Created by the segmenter; signal columns stay null until later features fill them. Those features write a new run, never patch an old one. |
| Prediction | (`segment_id`, `label`, `model`, `run_id`) | One row per segment × label × model run. |
| Moment | `moment_id` = `{video_id}_{start_ms}_{primary_label}` | Derived by the merge rules in research §12. |
| Run | `run_id` = `{stage}-{YYYYmmddTHHMMSS}-{4hex}` | One row per successful stage execution, written when the stage ends. Its `runs.parquet` marks the run folder complete; failed runs have none and are ignored by readers. `git_commit` (nullable, added in 002) records the code version. |
| Review | `review_id` (uuid4 hex) | Append-only. A correction is a new row; there is no update or delete. |

## Validation rules (applied on every write)

1. The column set must equal the schema exactly. Missing or extra columns are rejected, so identity-like columns such as `person_id`, `track_id` or `embedding` can never be written.
2. Types must cast to the pyarrow schema, and non-nullable fields must have no nulls.
3. Enums:
   - `split` ∈ {dev, eval}
   - `orientation` ∈ {landscape, portrait}
   - `kind` ∈ {live, edited, clip, promo}
   - `phase` ∈ `pipeline.yaml: video_phases`
   - `label_group` ∈ {activity, category, actor, audio, expression, indicator, status}
   - `target_type` ∈ {segment, moment}
   - `label_role` ∈ {primary, secondary, indicator}
   - `action` ∈ {accept, edit, reject, defer, add}
   - `source` ∈ {ai, staff}
   - `mode` ∈ {blind, review}
4. Ranges:
   - `score` and `confidence` must be in [0, 1].
   - `duration_s`, `fps` and `runtime_s` must be > 0.
   - Counts must be ≥ 0.
5. Times: `end_s > start_s` and `start_s ≥ 0`.
6. Primary keys must be unique within the write.
7. Reviews: `new_label` is required when `action` ∈ {edit, add} and must be null otherwise.
8. A failed write writes nothing. Pipeline-output files are never overwritten.

Cross-table referential checks (e.g. every `predictions.segment_id` exists) are run by `validate_dataset(dir)` and by the tests, not on each single-table write.

## Identity exclusion

No table has a column for person, track, face or embedding. Person counts and face counts are integers aggregated per segment (Constitution IV).

## Storage

- `{OUTPUT_DIR}/runs/{run_id}/{videos,segments,predictions,moments,runs}.parquet` holds pipeline outputs, one folder per run, immutable.
- `{OUTPUT_DIR}/reviews.duckdb` is a single append-only log shared across runs.
- The mock generator writes to the same layout under a run ID such as `mock-…`, so downstream code has a single read path (Constitution XII).
