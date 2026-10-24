# Contract: Output Tables

Source of truth is `src/tpa_index/schema.py`. This document must match it. Nullable means the value may be empty.

## videos

| column | type | nullable | notes |
|---|---|---|---|
| video_id | string | no | opaque (`v01`) |
| filename | string | no | relative to `DATA_DIR`; never logged |
| duration_s | float64 | no | > 0 |
| fps | float64 | no | > 0 |
| orientation | string | no | landscape \| portrait |
| kind | string | no | live \| edited \| clip \| promo |
| phase | string | no | from `pipeline.yaml: video_phases` (default early, mid, showcase, unknown) |
| cohort | string | yes | free text |
| split | string | no | dev \| eval |

## segments

| column | type | nullable | notes |
|---|---|---|---|
| segment_id | string | no | `{video_id}_{start_ms}` |
| video_id | string | no | |
| start_s | float64 | no | ≥ 0 |
| end_s | float64 | no | > start_s |
| person_count | int32 | yes | per-segment aggregate |
| faces_detected | int32 | yes | per-segment aggregate |
| has_speech | bool | yes | |
| transcript_chars | int32 | yes | count only, no text |
| is_idle | bool | yes | |

## predictions (long format)

| column | type | nullable | notes |
|---|---|---|---|
| segment_id | string | no | |
| label | string | no | name from `labels.yaml` |
| label_group | string | no | activity \| category \| actor \| audio \| expression \| indicator \| status |
| score | float32 | no | 0–1 |
| model | string | no | |
| run_id | string | no | |

## moments

| column | type | nullable | notes |
|---|---|---|---|
| moment_id | string | no | `{video_id}_{start_ms}_{primary_label}` |
| video_id | string | no | |
| start_s | float64 | no | |
| end_s | float64 | no | > start_s |
| primary_label | string | no | |
| secondary_labels | list\<string\> | no | may be empty |
| indicators | list\<string\> | no | may be empty |
| confidence | float32 | no | 0–1 |
| needs_review | bool | no | |
| run_id | string | no | |

## runs

| column | type | nullable | notes |
|---|---|---|---|
| run_id | string | no | |
| stage | string | no | e.g. `segment`, `mock` |
| model | string | yes | null for model-free stages |
| config_hash | string | no | 12 hex chars |
| hardware | string | no | GPU model or Mac chip or CPU |
| n_segments | int32 | no | ≥ 0 |
| runtime_s | float64 | no | > 0 |
| peak_mem_gb | float64 | yes | null if unmeasurable |
| started_at | timestamp[us, UTC] | no | |
| git_commit | string | yes | short commit (7) + `+dirty` if tracked files changed, or `nogit`; null for runs written before 002 |

## reviews (DuckDB `reviews` table, append-only)

| column | type | nullable | notes |
|---|---|---|---|
| review_id | string | no | uuid4 hex |
| target_type | string | no | segment \| moment |
| target_id | string | no | |
| label | string | no | label being reviewed |
| label_role | string | no | primary \| secondary \| indicator |
| action | string | no | accept \| edit \| reject \| defer \| add |
| new_label | string | yes | required iff action ∈ {edit, add} |
| source | string | no | ai \| staff |
| mode | string | no | blind \| review |
| guideline_version | string | no | from `labels.yaml` |
| reviewer | string | no | display name or initials |
| note | string | yes | |
| created_at | timestamp[us, UTC] | no | |
