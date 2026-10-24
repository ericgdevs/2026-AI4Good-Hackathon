# Contract: Official Annotation Export (case format)

This is the public format judges and TPA receive. Renaming, reordering or adding a field is a breaking change.

## Fields (exact order)

`video_id, start_s, end_s, primary_label, secondary_labels, indicators, contextual_observation, confidence, source, review_decision, guideline_version, reviewer_note, review_date`

## JSON

A top-level array. One object per moment, with keys in the order above.

```json
[
  {
    "video_id": "mock01",
    "start_s": 12.0,
    "end_s": 36.0,
    "primary_label": "dance",
    "secondary_labels": ["performance", "actor_participant", "music"],
    "indicators": ["group_performance"],
    "contextual_observation": "camera_problem: framing cuts off the left side",
    "confidence": 0.734,
    "source": "staff",
    "review_decision": "edited",
    "guideline_version": "0.5.0",
    "reviewer_note": "was singing, it is dance",
    "review_date": "2026-10-24"
  }
]
```

- `secondary_labels` and `indicators` are arrays, which may be empty.
- `contextual_observation`, `reviewer_note` and `review_date` are `""` when empty, never `null`, so JSON and CSV read the same.

## CSV

- UTF-8 with a header row in the field order above, written with standard quoting (fields with commas, quotes or line breaks are quoted, and inner quotes are doubled).
- List fields are joined with `;` and no spaces (`performance;actor_participant;music`). An empty list is an empty cell.

## Values

| Field | Allowed |
|---|---|
| `source` | `ai`, `staff` |
| `review_decision` | `accepted`, `edited`, `rejected`, `deferred`, `pending` |
| `review_date` | `YYYY-MM-DD` (UTC) or empty |
| `confidence` | 0–1, 3 decimals |
| `start_s`, `end_s` | seconds, 2 decimals, `end_s > start_s` |

## Decision rules (summary of spec FR-005 to FR-010)

| Latest review of the primary label | `review_decision` | `primary_label` |
|---|---|---|
| none | `pending` | AI label |
| accept | `accepted` | AI label |
| edit | `edited` | `new_label` |
| reject | `rejected` (omitted with `--verified-only`) | AI label |
| defer | `deferred` | AI label |

- **Secondary and indicator labels**: rejected → dropped; edited → replaced; deferred or unreviewed → kept; staff-added → appended (and the record's `source` is `staff`).
- **Never present**: reviewer names, filenames, paths, transcript text.

## Files

| Path | Written by | Overwritten? |
|---|---|---|
| `{OUTPUT_DIR}/exports/annotations-{YYYYmmddTHHMMSS}-{4hex}-g{guideline}.json` / `.csv` | `tpa-index export` | never |
| `{OUTPUT_DIR}/exports/annotations-latest.json` / `.csv` | `tpa-index export` | yes, atomically |
| `docs/samples/annotation_export_sample.json` / `.csv` | `tpa-index export --sample` (mock only) | yes (checked in) |
