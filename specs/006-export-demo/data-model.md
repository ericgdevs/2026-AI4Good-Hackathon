# Data Model: 006

No table gains or loses a column. Everything below is derived at read time or written as files outside the Parquet/DuckDB contract.

## Inputs (existing, read only)

| Source | Read through | Used for |
|---|---|---|
| moments (official) | `read_official("moments")` | export records, timeline |
| reviews (`mode = review`, moment targets) | `read_reviews()` → `review.latest_state(..., "review")` | current label state |
| reviews (`mode = blind`, segment targets) | `latest_state(..., "blind")` | verified actor labels for the staff-led rule only |
| segments + 004 signals | `schema.read_signals()`, with `transcript_chars` dropped at once | person count, faces, speech yes/no |
| predictions (main, audio, expression) | `read_official("predictions")`, `stage="audio"`, `stage="expression"` | audio cues, expression suggestion, actor/instruction scores |
| videos (official) | `review.review_videos()` (respects the eval lock) | video picker, plain names |
| runs | `schema.list_runs()`, plus each run folder's `videos.parquet` for durations | benchmark summary |
| 005 results | `{OUTPUT_DIR}/results/insights-*/` (`manifest.json`, `headline.csv`, `time_breakdown.csv`) | insights page |
| `config/labels.yaml`, `config/theme.yaml`, `config/guideline_history.yaml` | `config` loaders | vocabulary, palette, guideline check |

## Export record (derived, written to files)

| Field | Type | Rule |
|---|---|---|
| `video_id` | string | from the moment |
| `start_s`, `end_s` | number, 2 dp | from the moment |
| `primary_label` | string | the AI primary, or `new_label` if the latest primary review is an edit |
| `secondary_labels` | list[string] | AI secondaries minus rejected, edits replaced, plus surviving staff adds (role secondary). De-duplicated, in order |
| `indicators` | list[string] | the same rule for role indicator |
| `contextual_observation` | string or empty | observation rows in time order, `type` or `type: note`, joined with `; `, exactly as stored |
| `confidence` | number, 3 dp | the moment's model confidence |
| `source` | `ai` \| `staff` | `staff` if the primary was edited or any staff-added label survived |
| `review_decision` | `accepted` \| `edited` \| `rejected` \| `deferred` \| `pending` | from the latest review of the primary label |
| `guideline_version` | string | from the primary's deciding review, else the current config |
| `reviewer_note` | string or empty | note of the newest non-reject review on the moment |
| `review_date` | `YYYY-MM-DD` (UTC) or empty | date of that same review |

Validation:
- the field set and order are exactly the 13 above;
- the decision and source values come from the enums above;
- no reviewer name, filename, path or transcript text appears;
- `--verified-only` removes the `rejected` records.

## Export file set

- `{OUTPUT_DIR}/exports/annotations-{ts}-{hex}-g{guideline}.json|.csv`: created exclusively, never overwritten.
- `{OUTPUT_DIR}/exports/annotations-latest.json|.csv`: replaced atomically on each run.
- `docs/samples/annotation_export_sample.json|.csv`: mock only, about 20 records, checked in.

## Segment signal row (view model, never stored)

`segment_id` (internal only, not shown), `time`, `people` (int or "not available"), `speech` (Yes / No / not available), `audio_cues` (list of display names, "none" or "not available"), `expression` ("AI suggestion, unvalidated: X", "none", "no faces" or "not available"), `led_by` (Youth / Staff-led (staff-verified) / Staff-led (model estimate)).

The allow-list has no transcript field.

## Staff-led state (derived per segment)

| State | Condition |
|---|---|
| staff-led, staff-verified | the verified actor labels for the segment include staff facilitation and not participant |
| youth, staff-verified | verified actor labels exist and do not meet the line above |
| staff-led, model estimate | no verified actor labels, (`actor_staff_facilitation` ≥ t OR `instruction` ≥ t) AND `actor_participant` < t |
| youth, model estimate | no verified actor labels, and the condition above is false |

Verified source order: the segment's blind labels, then the review labels of the moment that contains the segment.

## Insights view (derived from 005 files)

- **Folder**: the newest `insights-*` folder whose `manifest.json` exists and has `schema_version == 1`.
- **Headline row**: `indicator`, `phase` (early / mid / showcase), `value`, `unit`, `tag` (shown verbatim), `verified_share`, `change_vs_early`.
- **Shown indicators**: `performance_share`, `solo_moments_count` (and `solo_moments_minutes` if present), `youth_voice_minutes`, `positive_expression_rate`.
- **Context**: `time_breakdown.csv`, scope `all`: staff-led time (verified plus model) and idle / setup time.

States: **available**, **not available yet** (no complete folder), or **unsupported format** (`schema_version` or columns don't match).

## Theme palette (`config/theme.yaml`)

| Key | Value |
|---|---|
| `background` | `#F8F7F8` |
| `surface` | `#FFFFFF` |
| `text` | `#201C25` |
| `text_secondary` | `#6F717A` |
| `accent` | `#7851B9` |
| `accent_deep` | `#4100AD` |
| `chart_series` | `[#7851B9, #0099B3, #FF4255]` |
| `credit` | `Built for The Performance Academy` |

Validation:
- every value is `#RRGGBB`;
- `chart_series` has at least 3 entries;
- contrast is ≥ 4.5:1 for `text`, `text_secondary`, `accent` and `accent_deep` on both `background` and `surface`, and for white on `accent`.

`chart_series[1:]` are never used as text colours.

## Guideline history (`config/guideline_history.yaml`)

The file maps each version to a fingerprint, for example `"0.5.0": "<12 hex>"`.

The fingerprint is sha256[:12] over the sorted label (name, group, source) triples plus the sorted observation-type names.

**State rules**:
- **Recorded, fingerprint matches**: ok.
- **Recorded, fingerprint differs**: refused ("vocabulary changed without bumping guideline_version").
- **Not recorded**: ok, and the `check` command prints the line to add.

## Benchmark summary (written to `{OUTPUT_DIR}/benchmarks/`)

There is one row per stage (segment, predict, transcript, audio, people, expression, moments), plus `total`:

- `stage`;
- `mac_model`, `mac_hardware`, `mac_runtime_s`, `mac_s_per_video_hour`, `mac_peak_mem_gb`;
- `hpc_model`, `hpc_hardware`, `hpc_s_per_video_hour`, `hpc_peak_mem_gb`;
- `status`: ok / failed / no HPC run / incomplete: missing ….

The JSON and Markdown summaries have no filename or path. They carry the opaque video ID, which is fine because the outputs folder is outside the repo.
