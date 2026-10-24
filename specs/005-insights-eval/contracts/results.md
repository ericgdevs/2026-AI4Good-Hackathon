# Contract: Result Folders (005 → 006)

006 reads 005's results by files only. This contract is the interface. A change that removes or renames anything here bumps `schema_version`.

## Location and completion

```
{OUTPUT_DIR}/results/
  evaluate-20261024T213005-1a2b/      # one per `tpa-index evaluate` run
  insights-20261024T213040-3c4d/      # one per `tpa-index insights` run
```

- The folder name is `{kind}-{YYYYmmddTHHMMSS}-{4 hex}`. It is created once and never modified afterwards.
- **A folder is complete when it contains `manifest.json`.** That file is written last, atomically (temp file + rename). Readers MUST ignore folders without it.
- **Latest result**: the complete folder of that kind whose name sorts last.
- Everything here is aggregate. There are no filenames, no transcript text, no frames, and no per-person or per-face fields. Video IDs are opaque (`v01`).

## manifest.json

```json
{
  "schema_version": 1,
  "kind": "evaluate",
  "created_at": "2026-10-24T21:30:05Z",
  "git_commit": "abc1234",
  "config_hash": "0123456789ab",
  "guideline_versions": ["0.4.0", "0.5.0"],
  "official_runs": {"predict": ["predict-…"], "moments": ["moments-…"], "audio": ["audio-…"], "people": ["people-…"]},
  "files": {"metrics.csv": "<sha256>", "metrics.json": "<sha256>"},
  "limitations": ["single labeler: no human-vs-human agreement", "..."]
}
```

## Tag values (exact strings)

| `tag` | meaning |
|---|---|
| `staff-verified` | every contributing segment's label is verified by staff (blind label or review decision) |
| `model estimate (unvalidated)` | no contributing segment is verified |
| `model estimate (learned)` | a per-segment prediction of the final learned model (`learned_predictions.csv`) |
| `mixed (NN% staff-verified)` | NN = round(100 × verified_share); 1 ≤ NN ≤ 99 |
| `metric` | an evaluation number (model vs blind labels); carries an interval instead of a verified share |

Every indicator row also has `verified_share` (0–1) as a number, so readers never parse the tag.

## evaluate-* folder

| file | content |
|---|---|
| `metrics.csv` | one row per (system, scope, group, label, metric) |
| `metrics.json` | the same rows plus headline numbers, the eval-set description and limitations |
| `confusion.csv` | system, human_primary, model_primary, count |
| `signal_indicators.csv` | per indicator: support, tp, fp, fn, precision, recall, f1, ci_low, ci_high, not_available, status, tag, system (`zero-shot` or the learned system; added at the end) |
| `ablation.csv` | one row per system: n_segments, primary_accuracy (+ CI), macro_f1_activity/category/actor (+ CI), kappa_primary/category (+ CI), rule fire counts |
| `topup.csv` | label, predicted_positive, tp, precision (top-up labels only; empty with header if none) |
| `failure_gallery.csv` | rank, segment_id, video_id, model_primary, human_primary, model_strength |
| `learned_predictions.csv` | final learned model, every official segment: segment_id, video_id, primary, primary_prob, activities, categories, actors, tag (`model estimate (learned)`); label lists are `;`-joined |
| `ablation.png`, `per_label_f1.png`, `confusion.png`, `coverage.png` | charts |
| `summary.md` | slide text: headline numbers with support and 95% CI, kappa sentence, limitations |
| `manifest.json` | written last |

### metrics.csv columns

`system, scope, group, label, metric, value, ci_low, ci_high, support, n_segments, status, tag`

- `system`: `zero-shot` | `zero-shot + signal rules` | `learned (cross-validated on our labels)`.
- `scope`: `all` | a video ID.
- `group`: `activity` | `category` | `actor` | `primary` | `uncertainty`.
- `metric`: `precision` | `recall` | `f1` | `macro_f1` | `accuracy` | `accuracy_real_activity` | `kappa` | `rate` | `coverage` | `accuracy_confident` | `accuracy_flagged`.
- `status`: `ok` | `too few to score` | `undefined`. When the status isn't `ok`, `value` and the CI are empty.
- `tag`: always `metric`.

## insights-* folder

| file | content |
|---|---|
| `indicators.csv` | one row per (indicator, scope) |
| `indicators.json` | the same rows, plus the headline and time breakdown |
| `headline.csv` | indicator, phase, value, unit, tag, verified_share, change_vs_early |
| `time_breakdown.csv` | scope, youth_s, staff_led_verified_s, staff_led_model_s, idle_s, total_s |
| `headline.png`, `time_breakdown.png` | charts |
| `summary.md` | slide text; every number carries its tag |
| `manifest.json` | written last |

### indicators.csv columns

`indicator, scope_type, scope, value, unit, numerator, denominator, base, tag, verified_share, status`

- `indicator`: `performance_share` | `active_participation_share` | `solo_moments_count` | `solo_moments_minutes` | `youth_voice_minutes` | `activity_share:<label>` | `activity_diversity_count` | `activity_diversity_entropy` | `audience_response_per_perf_min` | `positive_expression_rate` | `original_work_minutes` | `instruction_participation_ratio` | `coverage_faces` | `coverage_reviewed` | `expression_index` (optional, with `tag` = `model estimate (unvalidated)` and `status` = `example`).
- `scope_type`: `video` | `phase` | `all`. `scope` is a video ID, a phase name (`early`, `mid`, `showcase`) or `all`.
- `unit`: `share` (0–1) | `minutes` | `count` | `ratio` | `per_minute` | `index_0_100`.
- `base`: a short text naming the denominator, for example `youth non-idle time` or `segments with faces`.
- `status`: `ok` | `signal not available` | `not tagged yet` | `no videos in this phase`. When the status isn't `ok`, `value` is empty.

## Stability promise

- Columns are only ever added at the end. Readers must select columns by name.
- Tag strings are exactly as above.
- Phases come from `videos.phase`. `unknown` never appears as a headline scope; it is listed in `indicators.json` under `excluded_videos`.
