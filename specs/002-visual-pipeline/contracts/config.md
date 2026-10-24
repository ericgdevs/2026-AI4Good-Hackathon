# Contract: Config Changes (002)

## config/labels.yaml

- `guideline_version: "0.2.0"`
- Each label may have a new optional `source: model | rules | staff` field. The default is `model`. `rules` labels are computed flags; `staff` labels (the indicators) are set by staff only. Neither has prompts. A rules label has no `prompts` and needs no `color` contrast rule beyond `#RRGGBB`.
- The status group is now:

```yaml
  - name: uncertain_mixed
    group: status
    source: rules          # written by rules (model "rules"), never by an image model
    color: "#525252"
  - name: needs_review
    group: status
    source: rules          # = moments.needs_review; prompts removed
    color: "#bdbdbd"
```

- `uncertain` and `mixed` are removed. The status group is rules-only.
- Validation: a model label still needs 3–5 prompts. A rules label must have no prompts. Every group still needs at least one label.

## config/pipeline.yaml (new keys)

```yaml
visual_groups: [category, actor, activity]
label_thresholds: {}             # optional {model name as in predictions.model: {label: t}}; fallback label_threshold
models:
  main:
    hpc: google/siglip2-so400m-patch14-384   # used on cuda
    mac: google/siglip2-base-patch16-224     # used on mps/cpu
  motion:
    enabled: false
    id: microsoft/xclip-base-patch32
    labels: [dance, acting_theater]          # must be activity labels
    frames_per_window: 8                     # X-CLIP needs exactly its trained frame count
    other_prompt: "people doing some other activity"
  batch_windows: {cuda: 64, mps: 8, cpu: 4}
idle:
  prompts:
    - "an empty room"
    - "people setting up equipment before a show"
    - "people waiting around with nothing happening"
  score_threshold: 0.1           # fallback; per-model value from calibrate wins
  change_threshold: 0.02         # mean |frame diff| on 64x64 grayscale in 0-1; hand-set, never calibrated
  calibrate_top_pct: 20          # idle percentile, separate from the label one (idle share differs from any label's)
calibrate:
  top_pct: 10                    # label thresholds: (100 - top_pct) percentile of dev scores
  min_threshold: 0.001           # absolute floor for label and idle thresholds (so400m: cos ~0.08)
  null_group: "null"             # quoted: bare null is YAML for nothing; scored by predict, never a result
  null_percentile: 99            # null floor = this percentile of each dev segment's max null score
  min_segments: 20               # fewer dev segments than this: keep the fallback
```

Removed keys: `moments.needs_review_below` and `moments.conflict_margin`, replaced by per-label thresholds, `moments.close_ratio` and `moments.needs_review_margin`. (`uncertain_margin`, an absolute top1 − top2 gap, was removed after the first real run in favour of `moments.close_ratio`.)

New moments key:

```yaml
moments:
  close_ratio: 0.8           # uncertain_mixed if 2nd activity strength (score/threshold) >= this x top
  needs_review_margin: 0.2   # needs_review if confidence < threshold * (1 + margin)
```

Validation:
- `visual_groups` ⊆ the label groups, and each group has at least one model-scored label;
- `label_thresholds` values are in [0, 1], and each label exists and is model-scored;
- `0 ≤ moments.close_ratio ≤ 1`;
- `0 ≤ moments.needs_review_margin ≤ 1`;
- `batch_windows` values ≥ 1;
- every `motion.labels` entry is an activity label;
- `idle.prompts` has 1–5 entries;
- both idle thresholds are in [0, 1];
- `0 < idle.calibrate_top_pct < 100` and `0 < calibrate.top_pct < 100`;
- `0 ≤ calibrate.min_threshold ≤ 1`;
- `0 < calibrate.null_percentile ≤ 100`;
- `calibrate.null_group` is a quoted string naming a group with model labels, and not a visual, primary or secondary group;
- `calibrate.min_segments ≥ 1`.

## config/official_runs.yaml (new, committed)

```yaml
# Run IDs that count for results, per stage. Fill in with `tpa-index runs`.
predict: []
moments: []   # optional: rebuilt moments take precedence over predict's
segment: []
```

Run IDs only, never filenames or metadata.

## config/thresholds.local.yaml (new, gitignored, written by `tpa-index calibrate`)

```yaml
label_thresholds:
  <model name>: {<label>: <0-1>, ...}
idle_thresholds:
  <model name>: <0-1>
```

`load_pipeline()` merges this file over `pipeline.yaml: label_thresholds`, and the local file wins. `TPA_THRESHOLDS_FILE` overrides its path, for tests and checkpoints only. It is never committed, because its values are derived from footage (Constitution I).
