# Contract: Config Changes (002)

## config/labels.yaml

- `guideline_version: "0.2.0"`
- Each label may have a new optional `source: model | rules` field. The default is `model`. A rules label has no `prompts` and needs no `color` contrast rule beyond `#RRGGBB`.
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
uncertain_margin: 0.1            # top1 - top2 activity gap below this => uncertain_mixed
label_thresholds: {}             # optional {model_id: {label: t}}; fallback label_threshold
models:
  main:
    hpc: google/siglip2-so400m-patch14-384   # used on cuda
    mac: google/siglip2-base-patch16-224     # used on mps/cpu
  motion:
    enabled: false
    id: microsoft/xclip-base-patch32
    labels: [dancing, acting]                # must be activity labels
    frames_per_window: 8                     # X-CLIP needs exactly its trained frame count
    other_prompt: "people doing some other activity"
  batch_windows: {cuda: 64, mps: 8, cpu: 4}
idle:
  prompts:
    - "an empty room"
    - "people setting up equipment before a show"
    - "people waiting around with nothing happening"
  score_threshold: 0.1           # SigLIP sigmoid scale; tune on dev split
  change_threshold: 0.02         # mean |frame diff| on 64x64 grayscale in 0-1
```

Removed keys: `moments.needs_review_below` and `moments.conflict_margin`, replaced by per-label thresholds and `uncertain_margin`.

Validation:
- `visual_groups` ⊆ the label groups, and each group has at least one model-scored label;
- `label_thresholds` values are in [0, 1], and each label exists and is model-scored;
- `0 ≤ uncertain_margin < 1`;
- `batch_windows` values ≥ 1;
- every `motion.labels` entry is an activity label;
- `idle.prompts` has 1–5 entries;
- both idle thresholds are in [0, 1].

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
  <model_id>: {<label>: <0-1>, ...}
```

`load_pipeline()` merges this file over `pipeline.yaml: label_thresholds`, and the local file wins. It is never committed, because its values are derived from footage (Constitution I).
