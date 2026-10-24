# Contract: Config (005)

There's one new block in `config/pipeline.yaml`, validated by `config.load_pipeline()`. Nothing else changes in config. `labels.yaml` is unchanged: indicator labels stay `source: staff`.

```yaml
insights:
  min_support: 5                   # labels with fewer eval positives: "too few to score"
  bootstrap: {resamples: 1000, block: 10, seed: 2026}
  failure_gallery: 20
  applause_after_s: 10             # applause within this many seconds after a performance counts as response
  audio_threshold: 0.5             # fixed AST probability for every audio condition (NOT the calibrated top-10% value)
  primary_like: [singing, lyricism_spoken_word, acting_theater, dance]   # "performing" primaries (used by rules)
  fusion:                          # system "zero-shot + signal rules": drop a visual call unless a condition holds
    - {label: singing, require_any: [{audio: singing_voice}, {audio: music}]}
    - {label: lyricism_spoken_word, require_any: [{audio: speech}, {has_speech: true}]}
  indicator_rules:                 # signal-based model estimates (always "model estimate (unvalidated)")
    solo_on_stage:
      all: [{person_count: {eq: 1}}, {any: [{clears: performance}, {primary_in: primary_like}]}]
    group_performance:
      all: [{person_count: {gte: 2}}, {any: [{clears: performance}, {primary_in: primary_like}]}]
    voice_projection:
      all: [{any: [{indicator: solo_on_stage}, {indicator: group_performance}]},
            {any: [{audio: singing_voice}, {audio: speech}]}]
```

## Condition types (the whole grammar)

| condition | true when |
|---|---|
| `{audio: <label>}` | the AST probability of that audio label ≥ `insights.audio_threshold` (fixed, default 0.5) |
| `{has_speech: true}` | the transcript stage set `has_speech` |
| `{clears: <visual label>}` | the main-model score ≥ its calibrated threshold |
| `{primary_in: <list name or list>}` | the model primary is in the list (`primary_like` refers to the key above) |
| `{person_count: {eq\|gte\|lte: n}}` | the people stage's count compares true |
| `{indicator: <name>}` | that indicator rule fired on the segment (rules are evaluated in file order) |
| `{all: [...]}`, `{any: [...]}` | combinators |

A condition whose signal is missing for the segment (no official run, or a null column) is **not available**. A rule needing it does not fire, and the segment counts as "signal not available" for that rule.

## Validation (load time, exit 3)

- `insights.min_support`, `failure_gallery`, `bootstrap.resamples` and `bootstrap.block` are integers ≥ 1; `applause_after_s` ≥ 0; `audio_threshold` is within (0, 1).
- Fusion `label`, `clears` and `primary_in` entries are model-scored visual labels. `audio` entries are audio labels. `indicator_rules` keys are `indicator`-group labels from `labels.yaml`.
- Unknown condition keys are rejected, naming the key.
- `{indicator: x}` may only refer to a rule defined earlier in the block.

## Thresholds

- **Visual labels** (activity, category, actor, under the main model): calibrated per-model thresholds from the merged config (`thresholds.local.yaml`, dev videos only). A missing one stops the command (exit 3) and names the label, the model and the calibrate command. There is no fallback to `label_threshold`.
- **Audio labels** (`singing_voice`, `music`, `speech`, `applause`, `silence`): the fixed `insights.audio_threshold` (default 0.5) for every 005 rule and indicator.
  - **Why:** AST outputs real sigmoid probabilities. Calibration takes each label's top 10% on dev videos, which put `music` and `speech` near 0.95. As a fusion veto that would remove most real singing and spoken word. A plain probability cut is the honest reading of AST.
  - **Calibrated audio thresholds** stay in `thresholds.local.yaml` for other uses; 005 does not read them.
  - This is listed in the limitations: the 0.5 cut is a fixed choice, not tuned on dev data.
- **Expression labels**: no threshold. The positive expression rate uses the argmax over the mapped labels.
