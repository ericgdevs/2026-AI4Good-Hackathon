# Contract: Python API and CLI (005)

This extends the 001–004 contracts. No existing signature changes.

## Python (`tpa_index`)

```python
# config.py (new)
required_threshold(pipeline, model, label, stage_hint="predict") -> float
#   calibrated per-model threshold; ConfigError naming label, model and `tpa-index calibrate --stage <stage_hint>`
#   if missing (no fallback to label_threshold)

# view.py (new): one row per official segment, everything 005 needs, no filenames, no text
check_inputs(kind: str, root=None) -> None
#   kind "evaluate" | "insights": required official stages cover the videos in scope, and every calibrated visual
#   threshold the metrics and rules need exists (audio uses insights.audio_threshold); one ConfigError listing every missing item
segment_view(pipeline, labels, root=None) -> pandas.DataFrame
#   columns: segment_id, video_id, split, phase, start_s, end_s, slice_s (non-overlapping time), is_idle,
#   person_count, has_speech, faces_detected, model_cleared (set), model_primary, model_strength, uncertain_flag,
#   needs_review, no_call, audio_ok (dict label->bool|None), expr_top (label|None)
human_labels(labels, pipeline, root=None) -> pandas.DataFrame
#   blind main-sample segments: segment_id, primary, activities, categories, actors, indicators, queue, skipped
verified_labels(view, labels, pipeline, root=None) -> pandas.DataFrame
#   per segment and group: verified label set or None (blind first, then review decisions on the containing moment)

# rules.py (new): the insights.fusion / insights.indicator_rules grammar
evaluate_condition(cond, seg, ctx) -> bool | None      # None = signal not available
apply_fusion(view, rules, ctx) -> (view_fused, fire_counts)
indicator_estimates(view, rules, ctx) -> DataFrame     # segment_id, indicator, fired (bool|None)

# evaluate.py (new)
label_metrics(human_sets, model_sets, labels, min_support) -> list[dict]
primary_metrics(human_primary, model_primary) -> dict   # accuracy, accuracy_real_activity, confusion, kappa
category_kappa(human_categories, model_category) -> dict  # skips segments with 2+ human categories (counted)
uncertainty_metrics(eval_df) -> dict
block_bootstrap(eval_df, stat_fn, block, resamples, seed) -> (low, high, n_valid)
run_evaluate(pipeline, labels, root=None) -> dict       # everything for the evaluate result folder

# learned.py (new): logistic regressions on official per-segment signals (no embeddings)
features(view) -> (np.ndarray, list[str])           # missing signal -> 0 + a "<source>_missing" column
block_ids(df, block) -> np.ndarray                  # same blocks as block_bootstrap (shared helper)
cross_validated(df, X, labels, cfg, block) -> pd.DataFrame   # view-like model columns per labeled row, each from a
                                                             # model that never saw its block
fit_final(df, X_all, view, labels, cfg) -> pd.DataFrame      # predictions for every official segment
learned_view(view, preds) -> pd.DataFrame           # the view with model columns replaced (for metrics and insights)

# indicators.py (new)
staff_led(view, verified) -> Series[bool], source Series["verified"|"model"|None]
youth_indicators(view, verified, estimates, pipeline) -> list[dict]   # rows of indicators.csv
headline(rows, videos) -> list[dict]
run_insights(pipeline, labels, root=None) -> dict

# report.py (new)
new_result_dir(kind, root=None) -> Path                 # OUTPUT_DIR/results/{kind}-{ts}-{hex}
write_result(kind, payload, root=None) -> Path          # CSV/JSON/charts/summary, manifest.json last
latest_result(kind, root=None) -> Path | None           # newest folder with manifest.json (for 006)
```

## CLI

| command | does | exit codes |
|---|---|---|
| `tpa-index evaluate [--dry-run]` | checks inputs, computes Part 1 (metrics, ablation, signal-based indicators, failure gallery), writes an `evaluate-*` result folder; prints headline numbers with CIs | 0; 3 missing official stage or threshold, or no eval labels; 2 guard |
| `tpa-index insights [--dry-run]` | checks inputs, computes Part 2 (indicators per video and phase, headline, optional index), writes an `insights-*` result folder; prints the headline with tags | 0; 3 missing official stage or threshold; 2 guard |

`--dry-run` computes and prints everything and writes nothing. Both commands print only IDs and aggregate numbers.
