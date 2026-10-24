# Contract: Python API and CLI (005)

This extends the 001–004 contracts. No existing signature changes. Signatures are as implemented.

## Python (`tpa_index`)

```python
# config.py (new)
required_threshold(pipeline, model, label, stage_hint="predict") -> float
#   calibrated per-model threshold; ConfigError naming the label, the model and `tpa-index calibrate --stage <hint>`
#   if missing (never falls back to label_threshold)
# load_pipeline() also validates insights: (rule grammar, learned:, labels_source)

# view.py (new): one row per official segment; no filenames, no transcript text
main_model(predictions) -> str | None                # most common non-rules model (the visual model)
current_phases() -> dict[str, str]                   # video_id -> phase from config/videos.local.yaml when it exists
                                                     # (validated); {} otherwise. Never the example config.
check_inputs(kind, pipeline, labels, root=None) -> None
#   kind "evaluate" | "insights": required official stages cover the videos in scope (predict or moments, audio,
#   people; plus transcript and expression for insights), every calibrated visual threshold exists, and (evaluate)
#   eval blind labels exist; one ConfigError listing every problem
segment_view(pipeline, labels, root=None) -> pandas.DataFrame
#   segment_id, video_id, split (stored runs), phase (local videos config, else stored runs), phase_source,
#   start_s, end_s, slice_s (non-overlapping), is_idle, person_count, has_speech, faces_detected, transcript_chars
#   (a count), strengths {group: {label: score/threshold}}, model_cleared {group: set}, model_primary,
#   model_strength, uncertain_flag, needs_review, no_call, moment_id, audio_ok {label: bool|None}, expr_top,
#   visual_scores, audio_scores, expr_scores (raw scores, used as learned features)
#   (pandas 3: missing strings and numbers read back as NaN; consumers use pd.isna)
human_labels(labels, pipeline, root=None) -> pandas.DataFrame
#   segment_id, video_id, queue (main|topup), skipped, primary, activities, categories, actors, indicators
verified_labels(view, labels, pipeline, root=None) -> dict[str, dict[str, set]]
#   segment_id -> {group: verified set}: a blind label verifies every group; otherwise review decisions on the
#   containing moment adjust the model's set for the groups they touch

# rules.py (new): the insights.fusion / insights.indicator_rules grammar
context(pipeline) -> dict
evaluate_condition(cond, seg, ctx, fired=None) -> bool | None    # None = signal not available
apply_fusion(view, fusion_rules, ctx) -> (DataFrame, {label: {"fired", "not_applicable"}})
indicator_estimates(view, rules, ctx) -> DataFrame               # segment_id, indicator, fired (bool|None)

# evaluate.py (new)
eval_frame(view, human) -> (DataFrame, counts)       # main-sample, non-skipped, eval-split; positional index
label_metrics(H, M, names, min_support) -> (rows, macro_f1 | None, excluded)
primary_metrics(human_primary, model_primary) -> dict   # accuracy, accuracy_real_activity, kappa, confusion
category_sides(df) -> (human, model, n_excluded)        # one category per side; 2+ human categories excluded
kappa(a, b) -> float | None ;  kappa_sentence(k) -> str
uncertainty_metrics(df) -> dict
block_ids(df, block) -> ndarray                      # shared by the bootstrap and the learned CV folds
block_bootstrap(df, stat_fn(idx), block, resamples, seed) -> (low, high, n_valid)
system_metrics(system, df, labels, pipeline) -> {"rows", "headline", "confusion"}
signal_indicator_table(df, estimates, rules, pipeline, system="zero-shot") -> list[dict]
topup_precision(view, human, labels) -> list[dict]
failure_gallery(df, n) -> list[dict]
run_evaluate(pipeline, labels, root=None) -> dict    # three systems, learned_predictions, limitations

# learned.py (new): logistic regressions on official per-segment signals (no embeddings)
features(view, labels, pipeline) -> (ndarray, column names)   # missing signal -> 0 + "<source>_missing" flag
cross_validated(df, labels, pipeline) -> (DataFrame, folds)   # grouped K-fold over block_ids; each row predicted by
                                                              # a model that never saw its block
fit_final(labeled, view, labels, pipeline) -> DataFrame       # every official segment, tag "model estimate (learned)"
learned_view(view, final) -> DataFrame               # the view with model columns replaced (insights fallback)

# indicators.py (new)
tag(verified_share, model_tag="model estimate (unvalidated)") -> str
segment_table(view, verified, estimates, labels) -> DataFrame # labels (verified first) and time class per segment
time_breakdown(seg, scopes) -> list[dict]
headline(rows, phases_present) -> list[dict]          # per-hour minutes indicators, early vs (mid) vs showcase
run_insights(pipeline, labels, root=None) -> dict     # labels_source learned (default) refits fit_final

# report.py (new)
results_root(root=None) -> Path
new_result_dir(kind, root=None) -> Path               # OUTPUT_DIR/results/{kind}-{ts}-{hex}
write_result(kind, payload, root=None) -> Path        # CSV/JSON/charts/summary, manifest.json last (atomic)
latest_result(kind, root=None) -> Path | None         # newest folder with manifest.json (for 006)
show(value, unit) -> str ;  show_change(value, unit) -> str   # display (pp for shares, min/h for per_hour)
```

## CLI

| command | does | exit codes |
|---|---|---|
| `tpa-index evaluate [--dry-run]` | checks inputs; computes Part 1 for three systems (zero-shot, + signal rules, learned cross-validated): metrics with block-bootstrap CIs, kappa, ablation, signal-based indicators, top-up precision, failure gallery, learned predictions; writes an `evaluate-*` folder; prints headline numbers with CIs | 0; 3 missing official stage, threshold or eval labels; 2 guard |
| `tpa-index insights [--dry-run]` | checks inputs; model fallback per `insights.labels_source` (learned refits the final model); computes the youth-only indicators per video and phase (phase from the local videos config, stored fallback) and the per-hour headline; writes an `insights-*` folder; prints the headline with tags | 0; 3 missing official stage or threshold, an invalid local videos config, or labels_source learned without eval labels; 2 guard |

`--dry-run` computes and prints everything and writes nothing. Both commands print only IDs and aggregate numbers.
