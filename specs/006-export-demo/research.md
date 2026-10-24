# Research: 006 Export, Demo and Mac Benchmark

All decisions reuse what 001–004 already built. No new dependencies.

## R1. Current label state for the export

- **Decision**: Reuse `review.latest_state(read_reviews(root), "review")`. It is keyed by (target_id, label) and takes the newest row by (created_at, review_id), so ties are broken the same way on every run (FR-004). Only `mode = review` rows on moment targets count. Blind rows are segment targets and feed 005, not the export.
- **Rationale**: Review mode already uses this function to show state, so the export and the app can never disagree.
- **Alternatives**: A DuckDB window query was rejected: it would be a second implementation of the same rule.

## R2. Building one record per moment

- **Decision**: One pure function, `build_records(moments, state, labels, verified_only=False) -> list[dict]`, with no I/O. It follows FR-005 to FR-010:
  - **Primary**: the latest row for (moment, primary label) decides. None → `pending`; accept → `accepted`; edit → `edited`, and `primary_label` becomes `new_label`; reject → `rejected`; defer → `deferred`.
  - **Secondaries and indicators**: AI labels with reject are dropped, edits are replaced by `new_label`, and the rest are kept. Then staff `add` rows with role secondary or indicator are appended, unless a later row for the same key rejects them. Lists are de-duplicated and keep their order.
  - **Primary role**: review mode never adds a label with role primary (`role_of` is called without a chosen primary), so added labels only ever go into the lists.
  - **`source`**: `staff` if the primary was edited or any staff-added label survived, else `ai`. Observations do not change `source`, because `contextual_observation` is staff by definition.
  - **`contextual_observation`**: the rows with role `observation`, in time order, as `type` or `type: note`, joined with `; `, exactly as stored (FR-008, Clarification 8).
  - **`guideline_version`**: from the primary's deciding row, else the current config version.
  - **`reviewer_note` / `review_date`**: from the newest row on the moment whose action is not `reject` (FR-010). `review_date` is the UTC date `YYYY-MM-DD`.
  - **`--verified-only`**: drops records whose decision is `rejected`.
- **Rationale**: A pure function is easy to test exhaustively and is shared by the CLI export and the Devpost sample (FR-014: same logic).

## R3. Export files

- **Decision**: Files go under `{OUTPUT_DIR}/exports/`:
  - `annotations-{YYYYmmddTHHMMSS}-{4hex}-g{guideline}.json` and `.csv`, created exclusively, so they are never overwritten;
  - `annotations-latest.json` and `.csv`, replaced atomically (temp file plus rename).

  JSON is a plain array of records, so each record has exactly the 13 fields. CSV uses the standard library `csv` module with full quoting rules. List fields are joined with `;` (labels match `[a-z0-9_]+`, so they never contain `;`). Numbers: `start_s` and `end_s` have 2 decimals, `confidence` has 3.
- **Rationale**: The 4-hex suffix handles two runs in the same second (edge case), and matches the runs folders. A plain array keeps "exactly these fields" literally true.
- **Alternatives**: A wrapper object with metadata was rejected, because a judge reading the file would see extra fields.

## R4. Devpost sample from mock data only

- **Decision**: `tpa-index export --sample` runs in two steps:
  1. It generates the mock dataset (`mock.generate(out_root=tmpdir, seed=0)`) in a temporary folder, takes the first 20 moments, and builds scripted review rows. The script guarantees every decision value, a rejected secondary, an edited secondary, one staff-added label, one observation and one label reviewed twice. Staff names are `mock-staff-a`/`b`, which never reach the export.
  2. It calls `build_records` and writes `docs/samples/annotation_export_sample.json` and `.csv`, with fixed (non-timestamped) names.

  It never touches OUTPUT_DIR or the real reviews database: the mock goes to a temp root, and the reviews are built in memory, not saved.
- **Rationale**: This satisfies FR-014 and Constitution XIII (public samples from mock data). A test regenerates the sample and compares it with the checked-in copy, so it cannot drift.
- **Alternatives**: Reusing `mock._reviews` was rejected: it is random and does not guarantee coverage.

## R5. Mac benchmark

- **Decision**: `tpa-index benchmark --video-id X [--videos PATH]` calls the existing commands in-process through `cli.main([...])`, in this order:
  1. `segment`
  2. `predict`
  3. `signals --stages transcript,audio,people,expression --force`
  4. `moments`

  Each step gets `--video-id X`. Every stage already writes its own runs row with hardware, model, runtime and peak memory (`track_run`). The command finds its new run IDs by diffing `list_runs()` before and after, and never edits `official_runs.yaml` (FR-028). The summary is written to `{OUTPUT_DIR}/benchmarks/benchmark-{ts}-{hex}.json` and `.md`, and printed.
- **Per video-hour**: `runtime_s / (duration_s / 3600)`. `duration_s` comes from the `videos.parquet` in each run folder (every stage writes it).
  - The HPC side uses the official runs of each stage, summed: total runtime divided by total video-hours of the videos they cover. Their hardware string is shown as recorded.
  - The end-to-end total is the sum per video-hour over the stages present, with peak memory the maximum. It is marked "incomplete: missing …" when a stage has no row on either side.
- **Failure handling (FR-030)**: a failing step is caught and recorded in the summary as "failed", and the next steps still run. `signals` already isolates per-stage failures.
- **Known behaviour to flag**:
  - 004 stages and `moments` work on the **official** segments and predict output for that video. The promo must already be in the official runs (synced from HPC); otherwise those stages stop with "no official segments". Timings stay representative, because the same work is done.
  - Per-run model load time is amortised over a short promo, so the Mac's per-hour figure is pessimistic. The summary says so.
- **Rationale**: This adds no new pipeline code paths. Each stage runs exactly as it does in production, which is what the benchmark should measure.

## R6. Theme in one config place

- **Decision**: A new file, `config/theme.yaml`, with `background`, `surface`, `text`, `text_secondary`, `accent`, `accent_deep`, `chart_series` (a list of three) and `credit` ("Built for The Performance Academy"). A new loader, `config.load_theme()`, validates `#RRGGBB` values and checks WCAG contrast on load: text, text_secondary, accent and accent_deep against both background and surface, and white against accent, all ≥ 4.5. `app.theme()` builds the Gradio theme from it with `.set(...)` variables (body background, text colours, primary button, selected tab, headers). Charts take `chart_series` from the same loader.
- **Rationale**: A separate file keeps `pipeline.yaml` untouched, so the 005 merge (which adds an `insights:` block there) has no conflict. Checking contrast in the loader means a staff palette edit that breaks AA fails at startup, not silently.
- **Alternatives**:
  - An `app.theme` block in `pipeline.yaml` was rejected to keep the 005 merge clean.
  - A CSS file was rejected: Gradio theme variables already cover the needed slots.
- **Test**: no palette hex value appears in any tracked file under `src/` or `config/` other than `theme.yaml` (FR-031b).

## R7. Readable label names and plain video names

- **Decision**:
  - **Labels**: `display_label(name)` replaces `_` with a space and capitalises the first letter (`lyricism_spoken_word` → "Lyricism spoken word"). If a label in `labels.yaml` has an optional `display:` key, that text is used instead. Only the demo, insights and review panels use display names. The blind checklist keeps its current text, so its snapshot test stays byte-identical.
  - **Videos**: `video_name(video)` gives `"Video {n} · {phase}"`, where n is the trailing number of `video_id` without leading zeros (`v04` → 4, `mock04` → 4), and `· {kind}` is appended only when two shown videos would otherwise get the same name. `phase` `unknown` reads "phase not set". No filename is ever read for display.
- **Rationale**: This needs no new config for videos (Clarification 4). Labels work without config, with an optional key for nicer wording.
- **Note**: Adding `display:` keys would change `labels.yaml`. The guideline fingerprint (R8) leaves `display` out, so it needs no guideline bump.

## R8. Refusing a new observation type without a guideline bump (FR-026)

- **Decision**: A new tracked file, `config/guideline_history.yaml`, maps each released guideline version to a fingerprint: sha256[:12] of the sorted (name, group, source) of every label, plus the sorted observation-type names. Prompts, colours, descriptions and `display` are left out. `config.check_guideline(labels)`:
  - refuses (ConfigError) when the current version is recorded with a different fingerprint, naming the missing bump;
  - when the version is not recorded yet, prints the line to add.

  It is called by `tpa-index check` and by `app.launch()`, not by `load_labels()`.
- **Rationale**: Existing tests load renamed vocabularies through `load_labels(path)` without bumping. Putting the check in `load_labels` would break them, and FR-026 only asks for the app and config check. The file is seeded with `0.5.0` and the current fingerprint.
- **Alternatives**:
  - Comparing against stored reviews was rejected: reviews don't store the vocabulary.
  - A test-only snapshot was rejected: staff don't run tests.

## R9. Timeline signals view

- **Decision**: A new `Timeline` tab with a `DemoController` in a new `demo.py`. It reads through the same official paths the review tab uses (Constitution XII):
  - `read_official("moments")`;
  - `schema.read_signals()` for person count, faces and has_speech. `transcript_chars` is dropped right after reading and never reaches a view model;
  - `read_official("predictions", stage="audio")` and `stage="expression"`;
  - the main predictions, for the actor and instruction scores.

  Thresholds come from `config.threshold(pipeline, model, label)`. Videos come from `review.review_videos()`, so eval videos stay hidden until blind labeling is finished (the same lock as review mode). Per segment it shows:
  - time;
  - people (`person_count`, or "not available");
  - speech (Yes / No / not available);
  - audio cues (display names of audio labels at or above threshold, or "none" / "not available");
  - expression ("AI suggestion, unvalidated: Positive", "none", or "no faces");
  - led by (Youth / Staff-led (staff-verified) / Staff-led (model estimate)).

  Moments are listed above it with readable labels, confidence and review status.
- **Staff-led rule (FR-018)**:
  - **Verified**: for the segment, the latest blind-mode actor labels on that segment, else the review-mode actor labels on the moment that contains it. Staff-led = staff facilitation present and participant absent.
  - **Model only**: (`actor_staff_facilitation` ≥ threshold OR `instruction` ≥ threshold) AND `actor_participant` < threshold.
  - The three label names (`actor_staff_facilitation`, `instruction`, `actor_participant`) are module constants in `demo.py`. They are not in config, so `pipeline.yaml` stays untouched for the 005 merge. A startup check confirms each exists in `labels.yaml`, and fails with a plain message if the vocabulary changes. If 005 puts the rule's labels in its `insights:` block, 006 reads them from there after the merge.
- **Rationale**: 006 cannot import 005's code (it is not merged and must stay file-coupled), and 005's files are aggregates, not per segment. So the rule is restated in one small function with a test table that copies 005's cases. Flagged for after the merge: if 005 exposes the rule as a function, switch to it.

## R10. Insights page

- **Decision**: An `InsightsController` in `demo.py`:
  - It finds the newest complete folder: under `{OUTPUT_DIR}/results/`, the `insights-*` folders that contain `manifest.json`, sorted by name, taking the last.
  - It refuses (with "Insights are not available yet") a missing folder, `schema_version != 1`, or `headline.csv` missing or lacking the needed columns.
  - It reads `headline.csv` by column name (`indicator, phase, value, unit, tag, verified_share, change_vs_early`).
  - It draws four matplotlib bar charts (already locked) on a white background. Series colours come from `theme.chart_series`; the phase order is early, (mid if present), showcase. Each bar's label is `value + tag` in text colour.
  - Beside the charts is a plain table with the same numbers and the exact tag strings.
  - It shows `time_breakdown.csv` "staff-led time" and "idle / setup time" for scope `all` as context, labelled as excluded from youth indicators, plus the manifest's `limitations`.

  The headline indicators are `performance_share`, `solo_moments_count` (with `solo_moments_minutes` if present), `youth_voice_minutes` and `positive_expression_rate`. Rows with `status != ok` show 005's status text and no bar.
- **Rationale**: The tags come straight from the file (FR-022). Drawing from the CSV, not 005's PNGs, lets every number show its tag and use the palette.
- **Alternatives**: `gr.BarPlot` was rejected: it gives weaker control over per-bar text and the palette than matplotlib, which 005 already uses.

## R11. Transcript and filename safety in the UI

- **Decision**: View models are plain dicts built in `demo.py`, from an allow-list of columns. `transcript_chars` and any transcript JSON are never read by `demo.py`.
- **Tests**:
  - render the controllers' outputs (and the Gradio config of `build_app`) for a mock video whose run folder has a planted transcript sentinel and large `transcript_chars`, and assert that neither the sentinel nor the count appears;
  - scan for every mock filename and path (`redact` patterns).
- **Rationale**: An allow-list beats a block-list for "never show".

## R12. Keeping the 005 merge easy

- **Decision**:
  - `cli.py`: add `export` and `benchmark` subparsers in one contiguous block right before `args = parser.parse_args(argv)`, with their handlers as new functions at the end of the handler section. `_check` gains one line (`check_guideline`).
  - `pipeline.yaml`: not changed.
  - `app.py`: two new tabs appended after Review, plus `theme()` and launch changes.

  005 adds `evaluate` and `insights` subparsers and an `insights:` block in `pipeline.yaml`, so the only likely conflict is neighbouring lines in `main()`, which is trivial to resolve.
