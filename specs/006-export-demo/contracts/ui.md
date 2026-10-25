# Contract: App screens (judge-facing)

This is one app (Constitution XII). The tabs are: Blind labeling, Review (both 003), **Timeline** (new) and **Insights** (new). Every tab:

- uses the light theme from `config/theme.yaml`;
- shows the credit "Built for The Performance Academy" in the header and the footer;
- shows no logo or image;
- shows no filename, path or transcript text.

## Redesign (Draft 3 · Scorecard)

The tabs are, in order: **Scorecard** (opens first), **Search moments** (replaces the Timeline layout), **Review queue** (a three-pane layout), **Process a video**, **Blind labels**, **Export** and **Help**. The tables below for Timeline and Insights describe the data each one shows; the Scorecard and Search tabs show the same data through `demo.InsightsController` and `demo.DemoController`.

- Top bar in `accent_deep` with the app name "This Is ME · TPA", white tab labels and an underlined active tab. White boxes have a 1px `border` and a 6px radius. Every colour comes from `config/theme.yaml`. IBM Plex Sans and Mono come from Gradio's bundled font files (`LocalFont`); there are no external font or CDN URLs.
- Scorecard: compare two phases (default early vs showcase). Youth voice time for the second phase is shown large, with the first phase beside it. "Youth indicators by phase" shows the four headline measures for both phases with their tags exactly as written. The Expression Index is not shown. Facts come from the ablation (the AI version behind the insights numbers) and the time breakdown. Download report runs `word_report.write_word_report(sample=False)`, or shows the empty-state text when there is no insights result.
- Search moments: a local filter (label names, status, "needs review", solo or group from indicator labels, else the most common person count). No transcript. Same eval lock as Review. Staff-led moments (most clips staff-led) are hidden unless the toggle is on. Selecting a result opens it in the Review queue.
- Review queue: the queue (`Not reviewed / Second look / Partly reviewed / Accepted / Edited`), the clip with "AI noticed · suggestions, unvalidated", and "AI suggests X · N× above its cut-off" (never a percentage). Activity checkboxes: unticking an AI label saves a reject, and ticking a new one saves a staff add. Accept or Edit also accepts the other ticked AI labels. Observation buttons save `expression_*` observations, and the note is saved as `reviewer_note`. Shortcuts A / E / R / D and ↓ work outside text fields. Element IDs `review-reminder`, `review-add-obs`, `review-obs` and `review-note` are kept.
- Process a video: one video at a time on the laptop models. A live preview shows a sampled frame about once a second, pixelated inside face boxes and over the head area of each person box, in memory only. With no face blur or no person detection, no image is shown. Runs go to `local_runs.yaml`, never official; their moments are view only in Search and Review.
- Help: the `config/help.yaml` questions, searchable. A Help drawer on every tab asks Ollama on 127.0.0.1 only, with the matched help entries, the question and the tab name, never app data.

## Header and footer

- Header: `TPA video index (local only)` with the credit below it, in secondary text.
- Footer: the credit.

## Timeline tab

| Element | Content |
|---|---|
| Video picker | `Video 4 · showcase` names. The eval videos are locked while blind labeling is unfinished (same rule and message as Review). |
| Moments table | time, primary label (readable), other labels (readable), confidence, review status |
| Segments table (for the selected moment, or the whole video) | time, people (person detection, counts only), speech, sounds, expression, led by. Built from 005's per-segment view (`view.segment_view`), so audio cues use 005's fixed `insights.audio_threshold`. |
| Expression cell | `AI suggestion, unvalidated: Positive` / `none` / `no faces` / `not available` |
| Led-by cell | `Youth` / `Staff-led · staff-verified` / `Staff-led · model estimate (unvalidated)` / `Idle / setup`. The class is 005's `indicators.segment_table` `cls` on the zero-shot view (005's insights folders hold aggregates only, and no model is trained while the app runs), so the tag is 005's exact string. |
| Toggle | "Show staff-led segments", **off by default**. When off, staff-led rows (both kinds) are hidden and a line says how many were hidden. Idle rows stay visible. |
| Clip | the selected moment, played through the 003 clip cutter (local, redacted errors) |
| Note | "Signals are observable cues from AI models, unvalidated. Staff-led time is not counted in youth indicators." |

Never shown: transcript words, transcript character counts, segment IDs, filenames, face crops.

## Insights tab

| Element | Content |
|---|---|
| Source line | `Results from insights run of 2026-10-24 21:30 UTC` (from `manifest.created_at`; no folder path) |
| Four charts | one per `headline.csv` indicator in file order (005: `performance_share`, `solo_moments_minutes_per_hour`, `youth_voice_minutes_per_hour`, `positive_expression_rate`), values formatted with 005's `report.show`. Bars are early → (mid, if present) → showcase, using the chart series colours on white. Each bar is labelled with its value and its exact tag string. |
| Table | indicator (readable), phase, value with unit, tag (verbatim), change vs early |
| Context | staff-led time and idle / setup time (scope `all`), labelled "excluded from youth indicators" |
| Limitations | the manifest's `limitations` list |
| Expression Index | 005's optional example index (`expression_index.csv`, phase rows), **below** the indicators and never instead of them: phase, index (0-100), `label` and the weakest-component tag, as written, in 005's order (never ranked), with 005's caveat text (`report.INDEX_CAVEAT`). Empty if the file is missing. No variant without performance share (005 provides none). |
| Ablation | from the newest complete `evaluate-*` folder (`ablation.csv`): system, segments, primary accuracy and macro-F1 activity with 95% CI, kappa primary, tag `metric`. Empty if there is no evaluate result. The evaluate folder is never used as the insights source. |
| Empty state | `Insights are not available yet. Run the insights step first.` (also used for an unsupported format, with "(unsupported results format)" added) |

Tag strings are shown exactly as in the file: `staff-verified`, `model estimate (learned)` (005's default fallback), `model estimate (unvalidated)`, `mixed (NN% staff-verified)`.

Rows with `status != ok` show the status text (for example `no videos in this phase`) in place of a bar.

## Review tab (003, small change)

- The observation picker lists every observation type from `labels.yaml`, so a staff-added type appears after a restart.
- Label names in the AI-labels panel use readable names. The stored values are unchanged.

## Blind labeling tab

Unchanged: the checklist text, order and shortcuts are byte-identical to before this feature.
