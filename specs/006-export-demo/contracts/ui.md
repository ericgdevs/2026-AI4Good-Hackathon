# Contract: App screens (judge-facing)

This is one app (Constitution XII). The tabs are: Blind labeling, Review (both 003), **Timeline** (new) and **Insights** (new). Every tab:

- uses the light theme from `config/theme.yaml`;
- shows the credit "Built for The Performance Academy" in the header and the footer;
- shows no logo or image;
- shows no filename, path or transcript text.

## Header and footer

- Header: `TPA video index (local only)` with the credit below it, in secondary text.
- Footer: the credit.

## Timeline tab

| Element | Content |
|---|---|
| Video picker | `Video 4 · showcase` names. The eval videos are locked while blind labeling is unfinished (same rule and message as Review). |
| Moments table | time, primary label (readable), other labels (readable), confidence, review status |
| Segments table (for the selected moment, or the whole video) | time, people, speech, audio cues, expression, led by |
| Expression cell | `AI suggestion, unvalidated: Positive` / `none` / `no faces` / `not available` |
| Led-by cell | `Youth` / `Staff-led (staff-verified)` / `Staff-led (model estimate)` |
| Toggle | "Show staff-led segments", **off by default**. When off, staff-led rows are hidden and a line says how many were hidden. |
| Clip | the selected moment, played through the 003 clip cutter (local, redacted errors) |
| Note | "Signals are observable cues from AI models, unvalidated. Staff-led time is not counted in youth indicators." |

Never shown: transcript words, transcript character counts, segment IDs, filenames, face crops.

## Insights tab

| Element | Content |
|---|---|
| Source line | `Results from insights run of 2026-10-24 21:30 UTC` (from `manifest.created_at`; no folder path) |
| Four charts | performance share, solo moments, youth voice time, positive expression rate. Bars are early → (mid, if present) → showcase, using the chart series colours on white. Each bar is labelled with its value and its exact tag string. |
| Table | indicator (readable), phase, value with unit, tag (verbatim), change vs early |
| Context | staff-led time and idle / setup time (scope `all`), labelled "excluded from youth indicators" |
| Limitations | the manifest's `limitations` list |
| Empty state | `Insights are not available yet. Run the insights step first.` (also used for an unsupported format, with "(unsupported results format)" added) |

Tag strings are shown exactly as in the file: `staff-verified`, `model estimate (unvalidated)`, `mixed (NN% staff-verified)`.

Rows with `status != ok` show the status text (for example `no videos in this phase`) in place of a bar.

## Review tab (003, small change)

- The observation picker lists every observation type from `labels.yaml`, so a staff-added type appears after a restart.
- Label names in the AI-labels panel use readable names. The stored values are unchanged.

## Blind labeling tab

Unchanged: the checklist text, order and shortcuts are byte-identical to before this feature.
