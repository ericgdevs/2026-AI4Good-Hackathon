# Feature Specification: Official Export, Judge Demo and Mac Benchmark

**Feature Branch**: `006-export-demo`

**Created**: 2026-10-24

**Status**: Clarified (spec only; implementation waits until 005-insights-eval is merged)

**Input**: User description: "006-export-demo: the official annotation export, a judge-facing demo view, and the Mac benchmark. SPEC ONLY for now; implementation waits until 005-insights-eval is merged." (full description in the /speckit-specify request of 2026-10-24: five deliverables, export fields, demo rules, staff-defined labels, Mac benchmark, UI polish, and the privacy and data-access rules repeated below.)

## Context

- `main` holds 001 (contracts), 002 (visual pipeline, calibrate, moments, official runs), 003 (blind labeling and review mode in the local app), 004 (per-segment signals: transcript, audio cues, person count, expressions), the shared blind queue fix and the job B speed-ups.
- 005-insights-eval (evaluation metrics and youth-only indicators) is being specified in parallel (`specs/005-insights-eval/spec.md` in the 004 worktree, still in clarification). 005 writes each run to a new folder under the outputs root with `metrics.json`, `metrics.csv`, `indicators.json`, `indicators.csv`, `failure_gallery.csv`, chart images and `summary.md` (005 US4-AS1, FR-026). Each indicator record has indicator, scope (video ID or phase), value, unit, time base and tag. A tag is "staff-verified", "model estimate (unvalidated)" or "mixed (x% staff-verified)" (005 FR-018). 006 reads those files only and never depends on 005's code.
- Track A (model-first): anything judges see must be clear and easy to read, not elaborate. Light theme only.
- What already exists and is reused, not rebuilt: the moments table (`video_id`, `start_s`, `end_s`, `primary_label`, `secondary_labels`, `indicators`, `confidence`), the append-only reviews table (accept, edit, reject, defer, add; roles primary, secondary, indicator, observation; source ai or staff; guideline version; note; time), the runs table (stage, model, hardware, runtime, peak memory), `read_official` for result-facing reads, `--video-id` on the segment, predict, signals and moments commands, the observation-type list in the label config (003 FR-017a: staff may add types, adding one bumps the guideline version), filename and path redaction (001), and the 127.0.0.1-only app (003).

### Rules (override everything)

- While building and testing, nobody (person or assistant) reads the data folder, `~/tpa-data`, `/Volumes/T7`, the real reviews database, `config/videos.local.yaml` or any `config/*.local.yaml`. Tests use mock data and synthetic clips only. The app is never started on real data during development, and there is no ssh or rsync to HPC.
- No identity: no names, no faces shown as identity, no individual youth tracked or counted as a person over time.
- No transcript text anywhere in this feature: not in the export, the sample, the demo, logs or errors. Speech appears only as yes/no.
- No clinical or diagnostic language. Expressions are observable cues, always shown as "AI suggestion, unvalidated".
- No filenames or paths anywhere a user or judge can see them.

## Clarifications

### Session 2026-10-24

- Q: How do per-label review states become one export record? → A: `review_decision` follows the primary label: accepted; edited (the new label becomes `primary_label`); rejected; deferred; pending if never reviewed. Rejected secondary labels are dropped. Staff-added labels are kept, and count as staff-sourced. Moments whose primary was rejected stay in the export with `rejected`, because it is an audit record. A `--verified-only` flag leaves them out.
- Q: What makes a segment staff-led? → A: The same rule as 005. Verified labels decide where they exist. For model-only segments, staff-led = (`actor_staff_facilitation` OR `instruction` clears its threshold) AND `actor_participant` does not. Those are tagged "model estimate".
- Q: Which 005 contract does the insights page use? → A: 005's output files (JSON/CSV plus tags). The headline charts are early vs showcase for performance share, solo moments, youth voice time and positive expression rate. Mid is shown only if present. Anything 005 has not decided yet is listed as a dependency, not guessed.
- Q: Plain video names? → A: "Video 4 · showcase" style: the number from the video ID plus the phase, plus kind where useful. No new config.
- Q: Devpost sample content? → A: About 20 mock moments that deliberately include every `review_decision` value, at least one staff-added label and at least one observation.
- Q: Benchmark scope? → A: The promo is chosen at run time with `--video-id` and is not stored in the repo. The benchmark includes the moments stage and an end-to-end total. Benchmark runs never go into `official_runs.yaml`.
- Q: Export file naming? → A: Each run writes a timestamped JSON/CSV pair with the guideline version in the name, plus a fixed "latest" copy.
- Q: What happens to observation types staff later remove? → A: The export keeps the stored value as-is.
- Q: Theme colours? → A: A light theme using The Performance Academy's palette as accents, defined in one configuration place. Background #F8F7F8 / white. Text #201C25, secondary text #6F717A. Primary accent #7851B9 (buttons, selected tabs, headline chart series), deep accent #4100AD (headers, emphasis). Chart series #7851B9, #0099B3, #FF4255; teal and coral are fills and borders only. Text contrast meets WCAG AA. Header/footer text "Built for The Performance Academy", with no TPA logo or images.

## User Scenarios & Testing *(mandatory)*

The users are TPA program staff (who review and export), hackathon judges (who watch the demo and read the sample export) and the three-person team (who run the export and the Mac benchmark). Youth appear in footage but are never identified. Every scenario can be shown with mock data and synthetic clips.

### User Story 1 - Export the official annotations (Priority: P1) 🎯

A team member or staff member runs one command and gets the case's official annotation file in both JSON and CSV. There is one record per moment, with exactly the fields the case asks for. Each label shows its current verified state: the latest review for that moment and label. The same logic, run on mock data, produces the sample export submitted to Devpost.

**Why this priority**: The export is required by the case. Without it the submission is incomplete, whatever the demo looks like.

**Independent Test**: Generate mock moments and mock reviews (accepted, edited, rejected, deferred, staff-added, unreviewed, a rejected secondary, and a label reviewed twice). Run the export with and without `--verified-only`, and check both files field by field against the expected records.

**Acceptance Scenarios**:

1. **Given** official moments and their reviews, **When** the export runs, **Then** it writes one JSON file and one CSV file with one record per moment. Every record has exactly these fields in this order: `video_id`, `start_s`, `end_s`, `primary_label`, `secondary_labels`, `indicators`, `contextual_observation`, `confidence`, `source`, `review_decision`, `guideline_version`, `reviewer_note`, `review_date`. No other field appears.
2. **Given** a label on a moment reviewed more than once, **When** the export runs, **Then** only the latest review for that (moment, label) counts.
3. **Given** a moment whose primary label was edited, **When** exported, **Then** `primary_label` is the edited label, `review_decision` is `edited` and `source` is `staff`.
4. **Given** a moment with no review at all, **When** exported, **Then** `source` is `ai`, `review_decision` is `pending`, and `reviewer_note` and `review_date` are empty.
5. **Given** a moment whose primary label was rejected, **When** the export runs normally, **Then** the moment is included with `review_decision` = `rejected`. **When** it runs with `--verified-only`, **Then** the moment is left out.
6. **Given** a rejected secondary label, **When** exported, **Then** it is missing from `secondary_labels`. **Given** a staff-added label, **When** exported, **Then** it appears in its list and the record's `source` is `staff`.
7. **Given** a moment with a staff observation, **When** exported, **Then** `contextual_observation` holds the observation type (and its note). Moments without one leave it empty. A type later removed from config is exported exactly as stored.
8. **Given** an export run, **When** it finishes, **Then** it has written a new JSON/CSV pair whose names carry the run time and guideline version, and has refreshed a fixed "latest" JSON/CSV pair. Earlier timestamped pairs are never overwritten.
9. **Given** mock data only, **When** the sample export is generated, **Then** it has about 20 moments covering all five `review_decision` values, at least one staff-added label and at least one observation. It contains no real video filename, no path and no transcript text, and its video IDs are mock IDs.
10. **Given** any export, **When** read, **Then** no reviewer name or any other personal identifier appears in any field.

---

### User Story 2 - Judges walk through one video's timeline (Priority: P1)

In the local app, a judge (guided by the team) picks a video by a plain name such as "Video 4 · showcase" and sees its moments laid out along a timeline. For each segment they can see the observable signals: how many people are visible, which audio cues are present, whether there is speech (yes/no), and an expression suggestion clearly marked "AI suggestion, unvalidated". Segments led by an instructor or staff member are marked as such. No transcript text, filename or path is shown anywhere.

**Why this priority**: This is how judges see that the model output is concrete, reviewable and privacy-safe. It reuses 003's timeline and 004's signals.

**Independent Test**: Load mock moments, mock signals and synthetic clips, and open the demo view for one mock video. Check what is on screen against the mock data. Scan every rendered element for transcript text, filenames and paths.

**Acceptance Scenarios**:

1. **Given** a video with moments and 004 signals, **When** the demo view opens it, **Then** each moment is shown on a timeline with its primary label, secondary labels and confidence, using readable label names.
2. **Given** a segment, **When** it is shown, **Then** it displays person count, audio cues present, speech yes/no and an expression suggestion labelled "AI suggestion, unvalidated".
3. **Given** a segment whose signal stage did not run, **When** shown, **Then** that signal reads "not available" rather than zero or "no".
4. **Given** transcript data exists for a video, **When** the demo view shows it, **Then** no transcript words or character counts appear on screen, in tooltips, in downloads or in error messages.
5. **Given** a segment with a verified staff-only actor label, **When** shown, **Then** it is marked staff-led. **Given** a model-only segment where `actor_staff_facilitation` or `instruction` clears its threshold and `actor_participant` does not, **When** shown, **Then** it is marked staff-led and tagged "model estimate". In both cases it is not counted in youth indicators.

---

### User Story 3 - Judges see the headline insights (Priority: P2)

On an insights page in the same app, judges see 005's headline results as simple charts: early phase vs showcase phase for performance share, solo moments, youth voice time and positive expression rate. A mid-phase value appears only if 005's files contain one. Every number shows the tag 005 gave it. Youth indicators exclude staff-led segments by default.

**Why this priority**: It turns per-moment output into a program-level story, but depends on 005's result files.

**Independent Test**: Place mock 005 result folders (written to 005's file contract) in a test output root. Open the insights page and check each chart, number and tag against the newest folder's files.

**Acceptance Scenarios**:

1. **Given** 005 result folders, **When** the insights page opens, **Then** it reads the newest complete folder and shows four headline charts (performance share, solo moments, youth voice time, positive expression rate), each with early and showcase values.
2. **Given** each number shown, **When** a judge reads it, **Then** it carries the tag from 005's file exactly ("staff-verified", "model estimate (unvalidated)" or "mixed (x% staff-verified)").
3. **Given** 005's files include mid-phase values, **When** the page opens, **Then** mid is shown as a third value. **Given** they do not, **Then** no mid value or empty placeholder appears.
4. **Given** a phase with no videos (005 says "no videos in this phase"), **When** the page opens, **Then** that side shows 005's wording and no change is drawn.
5. **Given** 005's files are missing or unreadable, **When** the page opens, **Then** it shows a plain message saying insights are not available yet, and the rest of the app still works.
6. **Given** the page, **When** read, **Then** it states in plain words that youth indicators exclude staff-led time. No clinical, diagnostic or individual-judgment wording appears.

---

### User Story 4 - Staff add their own observation label (Priority: P2)

A staff member adds a new observation type (for example "interview") with a short description to the label config, and bumps the guideline version. After the app restarts, the new type is offered in review mode when adding an observation. The blind checklist and the guideline-tested label groups are exactly the same as before.

**Why this priority**: It shows judges that staff, not engineers, own the vocabulary for context notes, without putting evaluation results at risk.

**Independent Test**: With a mock label config, capture the blind checklist and the review-mode observation list. Add "interview", reload, and compare both.

**Acceptance Scenarios**:

1. **Given** a new observation type in config with a bumped guideline version, **When** review mode loads, **Then** the new type is offered in the observation picker.
2. **Given** the same change, **When** blind mode builds its checklist, **Then** the checklist (groups, labels and order) is identical to the one before the change.
3. **Given** a new observation type added without bumping the guideline version, **When** the app or config check runs, **Then** it refuses with a plain message naming the missing bump.
4. **Given** an observation saved with the new type, **When** the export runs, **Then** it appears in `contextual_observation`, and the guideline version recorded is the one current when it was saved.

---

### User Story 5 - Prove the pipeline fits a Mac (Priority: P2)

A team member runs the full pipeline on the Mac for one short promo, chosen at run time with `--video-id`: segment, predict, the four 004 signal stages (transcript, audio, people, expression) with the Mac model sizes, and moments. Each stage records its runtime and peak memory in the runs table as usual. A summary then compares Mac and HPC per hour of video, for each stage and end to end.

**Why this priority**: It is the "fits constrained hardware" proof for judges and for TPA, who will not have HPC.

**Independent Test**: Run the benchmark on a synthetic mock clip and check that each stage wrote a runs row with runtime and peak memory. Then feed mock Mac and HPC runs rows to the summary and check the per-video-hour numbers by hand.

**Acceptance Scenarios**:

1. **Given** a video ID passed at run time, **When** the benchmark runs on the Mac, **Then** each of the seven stages (segment, predict, transcript, audio, people, expression, moments) writes a runs row with hardware, model, runtime and peak memory. The video ID is not written to any file in the repo.
2. **Given** Mac and HPC runs rows, **When** the summary is produced, **Then** it shows runtime per video-hour and peak memory on each machine, per stage and as an end-to-end total. It names the model used on each machine and notes that the Mac uses smaller models.
3. **Given** a stage with no HPC run to compare against, **When** the summary is produced, **Then** that row says "no HPC run" instead of a number, and the end-to-end total says which stages it is missing.
4. **Given** benchmark runs, **When** results are read anywhere else, **Then** the benchmark runs are not in `official_runs.yaml` and change no result.
5. **Given** the summary, **When** shown, **Then** it contains no filename or path.

---

### User Story 6 - A clean, readable app for judges (Priority: P3)

Every screen judges see uses a light theme with The Performance Academy's palette as accents, and readable label names (for example "Lyricism / spoken word" rather than `lyricism_spoken_word`). Videos have plain names such as "Video 4 · showcase" instead of raw IDs, and a filename is never shown. The header or footer reads "Built for The Performance Academy". There is no TPA logo or image.

**Why this priority**: Polish matters for judging but adds no new capability.

**Independent Test**: Render each demo screen with mock data. Check the theme colours against the palette, compute the contrast of every text/background pair, and check every label shown and every video name against the expected readable forms.

**Acceptance Scenarios**:

1. **Given** any demo or insights screen, **When** it renders, **Then** it uses the light theme regardless of the browser's dark-mode setting.
2. **Given** a label shown to a judge, **When** rendered, **Then** it uses its readable name.
3. **Given** a video shown to a judge, **When** rendered, **Then** its name is built from the number in its video ID and its phase (plus kind where it helps tell videos apart), never from the filename. No new config is needed.
4. **Given** the palette, **When** any screen renders, **Then**:
   - the background is #F8F7F8 or white;
   - text is #201C25, and secondary text is #6F717A;
   - buttons, selected tabs and the headline chart series use #7851B9;
   - headers and emphasis use #4100AD.
5. **Given** the palette is changed in its one configuration place, **When** the app restarts, **Then** every screen and chart picks up the change, and no colour value is repeated elsewhere.
6. **Given** a chart, **When** rendered, **Then** its series use #7851B9, #0099B3 and #FF4255 in that order. Teal and coral appear only as fills or borders, never as text.
7. **Given** any screen, **When** rendered, **Then** "Built for The Performance Academy" appears in the header or footer, and no TPA logo or image is shown.

### Edge Cases

- A moment's primary label was rejected and nothing replaced it: included as `rejected`, and left out under `--verified-only`.
- Two reviews of the same (moment, label) share the same timestamp.
- A label was edited, then the edit was later rejected.
- A secondary label was deferred: it stays in its list (only rejected labels are dropped).
- A moment has several observations, or the same observation type twice.
- A moment's reviews span two guideline versions.
- An observation type was removed from config after reviews used it: exported as stored.
- A reviewer note contains a comma, quote or line break (CSV escaping), or non-English characters.
- Two export runs in the same second.
- An official moments run covers a video that has no signal runs, or only some of them.
- A segment has faces but none large enough for an expression suggestion.
- A segment has both a verified actor label and model actor scores that disagree: the verified label wins.
- 005's output root holds a partly written folder, or a folder from an older 005 format.
- 005's files report zero staff-verified values, or no mid phase.
- The Mac benchmark runs out of memory on one stage: the stage is logged as failed, the others still run and record, and the end-to-end total is marked incomplete.
- A staff member removes an observation type that existing reviews already use.
- The export runs with no reviews database at all (everything pending).

## Requirements *(mandatory)*

### Functional Requirements

**Export**

- **FR-001**: The system MUST provide one export command that writes the official annotation export as JSON and CSV, built from official moments only.
- **FR-002**: Every export record MUST have exactly these fields, in this order: `video_id`, `start_s`, `end_s`, `primary_label`, `secondary_labels`, `indicators`, `contextual_observation`, `confidence`, `source`, `review_decision`, `guideline_version`, `reviewer_note`, `review_date`. No other field is allowed.
- **FR-003**: The export MUST contain one record per official moment, including moments whose primary label was rejected. A `--verified-only` option MUST leave out rejected-primary moments.
- **FR-004**: The current state of each label MUST be the latest review for that (moment, label). Earlier reviews are ignored. Ties MUST be broken the same way on every run.
- **FR-005**: `review_decision` MUST follow the primary label's current state: `accepted` (accept), `edited` (edit; `primary_label` becomes the new label), `rejected` (reject), `deferred` (defer), or `pending` (no review of the primary label).
- **FR-006**: Within a record, rejected secondary labels and rejected indicators MUST be dropped from their lists. Edited ones MUST be replaced by their new label. Deferred and unreviewed ones MUST be kept. Staff-added labels MUST be included in the list for their role.
- **FR-007**: `source` MUST be `staff` when the primary label was edited, or when the record includes any staff-added label. Otherwise it MUST be `ai`.
- **FR-008**: `contextual_observation` MUST come only from staff observation reviews on that moment, exactly as stored (even if the type was later removed from config). It MUST be empty when there are none. AI output MUST never fill it.
- **FR-009**: `guideline_version` MUST be the version recorded on the review that set `review_decision`, or the current config version for pending records.
- **FR-010**: `reviewer_note` and `review_date` (date only) MUST come from the latest review on the moment that is not a rejected label's review, and MUST be empty when the moment has no reviews.
- **FR-011**: In CSV, list fields (`secondary_labels`, `indicators`) MUST use one documented separator, and text MUST be correctly escaped. In JSON they MUST be arrays.
- **FR-012**: Each export run MUST write a new JSON/CSV pair whose names include the run time and the guideline version, and MUST refresh a fixed "latest" JSON/CSV pair. Earlier timestamped pairs MUST never be overwritten.
- **FR-013**: The export MUST NOT contain reviewer names, filenames, paths or transcript text.
- **FR-014**: A sample export for Devpost MUST be produced from mock data only, by the same export logic. It MUST have about 20 moments that deliberately cover all five `review_decision` values, at least one staff-added label and at least one observation. It is checked into the repo.

**Demo view**

- **FR-015**: The local app MUST gain a demo view that shows, for a chosen video, its moments on a timeline. Per segment it MUST show person count, audio cues present, speech yes/no and the expression suggestion marked "AI suggestion, unvalidated".
- **FR-016**: The demo view MUST show "not available" for any signal whose stage has no official run for that video.
- **FR-017**: The demo view and insights page MUST NOT show transcript text or transcript character counts anywhere, including tooltips, downloads and errors.
- **FR-018**: Staff-led segments MUST be marked in the timeline and excluded from youth indicators by default, using 005's rule. A verified actor label decides where one exists. For model-only segments, staff-led = (`actor_staff_facilitation` OR `instruction` clears its threshold) AND `actor_participant` does not clear its threshold. Model-only staff-led segments MUST be tagged "model estimate".
- **FR-019**: The app MUST stay bound to 127.0.0.1 only, as in 003.

**Insights page**

- **FR-020**: The app MUST gain an insights page that reads the newest complete 005 result folder under the outputs root.
- **FR-021**: The page MUST show four headline charts, each comparing early and showcase phases: performance share, solo moments, youth voice time and positive expression rate. Mid MUST be shown only when 005's files contain mid values.
- **FR-022**: Every number on the insights page MUST show the tag from 005's files exactly ("staff-verified", "model estimate (unvalidated)" or "mixed (x% staff-verified)"). Tags are never inferred by 006, and an untagged value MUST NOT be shown.
- **FR-023**: If 005's files are missing, unreadable or in an unexpected format, the page MUST show a plain "not available yet" message, and the rest of the app MUST keep working.

**Staff-defined labels**

- **FR-024**: A new observation type added to the label config (with a guideline version bump, per 003 FR-017a) MUST appear in review mode's observation picker after an app restart, with no code change.
- **FR-025**: Adding an observation type MUST NOT change the blind checklist (its groups, labels and order) or any guideline-tested label group.
- **FR-026**: A config with a new observation type but an unchanged guideline version MUST be refused with a plain message.

**Mac benchmark**

- **FR-027**: The team MUST be able to run segment, predict, the four 004 signal stages and moments on the Mac for one video chosen at run time with `--video-id`, using the Mac model sizes from config. Each stage MUST record hardware, model, runtime and peak memory in the runs table. The chosen video MUST NOT be written to any file in the repo.
- **FR-028**: Benchmark runs MUST NOT be added to `official_runs.yaml`, and MUST NOT change any result.
- **FR-029**: The system MUST produce a Mac vs HPC summary with, per stage and as an end-to-end total: runtime per hour of video, peak memory, and the model used on each machine. It shows "no HPC run" where there is nothing to compare, and marks the total incomplete when a stage is missing.
- **FR-030**: If one stage fails on the Mac, the failure MUST be recorded and the remaining stages MUST still run where their inputs exist.

**UI polish**

- **FR-031**: Every demo and insights screen MUST use the light theme only.
- **FR-031a**: The theme MUST use The Performance Academy's palette as accents:
  - background #F8F7F8 or white;
  - text #201C25, secondary text #6F717A;
  - primary accent #7851B9 (buttons, selected tabs, headline chart series);
  - deep accent #4100AD (headers, emphasis);
  - chart series #7851B9, #0099B3, #FF4255, in that order.
- **FR-031b**: All palette values MUST be defined in one configuration place. Screens and charts MUST read them from there, and colour values MUST NOT be repeated elsewhere.
- **FR-031c**: Every text/background pair MUST meet WCAG AA contrast (4.5:1 for normal text). Teal #0099B3 and coral #FF4255 MUST be used only as chart fills or borders, never as text, because they fail AA for small text on white.
- **FR-031d**: The header or footer MUST read "Built for The Performance Academy". The app MUST NOT show any TPA logo or image.
- **FR-032**: Labels MUST be shown by readable names. Videos MUST be shown as "Video N · phase" (plus kind where it helps), built from the video ID and metadata, never by filename or path, and with no new config.

**Tests required by this feature**

- **FR-033**: Automated tests (mock data only) MUST prove that:
  - the export has exactly the case's fields, in order;
  - the sample export contains no real filename, path or transcript text, and covers all five decision values;
  - the demo and insights views never show transcript text;
  - staff-led segments, both verified and model estimate, are excluded from youth indicators;
  - the blind checklist is unchanged when a new observation type is added;
  - every themed text/background pair meets WCAG AA, and palette values appear in only one place.

### Key Entities

- **Export record**: one official moment with its current verified labels and review state, in the case's 13 fields. Built from moments plus reviews, and never stored back.
- **Current label state**: the latest review per (moment, label). Derived, never stored.
- **Export file pair**: timestamped JSON and CSV named with the run time and guideline version, plus a fixed "latest" pair.
- **Sample export**: the export from about 20 mock moments for Devpost. A checked-in artifact.
- **Segment signals view**: per segment, person count, audio cues, speech yes/no, expression suggestion and staff-led mark (verified or model estimate), read from official runs and reviews.
- **Insights results**: the newest 005 result folder (indicator records with scope, value, unit, time base and tag). Read only.
- **Observation type**: an entry in the label config's observation list (003). Staff-extendable and versioned with the guideline.
- **Benchmark run**: an ordinary runs row from a Mac run of one video. Never official.
- **Theme palette**: the background, text, accent and chart-series colours, defined once in configuration and read by every screen and chart.
- **Benchmark summary**: per stage and end to end, Mac vs HPC runtime per video-hour, peak memory and model.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of export records have exactly the 13 case fields in the case's order, in both JSON and CSV (automated test).
- **SC-002**: For a mock set covering every review action, a rejected secondary, a staff-added label and a doubly reviewed label, 100% of exported records match the expected current state, with and without `--verified-only` (automated test).
- **SC-003**: The sample export has all five `review_decision` values, at least one staff-added label and one observation. It contains zero real filenames, zero paths and zero transcript words (automated scan against the mock video list and a planted transcript sentinel).
- **SC-004**: Zero transcript words or character counts appear in any rendered demo or insights screen for a mock video that has a mock transcript (automated scan).
- **SC-005**: Zero staff-led segments (verified or model estimate) are counted in youth indicators shown by default (automated test with mock staff-led segments of both kinds).
- **SC-006**: The blind checklist is identical before and after adding a new observation type, and the new type appears in review mode (automated test).
- **SC-007**: The Mac benchmark of one short promo records runtime and peak memory for all seven stages. The summary reports per-video-hour runtime per stage and end to end on both machines, or "no HPC run".
- **SC-008**: A judge can go from opening the app to one video's timeline with signals in under 1 minute, and to the insights headline in under 30 seconds more, without any raw ID needing explanation.
- **SC-009**: Zero filenames or paths appear on any judge-facing screen, in the benchmark summary or in any export file (automated scan with mock filenames).
- **SC-010**: 100% of text/background pairs meet at least 4.5:1 contrast. Measured at clarification:
  - #201C25 text: 15.7:1 on #F8F7F8, 16.7:1 on white;
  - #6F717A secondary text: 4.55:1 on #F8F7F8, 4.86:1 on white;
  - #7851B9 text: 5.35:1 on #F8F7F8, 5.71:1 on white; white text on #7851B9: 5.71:1;
  - #4100AD text: 10.75:1 on #F8F7F8, 11.49:1 on white.

  Teal and coral (about 3.2–3.4:1) meet the 3:1 bar for non-text graphics. A test fails if any palette text pair drops below 4.5:1, or if a palette value appears in more than one file.

## Assumptions

- The export covers all official moments (dev and eval videos). Segment-level blind reviews are not part of the export; they feed 005's evaluation.
- A staff `add` review counts as staff-sourced; an `accept` keeps the label AI-sourced.
- `confidence` is the moment's model confidence, unchanged by review.
- Several observations on one moment are joined into one `contextual_observation` value in time order (type, then note).
- `reviewer_note` comes from staff notes, which already carry the 003 no-names, no-judgments reminder. The export does not rewrite notes.
- The expression suggestion shown per segment is the highest expression label above its threshold, or "none" if no label clears it.
- The headline charts are drawn from 005's indicator records, so each number can show its tag. 005's own chart images are not reused.
- The benchmark is run by the team on a real promo after implementation. Development and tests use a synthetic clip. The HPC side uses the official HPC runs for the same stages.
- The demo view is a new tab in the existing 003 app, not a separate app, and keeps 003's local clip playback.
- The palette applies to the whole app, including blind and review mode, because it is one theme. Label colours from the label config are unchanged and stay separate from the palette.
- Secondary text #6F717A passes AA on #F8F7F8 only narrowly (4.55:1). It is not placed on any darker or tinted background.

## Dependencies on 005 (still open in 005's clarification)

These are not guessed here. 006 planning picks them up once 005 is merged.

- **Staff-led rule**: 005 Q1 is still open. 006 uses the rule in FR-018. If 005 settles on a different rule, FR-018 follows 005.
- **Headline file shape**: 005 FR-023 defines the headline (both values, tags, absolute change), but not whether it lives in `indicators.json` or a separate file, or its exact field names.
- **"Newest complete folder"**: 005 says folders are never overwritten, but not how a complete folder is marked or named.
- **Mid phase**: 005 Q6 (whether mid appears in its outputs) decides whether FR-021's mid value ever shows.
- **Solo moments**: 005 Q3 decides whether model estimates join staff `solo_on_stage`. 006 shows whatever tag 005 writes.
- **Youth voice time**: 005 Q7 (whether `has_speech` counts, whether staff time is excluded) changes the number, not 006's display.
- **Positive expression threshold**: 005 Q10 changes the number. 006 always shows it as "model estimate (unvalidated)".
