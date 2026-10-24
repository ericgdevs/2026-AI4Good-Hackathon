# Feature Specification: Review App

**Feature Branch**: `003-review-app`

**Created**: 2026-10-24

**Status**: Draft

**Input**: User description: "003-review-app: a local review app for TPA staff with two modes. **Blind labeling** (most urgent): about 300 seeded random eval-split segments, the full label checklist, no model output shown, built for speed, a 60-segment overlap set for a second labeler, and a rare-label top-up queue that is still shown blind. **Review**: a per-video timeline of moments with filters, playback in context, AI suggestions marked unvalidated, and accept/edit/reject/defer/add plus notes, each saved as an append-only review row. Local only, clips cut on demand from local footage into temp folders, light theme, launched with `tpa-index app`. Out of scope: the demo view, the official export (006) and syncing reviews (005)."

## User Scenarios & Testing *(mandatory)*

The users are TPA staff members (the labelers and reviewers) and the team running the hackathon evaluation. Youth appear in the footage but never use the app, and never appear in it as named people. Every scenario here can be demonstrated with mock data and synthetic clips.

### User Story 1 - Blind-label a fixed random sample quickly (Priority: P1) 🎯

A staff member opens the app, enters their first name or role, and starts blind labeling. The app shows one short segment at a time from a fixed random sample of eval-split segments. Under it is the full label checklist: primary activity, secondary activities, category, actor, indicators, and an uncertain/mixed option. Nothing from the model is visible. They tick what they see, press one key to save and move on, and watch a progress counter. If they close the app and come back later, they resume exactly where they stopped, with the same sample.

**Why this priority**: The blind labels are the evaluation ground truth (Constitution VIII). Without them no reported number can be trusted, and they must exist by Saturday morning.

**Independent Test**: With mock data and synthetic clips, start a session and label 5 segments using only the keyboard. Restart the app and confirm it resumes at segment 6 with the identical sample order. Confirm that no model score, model label or "AI" wording appears anywhere on the blind screen.

**Acceptance Scenarios**:

1. **Given** official runs that include eval-split videos, **When** a labeler starts blind labeling, **Then** they get the same sample of about 300 eval segments, in the same order, every session, and on every machine with the same official runs.
2. **Given** the blind screen, **When** it is shown, **Then** it contains the clip, the label checklist taken from the label config, the progress counter and the save controls, and no prediction, score, model-chosen ordering or AI marker.
3. **Given** a segment on screen, **When** the labeler picks a primary activity (or marks uncertain/mixed), ticks other labels and presses save-and-next, **Then** one review row is stored per chosen label, with mode blind, source staff, the reviewer, the guideline version and a timestamp, and the next segment appears.
4. **Given** a labeler tries to save without a primary activity or uncertain/mixed, **When** they press save, **Then** the save is refused with a short message, so "nothing chosen" is never confused with "not yet labeled".
5. **Given** a labeler has saved 120 segments, **When** they reopen the app, **Then** the counter shows 120 of the sample done and the 121st segment is shown.
6. **Given** a labeler goes back to a segment they already labeled and changes it, **When** they save, **Then** new rows record the change (including a reject row for any label they unticked). Earlier rows are never altered.

---

### User Story 2 - Second labeler and rare-label top-up (Priority: P1)

A second staff member labels a fixed 60-segment overlap set taken from the same sample, also blind, so that agreement between labelers can be measured later. After the main sample is done, labels that are still rare get a top-up queue of extra eval segments, also shown blind.

**Why this priority**: Without overlap there is no agreement measure, and without the top-up rare labels have too few examples to evaluate. Both are part of the Saturday ground truth.

**Independent Test**: With two reviewer names on mock data, the second reviewer's queue contains exactly the 60 overlap segments, in a fixed order. The top-up queue contains only eval segments that are not already in the sample, and its screen is identical to the blind screen.

**Acceptance Scenarios**:

1. **Given** a reviewer chooses "overlap set", **When** they label, **Then** they see exactly the 60 overlap segments, which are the same for everyone and drawn from the main sample, and their rows are stored like any blind label.
2. **Given** the main sample is done, **When** a labeler opens the top-up queue, **Then** it lists eval segments for the labels that are still rare, never repeats a segment already blind-labeled by that reviewer, and is shown blind.
3. **Given** any blind segment, **When** its rows are later used for metrics, **Then** it is possible to tell whether it came from the random sample, the overlap set or the top-up queue, because top-up segments are not a random sample.
4. **Given** official predict runs exist for the eval videos, **When** the top-up queue is built, **Then** for each rare label it takes up to 20 not-yet-labeled eval segments with that label's highest model scores. Those segments are tagged with queue `topup`, shuffled together, and shown without the target label. **Given** no official predict runs exist for the eval videos, **Then** the top-up queue is hidden with a short message.
5. **Given** top-up rows, **When** metrics use them, **Then** they count only toward per-label precision, never toward the random-sample metrics.

---

### User Story 3 - Review AI moments on a timeline (Priority: P2)

A staff member opens review mode, picks a video and sees its moments on a timeline. They can filter by label, by needs-review, and by review status (unreviewed, partly reviewed, done, deferred), and idle stretches are hidden by default. Selecting a moment plays it with a few seconds of context before and after. Its primary and secondary labels, indicators and confidence are shown, each clearly marked "AI suggestion, unvalidated" until a staff member reviews it.

**Why this priority**: Review turns model output into staff-verified moments (Constitution VI). It matters, but blind ground truth comes first.

**Independent Test**: On mock data, open a video, filter to needs-review, play a moment and confirm the padding. Confirm every unreviewed label shows the unvalidated marker and a reviewed label shows its review state instead.

**Acceptance Scenarios**:

1. **Given** a video with moments, **When** the timeline loads, **Then** all its moments are shown in time order with their primary label, and idle segments are hidden unless the "show idle" filter is on.
2. **Given** filters are set, **When** they are combined (for example, the label dancing plus needs-review plus unreviewed), **Then** only matching moments remain, and the count is shown.
3. **Given** a moment is selected, **When** it plays, **Then** playback covers the moment plus the configured padding before and after, clipped to the video's bounds.
4. **Given** a label with no review yet, **When** it is shown, **Then** it carries the "AI suggestion, unvalidated" marker and its confidence.
5. **Given** the label config gains new groups (for example, audio or expression labels from 004), **When** the app loads, **Then** those labels are shown and reviewable without any code change.
6. **Given** the main blind sample or the overlap set is not finished, **When** review mode lists videos, **Then** eval-split videos are hidden, with a short message saying why. They appear once every main-sample segment has a blind save and every overlap segment has blind saves from at least two different reviewers.

---

### User Story 4 - Accept, edit, reject, defer or add (Priority: P2)

For each AI label on a moment, the reviewer can accept, edit (choose a different label), reject or defer it. They can also add a staff label the model missed, or add an observation, and attach an optional note. Each action is saved at once and is visible in the moment's review state.

**Why this priority**: These actions are the audit log that makes moments trustworthy and that evaluation reads.

**Independent Test**: On mock data, perform each of the five actions on one moment. Confirm one review row per action with mode review and the correct source, and that the moment's displayed state follows the latest action per label.

**Acceptance Scenarios**:

1. **Given** an AI label, **When** the reviewer accepts, rejects or defers it, **Then** one row is stored with that action, source ai, mode review, the reviewer, the guideline version and a timestamp.
2. **Given** an AI label, **When** the reviewer edits it to another label, **Then** one row is stored with action edit and the new label, and the moment now shows the new label as staff-verified.
3. **Given** a moment, **When** the reviewer adds a label from the vocabulary, **Then** a row is stored with action add and source staff.
4. **Given** a moment, **When** the reviewer adds an observation, **Then** they pick one type from the fixed observation list in config and may add a short note. It is stored with role `observation`, and a visible reminder next to the note says: no names, no judgments about a person.
5. **Given** several actions on the same label over time, **When** the moment is shown, **Then** only the latest action counts as the current state. All earlier rows stay in the log unchanged.

---

### User Story 5 - Keep footage and people safe while reviewing (Priority: P1)

Staff run the app on the Mac that holds the footage. Clips play from short temporary files cut on demand, which are removed afterwards. The app is reachable only from that machine and never sends anything out.

**Why this priority**: Constitution I–V are non-negotiable, and footage of youth must never leave the machine.

**Independent Test**: Launch the app and confirm it listens only on the local machine, has public sharing off, and makes no outbound network calls during a session. After the session, the temporary clip folder is empty and nothing new exists inside the repo.

**Acceptance Scenarios**:

1. **Given** the app is launched, **When** another machine on the network tries to open it, **Then** it cannot connect. There is no public share link in any mode.
2. **Given** a clip is needed, **When** it is cut, **Then** it is written only to a temporary folder outside the repo, and it is deleted when no longer shown or when the app exits.
3. **Given** the reviewer name prompt, **When** a name is entered, **Then** the prompt says it must be a staff first name or role. No screen shows a youth's name, a per-person score or a ranking.
4. **Given** an error while cutting or playing a clip, **When** it is shown, **Then** paths and filenames are redacted (as in 001).

---

### Edge Cases

- **No official runs, or no eval-split videos:** blind mode explains what is missing and does not invent a sample.
- **Fewer than 300 eval segments exist:** the sample uses all of them, and the counter shows the real total.
- **Idle segments in the random sample:** they are kept, because an idle stretch is a real outcome to label. Only review mode hides idle by default.
- **The official runs change after labeling started:** the existing sample stays fixed. Segments that no longer exist are reported and skipped, never silently replaced.
- **Two reviewers have the app open at once on one machine:** both can save. Rows are appended and none is lost.
- **A video file is missing on disk:** that segment shows a clear, redacted error and can be skipped with a "skip (file missing)" action that is itself logged.
- **A moment near the start or end of a video:** the padding is clipped to the video's bounds.
- **A label removed from the vocabulary after reviews were saved:** old rows stay. The label shows as "retired" and can no longer be chosen.
- **The guideline version changes mid-session:** new rows carry the new version, and existing rows keep theirs.

## Requirements *(mandatory)*

### Functional Requirements

**Shared**

- **FR-001**: The app MUST read videos, segments and moments only through the official-run readers from 001/002, never from all runs.
- **FR-002**: Every staff action MUST be stored as a new row in the reviews log, following the 001 contract. Rows are never edited or deleted. The current state of a (target, label) pair is its latest row.
- **FR-003**: The reviews contract MUST gain `observation` as an allowed `label_role` (a 001 contract change). Rows stored before this change stay valid.
- **FR-004**: The label checklist and the label groups shown MUST come from the label config at runtime. New groups appear without code changes.
- **FR-005**: The app MUST ask for a reviewer (a staff first name or role) before any labeling, remember it for the session, and store it on every row.

**Blind labeling**

- **FR-006**: The app MUST build a random sample of about 300 eval-split segments from the official runs, using a fixed seed. The same official runs and seed MUST always give the same sample in the same order.
- **FR-007**: The sample, the 60-segment overlap subset and every top-up queue MUST be saved once they are created, with each segment's queue name and position. Later changes to official runs or predictions must not reshuffle them, and metrics can tell random-sample, overlap and top-up segments apart.
- **FR-008**: The blind screen MUST NOT show any model output: no scores, no suggested labels, no target label, no model-chosen order of checklist items, and no AI markers. The main sample and the overlap set MUST be chosen without model output. Only the top-up queue uses model scores, and only to choose segments, whose order is then shuffled.
- **FR-009**: The blind screen MUST let a trained labeler finish a segment in about 25 seconds: keyboard shortcuts for every checklist item and for save-and-next, plus a progress counter.
- **FR-010**: A blind save MUST require a primary activity or uncertain/mixed. It MUST store one row per chosen label (action add, source staff, mode blind, with the label role from the label's group) and a reject row for any label the same reviewer had chosen earlier on that segment and has now unticked.
- **FR-011**: Blind labeling MUST resume at the first segment in the reviewer's queue that they have not yet saved.
- **FR-012**: A second-labeler mode MUST present the fixed 60-segment overlap subset of the main sample, in a fixed order.
- **FR-013**: A top-up queue MUST cover labels that are still rare after the main sample (fewer than a configured number of blind positives). For each, it takes up to 20 (configurable) not-yet-blind-labeled eval segments with the highest official model score for that label. The queue is tagged `topup`, shuffled, shown blind, and never repeats a segment the reviewer has already labeled. It is available only when official predict runs cover the eval videos; otherwise it is hidden with a message. Top-up rows count only toward per-label precision, never toward random-sample metrics.

**Review**

- **FR-014a**: Review mode MUST hide eval-split videos until every main-sample segment has a blind save and every overlap segment has blind saves from at least two different reviewers. The reason is shown while they are hidden.
- **FR-014**: Review mode MUST show a per-video timeline of moments, filterable by label, by needs-review and by review status (unreviewed, partly reviewed, done, deferred). Idle segments are hidden by default and can be shown with a toggle.
- **FR-015**: Selecting a moment MUST play it with configurable padding before and after (default 3 s), clipped to the video's bounds.
- **FR-016**: Each AI label MUST show its confidence and the marker "AI suggestion, unvalidated" until a review row exists for it. After that, its latest review state is shown.
- **FR-017**: For each AI label the reviewer MUST be able to accept, edit (to another vocabulary label), reject or defer it. For each moment they MUST be able to add a vocabulary label, or an observation chosen from the fixed observation-type list in config, each with an optional short note. A visible reminder next to the note says: no names, no judgments about a person. Every action stores one row with mode review.
- **FR-017a**: Observation types MUST live in the label config, next to the guideline version. Staff may add types, and adding one MUST bump the guideline version.

**Privacy and running**

- **FR-018**: The app MUST listen on the local machine only (loopback), MUST never create a public share link, and MUST make no outbound network calls, including telemetry and analytics.
- **FR-019**: Clips MUST be cut on demand from the local footage into a temporary folder outside the repo, and deleted when no longer shown and when the app exits.
- **FR-020**: Errors and logs MUST stay redacted, as in 001.
- **FR-021**: The app MUST show no youth names, no per-person outputs, and no individual scores or rankings (Constitution IV–V).
- **FR-022**: The app MUST use a light theme only (Constitution X).
- **FR-023**: The app MUST start with one command and MUST work fully on mock data and synthetic clips.

### Key Entities

- **Review row** (001 contract): one staff action on a segment or moment label. 003 adds `observation` to the allowed label roles.
- **Blind queue**: a saved, ordered list of segment IDs with a queue name (`main`, `overlap` or `topup`), the seed, the position and, for top-up, the target label (stored for metrics, never shown). It is created once and reused.
- **Observation type**: one entry in the config's fixed list of observation kinds (for example, a staging or audio problem). The list is versioned with the guideline.
- **Current label state**: for a (target, label) pair, the latest review row. It drives the review-status filters and the "unvalidated" marker.
- **Reviewer**: a staff first name or role typed by the user. No other identity is stored.
- **Clip**: a temporary, on-demand cut of local footage for one segment or moment (plus padding). It is never kept.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A trained labeler completes blind segments at a median of 25 seconds or less each, measured from first view to save.
- **SC-002**: The main sample (about 300 segments) can be fully blind-labeled by one person within about 2.5 hours of labeling time.
- **SC-003**: Restarting the app resumes at the correct segment and with the identical sample order 100% of the time.
- **SC-004**: An automated check finds 0 model-derived values on any blind-mode screen.
- **SC-005**: 100% of staff actions in both modes produce exactly one stored row per affected label, with reviewer, guideline version and timestamp.
- **SC-006**: A moment's clip starts playing within 3 seconds of selecting it on the Mac.
- **SC-007**: During a full session, 0 outbound network connections are made and 0 files are left in the temporary clip folder or created in the repo.
- **SC-008**: Adding a label group to the label config shows up in both modes with 0 code changes.

## Clarifications

### Session 2026-10-24

- Q: How does the rare-label top-up pick segments? → A: Option A, capped at 20 per rare label. Rows are tagged by queue `topup` and reported separately, as precision only. The queue appears only when official predict runs exist for the eval videos.
- Q: What is an observation? → A: Option B: a fixed, staff-extendable list of observation types in config (adding one bumps the guideline version), plus an optional short note with a visible no-names, no-judgments reminder.
- Q: Eval videos in review mode? → A: Option B: hidden until the main sample and the overlap set are blind-labeled (every main segment saved once, every overlap segment saved by at least 2 reviewers).

## Assumptions

- **Official runs:** blind mode needs only official runs that contain the eval videos' segments (predict or segment stage). Review mode needs official predict or moments runs.
- **Sample size and seed:** "about 300" means 300, or every eval segment if there are fewer. The seed, sample size, overlap size (60), top-up thresholds and padding all live in the pipeline config.
- **Overlap order:** the overlap set is the first 60 segments of the main sample's fixed order, so both labelers see them early.
- **Label roles:** a blind primary activity is stored with role primary, other activities with role secondary, and indicators with role indicator. Category, actor and the uncertain/mixed flag use role secondary.
- **Reviews log:** stays in its own local database file next to the pipeline outputs but outside the run folders (as in 001). Output sync never copies it.
- **Reviewer names:** free text. The app cannot verify them; the on-screen instruction and the staff guideline cover it.
- **Clip cutting:** clips are cut with the system's video tool on the Mac. Mock mode uses synthetic clips.
- **Out of scope:** the demo view, the official export (006), syncing reviews to HPC (005), computing metrics such as Cohen's kappa (005), and any multi-user server setup.
