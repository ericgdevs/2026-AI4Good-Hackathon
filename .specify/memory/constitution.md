<!--
Sync Impact Report
- Version change: template (unversioned) → 1.0.0 (initial ratification)
- Principles: all 13 defined from team input (template had 5 placeholder slots)
- Added sections: Core Principles I–XIII, Project Scope & Constraints, Development Workflow & Quality Gates, Governance
- Removed sections: none
- Templates:
  ✅ .specify/templates/plan-template.md — Constitution Check derives gates from this file; no edit needed
  ✅ .specify/templates/spec-template.md — no mandatory-section changes required
  ✅ .specify/templates/tasks-template.md — no new task categories mandated
  ⚠ .gitignore — data/model patterns are indented with leading spaces and do not match (see follow-ups)
  ⚠ .claude/settings.json — read-deny rules for data dir and local metadata (Principle III) not yet present
- Follow-up TODOs: none deferred in this document
-->

# TPA "This Is ME" Video Indexing Constitution

## Core Principles

### I. Data Stays Local or on UNF HPC (NON-NEGOTIABLE)

- Footage, frames, audio, transcripts, crops, all data-derived outputs, and dataset metadata
  (filenames, durations, video notes, phase assignments) MUST live only on the team's local
  machines or in UNF HPC private folders.
- Analysis runs locally. Data MAY be uploaded to UNF HPC only for training or heavy
  processing, under UNF HPC privacy restrictions.
- No other cloud: no Hugging Face compute, no cloud storage, no hosted/cloud generative AI.
- Nothing from the dataset is ever committed to git. Data and metadata paths MUST come from
  `.env` and point outside the repo.
- Retention: post-event retention is undecided. Data MUST NOT be deleted until organizers decide.
- Privacy freeze: until the organizers say otherwise, all data and derived annotations MUST NOT
  be reused for portfolios, papers, or other projects without written permission.

Rationale: the footage shows youth participants; custody of it is the project's first obligation.

### II. Open Models, Efficient Local Execution

- Models MUST be open-source Hugging Face models; only weights are downloaded, never hosted
  inference.
- Heavy processing and training run on UNF HPC (4x RTX PRO 6000 Blackwell). No Colab.
- The final pipeline and app MUST run end to end on a machine TPA owns, including a 16 GB laptop.
- Prefer efficient video understanding (sampling, segment-level processing) over brute-force
  frame-by-frame processing.
- Every pipeline run MUST report runtime and peak memory, for both HPC and the Mac.

Rationale: TPA must be able to run the system after the hackathon without our hardware.

### III. AI Assistants Never See Data (NON-NEGOTIABLE)

- Nothing from the dataset or its metadata is saved to Anthropic/Claude storage: not in chats,
  project docs, memory, or Claude Code context.
- Claude Code MUST be denied read access to the data directory and local metadata files via
  permission settings.
- Assistants MAY receive only: code, errors with paths and filenames redacted, schemas, mock
  data, and aggregate metric numbers that name no video or participant.

Rationale: assistant context is third-party storage; Principle I applies to it too.

### IV. No Identity (NON-NEGOTIABLE)

- No facial recognition, face embeddings, re-identification, or same-person matching, within
  or across videos.
- The unit of analysis is the time segment, not the person. Person counts and expression
  scores MUST be aggregated per segment.

### V. No Judgments About People (NON-NEGOTIABLE)

- No psychological, clinical, health, disability, demographic, or mental-state inferences,
  including as hidden labels, intermediate features, or prompts.
- Expression labels (Confident, Upset, Positive, Neutral, Negative) are observable,
  staff-verified proposals, never assessments.
- The system MUST NOT produce individual risk scores, participant rankings, behavioral
  rankings, or unreviewed claims about youth development.
- All indicators are program-level aggregates. Performance quality is not measured.

### VI. Humans Verify

- Every model output is an unvalidated suggestion carrying a confidence score and a source
  (`AI` or `staff`), and MUST be shown as such in the UI.
- Staff can accept, edit, reject, defer, or add labels, including an explicit
  "uncertain / mixed / needs review" state.
- Every review action MUST be logged with reviewer, date, and annotation-guideline version.
  The review log doubles as evaluation data.

### VII. Contracts First

- A shared segment-table schema and label config MUST be fixed before feature work begins.
- Mock data MUST let every component run without real footage.
- Specs lock interfaces, not model choices.
- Specs and code MUST NOT contain real filenames or dataset details.

### VIII. Measure Everything Shown

- Every reported number has a defined metric (per-label precision/recall, macro-F1), an eval
  set split by video (no video in both train and eval), and runtime plus hardware (Mac and HPC).
- A number without these is not shown in the app, deck, or Devpost submission.

### IX. Simplest Thing That Works

- Reuse libraries and pretrained models.
- No custom training beyond small classifiers on frozen embeddings.

Rationale: 48 hours, 3 people; every hour on infrastructure is an hour not spent on verification.

### X. Presentation-Ready

- Light mode everywhere: Gradio theme forced light; charts on white backgrounds with legends;
  screenshots and recordings captured in light mode.

### XI. Config Over Code

- Anything data-dependent MUST be configuration, not code: label vocabulary, segment length,
  frame sampling rate, and video metadata (e.g. early program vs. final showcase).
- Real video metadata lives in a local, gitignored config file. The repo ships only a mock example.

### XII. One App, Two Modes

- The same app runs the pipeline and reviews results.
- A demo view loads precomputed results so a presentation never waits on processing.
- The demo view MUST use the same data model and code paths as the live app; no separate mock UI.

### XIII. Deliverables Are First-Class

- The repo produces every required Devpost asset: runnable code, architecture diagram,
  structured annotation export (JSON/CSV), review interface, model/system card,
  data-governance note, and an end-to-end demo.
- Public samples MUST come from mock data or a public test video, never TPA footage.

## Project Scope & Constraints

- Event: 2026 AI4Good Hackathon, Case #1, Track A. Team of 3, 48 hours.
- Deadline: Devpost submission Sun 2026-10-25 10:00 AM ET.
- Goal: turn long program videos into a searchable, staff-verified timeline of performance and
  participation moments, plus aggregate observable indicators (participation, self-expression,
  original-work moments) that help The Performance Academy show funders program impact.
- Out of scope: measuring performance quality; any per-person output.

## Development Workflow & Quality Gates

- Every spec and plan MUST pass a Constitution Check against Principles I–XIII before design and
  again after design; violations go in the plan's Complexity Tracking table with justification.
- Before any commit: no data files, real filenames, real metadata, or `.env` contents are staged.
- Before any assistant interaction: inputs are limited to what Principle III allows.
- Before any reported number reaches a slide or the app: it meets Principle VIII.

## Governance

- This constitution supersedes other project practices. Where a spec, plan, or task conflicts
  with it, the constitution wins.
- Amendments: changes to Principles I–V require agreement from the whole team and, where they
  touch data rules, confirmation from Dr. Umapathy. Other principles may be amended by team
  agreement.
- Versioning: semantic versioning. MAJOR for removing or redefining a principle; MINOR for adding
  a principle or materially expanding guidance; PATCH for clarifications and wording.
- Compliance: every plan's Constitution Check and every review verifies compliance.

**Version**: 1.0.0 | **Ratified**: 2026-10-23 | **Last Amended**: 2026-10-23
