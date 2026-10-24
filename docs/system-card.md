# System Card: TPA Video Index

*2026 AI4Good Hackathon, Case #1, Track A. Placeholders in [brackets] are numbers or facts not yet recorded in the repo. Fill them from the latest results before submission.*

## Purpose and intended users

The Performance Academy (TPA) runs youth arts programs and films them. Hours of footage are hard to search, and hard to turn into evidence for funders.

This system turns long program videos into a **searchable, staff-verified timeline** of performance and participation moments, plus **program-level indicators** (for example, how much time youth spend performing, early in a program vs at the final showcase).

**Intended users** are TPA program staff. They review the AI's suggestions, correct them, and decide which numbers are good enough to share. The system supports staff judgment and does not replace it.

## What it does

- Splits each video into short overlapping windows (8 s windows with 2 s overlap; 4 s and 1 s for short clips) and scores each window against a fixed, staff-editable label vocabulary:
  - **activities**: singing, dance, acting/theater, spoken word, instrument and others;
  - **categories**: performance, creative practice, participation, instruction;
  - **who is leading**: youth or staff.
- Adds observable per-window signals: whether there is speech, audio cues (music, singing voice, speech, applause, silence), how many people are visible, and an expression suggestion on visible faces.
- Merges windows into **moments** (for example, "dance, 2:10–3:05") with a confidence score and a "needs review" flag.
- Offers a local **review app**: blind labeling for evaluation, and a review mode where staff accept, edit, reject or defer each AI label, add their own labels, or add an observation.
- Computes **youth-only program indicators** per video and per phase. Every number is tagged "staff-verified", "model estimate (unvalidated)" or "mixed (NN% staff-verified)".
- Exports the official annotation file (one record per moment, JSON and CSV) with source, review decision, guideline version and review date.

## What it does not do

- **No identification.** No face recognition, no face embeddings, no tracking of a person within or across videos. The unit of analysis is a time segment, never a person.
- **No judgments about people.** No psychological, clinical, health, demographic or mental-state inference. Expression labels describe what a camera sees (for example, a smile). They are never feelings, and are always shown as "AI suggestion, unvalidated".
- **No individual scores or rankings** of participants, and no measure of performance quality.
- **No cloud AI.** No hosted inference, no cloud storage, no generative AI services. Open model weights run locally.
- **No transcript text** leaves the HPC: it is never shown in the app or exported.

## Data

- **12 TPA program videos**, referred to only as **v01–v12**. Video kinds are live, edited, clip and promo. Each video has a program phase (early, mid, showcase or unknown) stored in a local, gitignored config.
- **Split by video**: v04 and v07 are the evaluation (eval) videos. All others are development (dev) videos, used for threshold calibration. No video is in both sets.
- **Where data lives**: only on the team's Mac and in UNF HPC private folders. Footage, frames, audio, transcripts, metadata and all data-derived outputs are never committed to git and never given to any AI assistant or cloud service. The repo ships only mock data and synthetic test clips.
- **Public samples** (the Devpost sample export, screenshots) are generated from mock data only.

## Pipeline and models

All weights are open, downloaded once, checked against a pinned sha256 where not hosted on Hugging Face, and run offline. The HPC uses larger models; the Mac uses smaller ones so the system fits a 16 GB laptop.

| Stage | What it produces | Model (HPC / Mac) | Source | Licence |
|---|---|---|---|---|
| Visual labels | per-window score for every label prompt | SigLIP2 so400m-patch14-384 / SigLIP2 base-patch16-224 | Hugging Face `google/siglip2-*` | [VERIFY: Apache-2.0 per the upstream model card] |
| Transcript | speech yes/no and a character count per window (text stays on HPC) | faster-whisper large-v3 / small | HF `Systran/faster-whisper-*` | MIT (OpenAI Whisper weights) |
| Audio cues | singing voice, music, speech, applause, silence | AST AudioSet 10-10-0.4593 (both) | HF `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD-3-Clause |
| People | number of visible people per window | YOLO11m / YOLO11n | GitHub `ultralytics/assets` v8.3.0 | **AGPL-3.0**: swap for a permissively licensed detector before deployment |
| Faces | face boxes, for the expression step only (never stored) | YuNet 2023mar | HF `opencv/face_detection_yunet` (pinned revision) | MIT |
| Expression | observable expression suggestion (positive, neutral, negative, upset) | EmotiEffLib enet_b0_8_va_mtl (ONNX, CPU) | GitHub `sb-ai-lab/EmotiEffLib` | Apache-2.0 |
| Motion (optional, off) | dance / acting check on clips | X-CLIP base-patch32 | HF `microsoft/xclip-base-patch32` | [VERIFY: upstream licence] |

Licences are as published upstream (`docs/weights.md`) and must be checked against each upstream page before final hand-off.

On top of the models:
- **Rules**: thresholds calibrated on dev videos only. "Uncertain / mixed" when no activity is clear or the top two are too close. Idle and setup detection. Signal rules (for example, "singing" needs a singing-voice or music cue).
- **Learned model**: logistic regression on the per-window signals (no images, no embeddings).
- **Moments**: neighbouring windows with the same main activity are merged.

## Human in the loop

- **Blind labeling**: staff label a seeded random sample of eval-video windows with **no model output shown**, using the full label checklist. These labels are the ground truth for evaluation. A second-labeler overlap set exists, but nobody labeled it (see Limitations).
- **Review mode**: for each moment, staff can accept, edit, reject or defer each AI label, add a label the model missed, or add an observation from a staff-editable list (for example, an audio problem). An on-screen reminder says: no names, no judgments about a person.
- **Audit log**: every action is one append-only row with reviewer, time and annotation **guideline version**. Nothing is overwritten; the current state of a label is its latest review.
- **Guideline versions**: any change to the label vocabulary bumps the guideline version. The app refuses to start if the vocabulary changed without a bump. Adding a staff observation type never changes the blind checklist, so evaluation labels stay comparable.

## Evaluation method

- **Ground truth**: blind labels on the held-out eval videos (v04, v07): about 205 labeled windows. Guideline 0.4.0 and 0.5.0 labels are pooled, because their checklists are identical.
- **Metrics**:
  - per-label precision, recall and F1 with support;
  - macro-F1 per label group;
  - primary-activity accuracy with a confusion matrix;
  - Cohen's kappa (model vs labeler) for primary activity and for category;
  - accuracy on confident vs flagged windows, and coverage.
- **Uncertainty**: 95% intervals from a **block bootstrap** (blocks of 10 consecutive labeled windows per video, 1,000 resamples, fixed seed), because neighbouring windows overlap and are correlated. No number is shown without its interval.
- **Ablation**: three systems on identical windows:
  1. **zero-shot**: visual labels and calibrated thresholds only;
  2. **zero-shot + signal rules**: audio, speech and person-count rules added;
  3. **learned**: logistic regression on all per-window signals, scored with **grouped cross-validation** (groups = the bootstrap blocks, 5 folds), so every prediction comes from a model that never saw its block.

### Results

| System | Primary accuracy (95% CI) | Macro-F1 activity (95% CI) | Kappa primary (95% CI) |
|---|---|---|---|
| Zero-shot | [RESULT] | [RESULT] | [RESULT] |
| Zero-shot + signal rules | [RESULT] | [RESULT] | [RESULT] |
| Learned (cross-validated) | [RESULT] | [RESULT] | [RESULT] |

- Category kappa: [RESULT: category kappa (95% CI)]
- Accuracy on confident vs flagged windows: [RESULT] vs [RESULT]; coverage [RESULT]
- Signal-based indicators (solo on stage, group performance, voice projection), F1: [RESULT]

## Known limitations

- **Single labeler.** Nobody labeled the overlap set, so there is no human-vs-human agreement to compare against.
- **Small eval set, mostly one video.** About 205 labeled windows from two videos, and v04 holds about 94% of them. Intervals are wide (about 21 bootstrap blocks).
- **Person counts include everyone visible.** Audience members, staff and passers-by are counted, not only performers. [TODO: describe the performer-size filter and its threshold once it is implemented; it is not yet recorded in the repo.]
- **Weak zero-shot actor and category scores.** The model rarely tells youth-led from staff-led reliably, so excluding staff time is mostly reliable only on reviewed clips. The share of time excluded by the model rule is reported.
- **Fixed audio threshold.** Audio rules use a fixed 0.5 probability, not a dev-tuned threshold, because the calibrated cut would veto real singing.
- **The learned model is trained on the eval labels.** Only its cross-validated scores are fair. Its predictions on the labeled windows themselves are in-sample.
- **Speech yes/no can include sung lyrics.** It means "words were transcribed".
- **Expressions need a visible face** of at least 48 px. Many windows have none.
- **Review decisions are per moment**, so one decision verifies every window in that moment.

## Efficiency

| | HPC (1 GPU, RTX PRO 6000 Blackwell) | Mac (16 GB laptop class) |
|---|---|---|
| Visual labels, per video-hour | [RESULT: HPC s per video-hour] (target: under 15 min per hour of video) | [RESULT: Mac benchmark] |
| Transcript / audio / people / expression, per video-hour | [RESULT: HPC per stage] | [RESULT: Mac benchmark per stage] |
| End to end, per video-hour (processing, excluding model loading) | [RESULT] | [RESULT: Mac benchmark total] |
| Peak memory | [RESULT] | [RESULT] |

Every pipeline run records its runtime, peak memory and hardware. The Mac numbers come from `tpa-index benchmark` on one promo video, which separates model-loading time from processing time.
