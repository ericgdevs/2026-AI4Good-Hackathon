# System Card: TPA Video Index

*2026 AI4Good Hackathon, Case #1, Track A. All numbers are aggregates from our final evaluation on held-out videos. No footage details, names or quotes appear here.*

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
- **No cloud AI.** No hosted inference, no cloud storage, no generative AI services. Open model weights run on private machines we control (a university compute node and our laptop).
- **No transcript text** leaves the private compute folder: it is never shown in the app or exported.

## Data

- **12 TPA program videos, about 5 hours**, referred to only as **v01–v12**. They give 2,373 eight-second windows (clips). Video kinds are live, edited, clip and promo. Each video has a program phase (early, mid, showcase or unknown), assigned per video by the team and stored in a local, git-ignored config.
- **Split by video**: v04 and v07 are the 2 held-out evaluation (eval) videos. The other 10 are development (dev) videos, used for threshold tuning. No video is in both sets.
- **Where data lives**:
  - videos on a private external drive and in a private university compute folder;
  - per-window numeric outputs in the same private locations;
  - transcripts in the private compute folder only;
  - faces never saved;
  - the review database on the team laptop;
  - results folders hold aggregates only;
  - real video names appear in only one git-ignored local file.

  Footage, frames, audio, transcripts and metadata are never committed to git and never given to any AI assistant or cloud service. The GitHub repo is private and holds code, mock data and docs only.
- **Public samples** (the Devpost sample export, screenshots) are generated from mock data only.

## Pipeline and models

All models are open source. Weights are downloaded once, checked against a pinned sha256 where not hosted on Hugging Face, and run offline on private machines we control. The university compute node uses larger models. The laptop runs smaller versions of the same models, with the same pipeline and commands, so the system fits a 16 GB laptop that TPA can own.

| Stage | What it produces | Model (compute node / laptop) | Source | Licence |
|---|---|---|---|---|
| Visual labels | per-window score for every label prompt | SigLIP2 so400m-patch14-384 / SigLIP2 base-patch16-224 | Hugging Face `google/siglip2-*` | Apache-2.0 |
| Transcript | speech yes/no and a character count per window (the text stays in the private compute folder) | faster-whisper large-v3 / small | HF `Systran/faster-whisper-*` | MIT (OpenAI Whisper weights) |
| Audio cues | singing voice, music, speech, applause, silence | AST AudioSet 10-10-0.4593 (both) | HF `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD-3-Clause |
| People | number of visible people per window (counts only) | YOLO11m / YOLO11n | GitHub `ultralytics/assets` v8.3.0 | **AGPL-3.0**: swap for an Apache-licensed detector before deployment |
| Faces | face boxes, for the expression step only (never stored) | YuNet 2023mar | HF `opencv/face_detection_yunet` (pinned revision) | MIT |
| Expression | observable expression suggestion (positive, neutral, negative, upset), scored in memory and always "unvalidated" | EmotiEffLib enet_b0_8_va_mtl (ONNX, CPU) | GitHub `sb-ai-lab/EmotiEffLib` | Apache-2.0 |

Licences are as published upstream (`docs/weights.md`) and must be checked against each upstream page before final hand-off.

On top of the models:
- **Rules**: thresholds calibrated on dev videos only. "Uncertain / mixed" when no activity is clear or the top two are too close. Idle and setup detection. Signal rules (for example, "singing" needs a singing-voice or music cue).
- **Learned model**: logistic regression on about 40 per-window scores (visual label scores, audio cues, speech, person count, expression). It uses no images and no embeddings.
- **Moments**: neighbouring windows with the same main activity are merged.

## Human in the loop

- **Blind labeling**: staff label a seeded random sample of eval-video windows with **no model output shown**, using the full label checklist. These labels are the ground truth for evaluation. A second-labeler overlap set exists, but nobody labeled it (see Limitations).
- **Review mode**: for each moment, staff can accept, edit, reject or defer each AI label, add a label the model missed, or add an observation from a staff-editable list (for example, an audio problem). An on-screen reminder says: no names, no judgments about a person.
- **Audit log**: every action is one append-only row with reviewer, time and annotation **guideline version**. Nothing is overwritten; the current state of a label is its latest review.
- **Guideline versions**: any change to the label vocabulary bumps the guideline version. The app refuses to start if the vocabulary changed without a bump. Adding a staff observation type never changes the blind checklist, so evaluation labels stay comparable. The blind labels and every evaluation number in this card were produced under guideline **0.4.0/0.5.0** (identical blind checklists). The review app now runs **0.6.0**, which only adds five optional staff observation buttons (Confident, Positive, Neutral, Negative, Upset) and changes no label used in the evaluation.

## Evaluation method

- **Ground truth**: 206 windows from the held-out eval videos (v04, v07), labeled blind by one team member with no AI output shown. Guideline 0.4.0 and 0.5.0 labels are pooled, because their blind checklists are identical.
- **Metrics**:
  - per-label precision, recall and F1 with support;
  - macro-F1 per label group;
  - primary-activity accuracy with a confusion matrix;
  - Cohen's kappa (model vs labeler) for primary activity and for category;
  - accuracy on confident vs flagged windows, and coverage.
- **Uncertainty**: 95% intervals from a **block bootstrap** (blocks of 10 consecutive labeled windows per video, 1,000 resamples, fixed seed), because neighbouring windows overlap and are correlated. Every metric has its interval in the evaluation output. This card shows the intervals for primary accuracy.
- **Ablation**: three systems on identical windows:
  1. **zero-shot**: visual labels and calibrated thresholds only;
  2. **zero-shot + signal rules**: audio, speech and person-count rules added;
  3. **learned**: logistic regression on all per-window signals, scored with **grouped cross-validation** (groups = the bootstrap blocks, 5 folds), so every prediction comes from a model that never saw its block.

### Results

| System | Primary accuracy (95% CI) | Macro-F1 activity | Macro-F1 category | Macro-F1 youth vs staff | Kappa primary |
|---|---|---|---|---|---|
| Zero-shot | 34% (26–42) | 22% | 14% | 10% | 0.17 |
| Zero-shot + signal rules | 36% (28–45) | 23% | 14% | 10% | 0.20 |
| Learned (cross-validated) | 46% (37–54) | 39% | 49% | 69% | 0.26 |

- Category kappa: 0.05 (zero-shot), 0.05 (+ signal rules), 0.39 (learned).
- Accuracy on confident vs flagged windows and coverage are computed by the evaluation, but are not among the final headline numbers reported here.
- Signal-based indicators, F1 with zero-shot labels → with learned labels: solo on stage 23% → 53%, voice projection 22% → 42%, group performance 76% → 58%.

**Youth-only indicators by phase** (model estimates; 4 early, 6 mid and 2 showcase videos):

| Indicator | Early | Mid | Showcase |
|---|---|---|---|
| Youth voice, minutes per hour | 13.6 | 20.0 | 31.3 |
| Solo moments, minutes per hour | 13.6 | 24.8 | 23.0 |
| Performance share of youth time | 12% | 26% | 76% |
| Positive expression rate (unvalidated) | 28% | 35% | 34% |
| TPA Expression Index (equal weights, optional example) | 21 | 34 | 50 |

- **Time**: 150.4 minutes of youth time. Staff-led time is reported separately: 3.5 minutes staff-verified and 77.3 minutes by model rule.
- These describe these videos at the footage level. They are not proof of change and not measures of individual youth.

## Known limitations

- **Single labeler.** One team member labeled the eval set. Nobody labeled the overlap set, so there is no human-vs-human agreement to compare against.
- **Small eval set, mostly one video.** 206 labeled windows from two videos, and v04 holds about 94% of them. Intervals are wide (about 21 bootstrap blocks).
- **Person counts include everyone visible.** Nearby non-performers (audience members, staff, passers-by) can be counted, not only performers. We tested a performer-size filter that ignores small, distant people. It had no measurable effect on the results, and we report that as an honest negative result.
- **Weak zero-shot actor and category scores.** The zero-shot model rarely tells youth-led from staff-led reliably (macro-F1 10%). The learned model does much better (69%), but excluding staff time is fully reliable only on reviewed clips. The share of time excluded by the model rule is reported.
- **Fixed audio threshold.** Audio rules use a fixed 50% probability, not a dev-tuned threshold, because the calibrated cut would veto real singing.
- **The learned model is trained on the eval labels.** Only its cross-validated scores are fair. Its predictions on the labeled windows themselves are in-sample.
- **Rare activities can't be learned.** With so few labeled examples, the learned model cannot pick up activities that appear only a handful of times.
- **Phases are assigned per video by the team**, so early vs showcase compares different videos, filmed differently. These are footage-level comparisons, not individual ones.
- **The idle detector did not fire on this footage**, so no time was set aside as idle or setup.
- **Speech yes/no can include sung lyrics.** It means "words were transcribed".
- **Expressions need a visible face** of at least 48 px. Many windows have none.
- **Review decisions are per moment**, so one decision verifies every window in that moment.

## Efficiency

Measured full run on a university compute node (GPU): all 12 videos, about 5 hours, 2,373 clips.

| Step | Models | Time for all 12 videos |
|---|---|---|
| Activity labels | SigLIP2 so400m | 27.6 min |
| Speech and sound | Whisper large-v3 + AST | under 15 min |
| People and expressions | YOLO11 + YuNet + EmotiEffLib | about 14 min, split across 4 GPUs in parallel (a people-only rerun took about 12.5 min) |
| Learned model, metrics and indicators | logistic regression and aggregation, on our laptop | seconds |
| Laptop (smaller models, same pipeline) | SigLIP2 base, Whisper small, AST, YOLO11n, YuNet + EmotiEffLib | ~1 min per 1-minute video (estimate; to be measured) |

The face and expression step was made about 11× faster (from about 9 s to about 0.8 s per clip) by pinning CPU threads, running face detection on a downscaled copy of each frame, and scoring faces in batches.

Every pipeline run records its runtime, peak memory and hardware. Laptop numbers can be produced with `tpa-index benchmark` on one video, which separates model-loading time from processing time.
