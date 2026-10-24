# Data Governance Note: TPA Video Index

*How we protect the youth in TPA's footage, and what staff must check before any number leaves TPA.*

## Privacy and safety rules

These rules are part of the project constitution, and the code and tests enforce them.

- **No identification.** No face recognition, no face embeddings, no re-identification, no "same person" matching within or across videos.
- **No tracking.** The unit of analysis is a time segment. People are counted per segment and never followed over time.
- **No face storage.** Faces are detected only to read an expression suggestion, in memory. No face images, crops, landmarks or embeddings are saved.
- **Transcripts stay on HPC.** Speech is transcribed on UNF HPC to tell whether someone is speaking. The text stays in the HPC run folder. It is never synced to a laptop, shown in the app, written to logs or exported. Only "speech yes/no" and a character count leave the HPC.
- **Expressions are observations, not feelings.** Labels such as "positive" describe what a camera sees (for example, a smile). They are always shown as "AI suggestion, unvalidated", and are never used as a measure of anyone's emotions, wellbeing or ability.
- **No clinical claims.** No psychological, clinical, health, disability, demographic or mental-state inference, including as hidden features or prompts.
- **No individual scores or rankings.** All indicators are program-level aggregates (per video or per program phase). Nothing is reported per participant, and performance quality is not measured.
- **Youth-only indicators.** Following TPA's guidance (2026-10-24) that the program cares about the youth, not the instructors, every TPA-facing indicator excludes staff-led time. That time is reported separately as "staff-led time", split into staff-verified and model-estimated.
- **No filenames or names anywhere public.** Videos are referred to as v01–v12. Public samples come from mock data only.

## Data handling and retention

- **Where data lives**: only on the team's Mac and in UNF HPC private folders, under UNF HPC privacy rules. No cloud storage, no cloud or hosted AI, and nothing from the dataset in git or in any AI assistant's context. Assistants are denied read access to the data folders by configuration.
- **Local processing**: all models are open weights, downloaded once and run offline.
- **Retention**: post-event retention has not been decided. **No data will be deleted until the organizers decide.**
- **No reuse**: the footage and all derived annotations will not be reused for portfolios, papers or other projects without written permission.

## Before using any number for funders, staff must check

1. **Its tag.** "staff-verified" numbers rest on reviewed labels. "model estimate (unvalidated)" numbers do not. "mixed (NN% staff-verified)" tells you how much was reviewed. Prefer verified numbers, and say which kind you are quoting.
2. **Its interval and base.** Every evaluation number has a 95% interval and a sample size. Every indicator names its denominator (for example, "youth non-idle time"). Don't quote a number without them.
3. **That staff time was excluded correctly.** Model-only staff exclusion is weak. Review the moments behind any headline number, especially early-program videos where instructors talk a lot.
4. **That the comparison is fair.** Early vs showcase compares different videos, of different lengths and filmed differently. Report it as a description of these videos, not as proof of change.
5. **The wording.** Describe observable activity ("time spent performing", "moments with visible smiles"), never feelings, ability or individual youth.
6. **That no clip, still or name goes out** without the consent TPA already requires for its footage.

## Decisions we made

- **We did not use face-based age estimation to exclude instructors.** It is unreliable at telling 15–17-year-olds from people in their early 20s, which is exactly the boundary that matters here. It would also be a biometric inference about minors, which contradicts our no-profiling principle. Instead we use:
  - **a performer-size filter**: [TODO: describe it once it is implemented; it is not yet recorded in the repo];
  - **a learned model** on per-segment signals (no images, no embeddings);
  - **a stated limitation**: model-only staff exclusion is weak and is reliable mainly on reviewed clips.
- **Blind labels come before review.** Evaluation labels were collected with no model output on screen, so the labeler could not be anchored by the AI.
- **Eval videos are held out.** Thresholds are calibrated on dev videos only. Eval videos stay hidden in review mode until blind labeling is finished.
- **Every review is append-only and versioned**, so any number can be traced to who reviewed what, when, under which guideline version.
- **Staff own the observation vocabulary.** Staff can add observation types in config without touching the labels used for evaluation, and the app refuses vocabulary changes made without a guideline-version bump.

## Deployment notes

- **Swap the people detector before deployment.** The YOLO11 weights and library are AGPL-3.0. That is fine for this project, but handing the pipeline to TPA or publishing it brings AGPL terms. Replace it with a permissively licensed person detector, or confirm the licence with the organizers.
- **Run locally.** The pipeline and review app run on a machine TPA owns (a 16 GB laptop is the target). The app listens on 127.0.0.1 only, never creates a share link and sends no telemetry.
- **Keep the rules in config.** Labels, observation types, thresholds and the colour theme are configuration. Changing the label vocabulary requires a guideline-version bump.
- **Verify licences** for every model against its upstream page before the final hand-off (`docs/weights.md`).
