# Implementation Plan: Extra Per-Segment Signals

**Branch**: `004-signals` | **Date**: 2026-10-24 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/004-signals/spec.md`

## Summary

Add one command, `tpa-index signals --stages …`, that runs four stages per video. Each stage writes its own immutable run into the existing contract tables:
- **transcript**: faster-whisper over the whole audio track, with VAD. It fills `has_speech` and `transcript_chars`, and writes the text only to `transcript.json`.
- **audio**: AST on each window. It writes `predictions` rows in the `audio` group through a config mapping, with the new `singing_voice` label.
- **people**: YOLO11, person class only. It fills `person_count`.
- **expression**: YuNet faces, then EmotiEffLib ONNX in memory. It fills `faces_detected` and writes `expression` predictions only when a usable face exists.

Transcript and audio share one temporary WAV. People and expression share one streaming frame pass. Each model loads once per job.

The resume key is a per-stage config hash: the existing `config_hash` over only the keys the stage uses. A new `read_signals` joins the four columns onto the official segments. `calibrate --stage audio|expression` works through the existing code.

The 002 official runs, the default `read_official`, the moments rules and blind mode are untouched. Tests prove each of these.

## Technical Context

**Language/Version**: Python 3.12, uv, package `tpa_index` (src layout)

**Primary Dependencies**: all already locked in `pyproject.toml`. faster-whisper 1.2.1 (ctranslate2 4.8.3, bundled silero VAD), transformers 5.20 (`ASTForAudioClassification`), ultralytics 8.4.184, opencv-python 5.0.0.93 (`FaceDetectorYN`, YuNet), emotiefflib 1.1.1 with its ONNX engine (onnx 1.23.2, onnxruntime 1.31.0), soundfile, PyAV, psutil. **No new dependencies.** YuNet needs nothing beyond opencv. EmotiEffLib's torch engine would need `timm`, so the ONNX engine is used instead (research R8).

**Storage**: existing per-run Parquet folders, plus `transcript.json` in transcript run folders (HPC or Mac scratch only). No schema change. `config/official_runs.yaml` gains four empty lists.

**Testing**: pytest. Default tests use fakes and synthetic clips only (no model loaded). `slow` tests load one real model at a time, at the smallest size.

**Target Platform**: UNF HPC (RTX PRO 6000 Blackwell, CUDA 13, Slurm) and the Mac (MPS/CPU, 16 GB-class, with the labeling app running).

**Project Type**: CLI and library (single project).

**Performance Goals**: each of the two Slurm jobs covers all mapped videos within `--time=03:00:00` on one GPU (SC-008). If not, a resubmit finishes the remaining pairs.

**Constraints**: no network at run time; weights from the local cache, sha256-pinned for GitHub assets; no identity data; transcript text never printed; peak memory recorded per run.

**Scale/Scope**: about 12 videos (the official run count), 4 stages, so about 48 runs. Hundreds of windows per video.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design (below).*

| Principle | Check | Result |
|---|---|---|
| I. Data local or UNF HPC | Runs on HPC or the Mac only. Transcripts stay on HPC (Parquet-only sync, test SC-011). Temp WAV in the job's `$TMPDIR`, deleted in `finally`. | ✅ |
| II. Open models, local execution | All open weights, run locally, offline. YOLO11 and EmotiEffLib come from official GitHub releases, a clarification recorded in the spec (FR-009a). Pinned by sha256 and listed in `docs/weights.md`. Runtime and peak memory recorded per run, on HPC and the Mac. Mac defaults are small (whisper base int8, yolo11n). | ✅ (clarification, not an amendment) |
| III. Assistants never see data | Development uses synthetic clips and fakes only. Nothing reads DATA_DIR or `*.local.yaml` outside the guarded paths. Logs carry IDs and error types only. | ✅ |
| IV. No identity | Detection only. No embeddings, tracks, landmarks or boxes are stored (landmarks dropped on return). Crops exist only in memory. Rows are per segment. Test SC-001. | ✅ |
| V. No judgments about people | Expression labels are the existing observable set. Model class names appear only as mapping keys. `confident` is unscored. Valence and arousal are dropped. No new prompts about feelings. | ✅ |
| VI. Humans verify | Outputs are predictions rows (AI suggestions). No UI in this feature. | ✅ |
| VII. Contracts first | No schema change. Config and interface contracts are in `contracts/`. Mock data and fakes let every stage run without footage. | ✅ |
| VIII. Measure everything | Calibration uses dev videos only (SC-009). No numbers are reported in this feature. | ✅ |
| IX. Simplest thing | Pretrained models, no training. Existing writers, run tracking, official mechanism and calibrate are reused. One new orchestration module and one model-wrapper module. | ✅ |
| X. Presentation-ready | No UI. | n/a |
| XI. Config over code | Models, mappings, thresholds and the minimum face size all live in config. | ✅ |
| XII. One app, two modes | App untouched. | n/a |
| XIII. Deliverables | `docs/weights.md` feeds the model card. `person_count` counting the audience is noted as a limitation. | ✅ |

No violations, so the Complexity Tracking table is not needed.

## Project Structure

### Documentation (this feature)

```text
specs/004-signals/
├── plan.md              # this file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/
│   ├── config.md        # labels/pipeline/official_runs additions, env overrides, validation
│   └── interfaces.md    # Python API, CLI, Slurm script
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks (not created here)
```

### Source Code: every file added or changed

```text
config/
├── labels.yaml               CHANGED  guideline_version 0.5.0; + singing_voice (audio)
├── pipeline.yaml             CHANGED  + models.transcript|audio|people|expression; + signals.{people,expression,audio}
└── official_runs.yaml        CHANGED  + transcript/audio/people/expression: [] (existing sections byte-identical)

src/tpa_index/
├── config.py                 CHANGED  validate new keys; stage_config_hash; weights_dir; pinned_weight; SIGNAL_STAGES
├── runs.py                   CHANGED  RunContext.runtime_s / peak_mem_gb overrides (optional, default None)
├── schema.py                 CHANGED  find_complete_run; read_signals (read_official untouched)
├── signal_models.py          NEW      WhisperTranscriber, AstTagger, YoloPersonCounter, FaceExpression (+ Protocols)
├── signals.py                NEW      assign_words, median_count, aggregate_expression, official_segments,
│                                      extract_audio, scratch_dir, clean_stale_scratch, run_signals, TranscriptError
└── cli.py                    CHANGED  `signals` subcommand; calibrate prints labels without scores (generic)

scripts/
├── slurm/signals.sbatch      NEW      one GPU, 8 CPUs, 64G, 03:00:00, offline env, passes --stages through
└── sync.exclude              CHANGED  + transcript*.json

.gitignore                    CHANGED  + transcript*.json (defence in depth; outputs/ already ignored)
docs/weights.md               NEW      every weight: source, version/sha256, licence (feeds the 006 model card)

tests/
├── fakes.py                  CHANGED  FakeTranscriber (sentinel text), FakeAudioTagger, FakePersonCounter,
│                                      FakeExpressionScorer (0 faces on solid frames), FailingStage
├── data/blind_checklist_0.4.0.json   NEW  frozen checklist from the 0.4.0 labels (generated before the bump)
├── test_signals.py           NEW      pure functions (assign_words, median_count, aggregate_expression, mapping
│                                      max/sum rules), orchestration with fakes: one run per pair, per-stage
│                                      failure isolation, resume/--force, segment-ID mismatch, shared decode
│                                      (frames decoded once, WAV extracted once), no-audio video
├── test_signals_privacy.py   NEW      SC-001 identity columns/files, SC-002 temp/crop dirs empty (success,
│                                      failure, simulated kill + stale cleanup), SC-003 sentinel transcript leak
│                                      (caplog, capsys, exception, Parquet), SC-004 0 faces → no rows,
│                                      SC-005 official 002 runs byte-identical + default reads identical
├── test_signals_slow.py      NEW      @slow, one model per test, smallest size, released after: whisper tiny
│                                      on a tone (no words), AST on silence/tone (mapping names resolve),
│                                      yolo11n on solid frame (0 persons), YuNet+EmotiEffLib on drawn shapes (0 faces)
├── test_config.py            CHANGED  singing_voice valid; mapping validation; stage hash ignores prompt edits and
│                                      other stages' keys but tracks own keys; blind checklist == 0.4.0 snapshot (SC-012)
├── test_predict.py           CHANGED  SigLIP predict writes no audio/expression rows with 0.5.0 labels
├── test_official.py          CHANGED  read_signals join/null/missing-segment error; moments + calibrate rebuild
│                                      official runs whose config_hash differs from the current config
├── test_calibrate.py         CHANGED  --stage audio/expression: dev only (SC-009), confident listed as unscored,
│                                      other models' thresholds kept
└── test_scripts.py           CHANGED  signals.sbatch directives; output-sync filter skips transcript.json (SC-011)
```

**Structure Decision**: single project, same layout as 001–003. The new code is in two modules: `signals.py` holds the pure logic and orchestration, and is fully testable with fakes. `signal_models.py` holds the library wrappers and is only imported when a real model is needed, so the default tests never import torch-heavy model code paths.

## Design notes for tasks

1. **Order**: config and labels (incl. the checklist snapshot, generated *before* the bump) → per-stage hash and `pinned_weight` → `RunContext` overrides → `find_complete_run` and `read_signals` → pure functions → orchestration with fakes → privacy tests → model wrappers and slow tests → CLI, calibrate line, Slurm, sync exclude, docs.
2. **Segment source (FR-004)**: `official_segments` uses the default `read_official("segments")` for the video (official moments or predict), else `stage="segment"`. The stage recomputes `windows(probe duration, effective settings)` and fails the pair on any ID difference.
3. **Shared people and expression pass**: per-stage try/except around `update`. The pass's runtime and peak are captured before `track_run`, because `track_run` resets the CUDA peak, and handed over through the new `RunContext` fields.
4. **Transcript leak guard**: the whole stage body is wrapped. The re-raise uses only the type name, `from None`. The test injects a sentinel through a fake transcriber and, separately, a fake that raises with the sentinel in its message.
5. **Calibrate**: no stage-specific code. With no null labels the floor reports "not applied". With no idle scores the idle line reports "too few, keeps fallback" and nothing idle is written. The one new generic line lists labels in scored groups that have no scores.

## Flags before /speckit-tasks

1. **ctranslate2 on CUDA 13 (HPC)**: the faster-whisper backend wheels target CUDA 12 and cuDNN 9. If loading fails on the node, transcript must fall back to CPU int8 (`models.transcript.hpc_device: cpu`), which is far slower for large-v3 and could exceed 3 h. Smoke-test one promo video first (quickstart step 4).
2. **No system `ffmpeg` assumed on HPC**: the audio extraction uses PyAV's bundled ffmpeg libraries rather than the `ffmpeg` binary. Same output, one less node dependency. This is a small deviation from the literal "ffmpeg" wording.
3. **EmotiEffLib and ultralytics can download silently**: both fetch from GitHub when a file is missing. The plan blocks this with absolute, sha256-checked paths, a patched EmotiEffLib path getter (marked `shortcut:`) and `YOLO_OFFLINE=1`. The committed config ships `sha256: null`, so real runs refuse to start until the team pins the weights on the login node.
4. **Peak memory under-reports for transcript on CUDA**: ctranslate2 allocates outside torch, so `torch.cuda.max_memory_allocated` misses it. The stage sets `RunContext.peak_mem_gb` to the larger of torch's peak and the device memory in use (`torch.cuda.mem_get_info`) just after transcribing. This is approximate, and the model card should say so.
5. **Overlapping runtimes**: people and expression record the same shared-pass runtime and peak, so do not sum them.
6. **`person_count` includes the audience**: this is a model-card limitation, and 005 decides how to use it.
7. **Resume ignores code changes**: `git_commit` is recorded but not part of the key. After a code fix, use `--force`.
8. **`docs/` does not exist yet**: `docs/weights.md` is the first file there. The full model card is a 006 deliverable.
