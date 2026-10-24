# Data Model: Extra Per-Segment Signals (004)

**No schema change.** Every table and column below already exists in `src/tpa_index/schema.py` (the 001 contract). This feature only fills nullable columns and adds rows.

## Stage run folder

`{OUTPUT_DIR}/runs/{run_id}/`, one per (video, stage), `run_id = {stage}-{YYYYmmddTHHMMSS}-{4 hex}` (from `track_run`).

| file | transcript | audio | people | expression |
|---|---|---|---|---|
| `videos.parquet` (1 row) | ✓ | ✓ | ✓ | ✓ |
| `segments.parquet` | ✓ | ✓ | ✓ | ✓ |
| `predictions.parquet` | – | ✓ | – | ✓ (only if ≥ 1 segment had a usable face; it may be empty) |
| `transcript.json` | ✓ | – | – | – |
| `runs.parquet` (written last: marks the run complete) | ✓ | ✓ | ✓ | ✓ |

No other file may exist in a stage run folder (privacy test SC-001).

## segments rows written by a stage

One row per official segment of the video. `segment_id`, `video_id`, `start_s` and `end_s` are copied from the official segments. Of the optional columns, only the stage's own are filled, and every other optional column is null.

| stage | fills | rule |
|---|---|---|
| transcript | `has_speech` (bool), `transcript_chars` (int32 ≥ 0) | words whose midpoint lies in [start_s, end_s); chars = Σ len(word.strip()) |
| audio | – (all optional columns null) | the rows exist so `_official_by_video` and resume see the segments |
| people | `person_count` (int32 ≥ 0) | floor(median of per-frame person counts at conf ≥ 0.4) |
| expression | `faces_detected` (int32 ≥ 0) | max usable faces (score ≥ 0.8, min side ≥ `min_face_px`) in any one sampled frame |

## predictions rows written by a stage

| stage | label_group | labels | model | score |
|---|---|---|---|---|
| audio | `audio` | singing_voice, music, speech, applause, silence (every audio label, every segment) | `ast-finetuned-audioset-10-10-0.4593` | max sigmoid over the label's mapped AudioSet classes |
| expression | `expression` | positive, neutral, negative, upset (only segments with `faces_detected ≥ 1`) | `enet_b0_8_va_mtl` | mean over usable faces of Σ softmax prob of the label's mapped classes |

Primary key `(segment_id, label, model, run_id)`, scores in 0–1, enforced by the existing `validate`. `confident` gets no rows.

## runs row

The existing columns, with these values:

| column | value |
|---|---|
| stage | `transcript` \| `audio` \| `people` \| `expression` |
| model | `faster-whisper-<size>` \| AST short name \| `yolo11<n/m/l>` \| `enet_b0_8_va_mtl` |
| config_hash | `stage_config_hash(stage, labels, pipeline, video)` (see research R1) |
| runtime_s, peak_mem_gb | people and expression share one pass, so both report that pass's values (overlapping, do not sum) |

## transcript.json (HPC or Mac scratch only, never synced, never printed)

```json
{"video_id": "v01", "model": "faster-whisper-large-v3", "language": "en",
 "segments": [{"start": 0.0, "end": 3.2, "text": "...",
               "words": [{"start": 0.1, "end": 0.4, "word": "..."}]}]}
```

`video_id` is the opaque ID. There is no filename, path or speaker field, and no speaker diarization.

## Config entities (new keys; see contracts/config.md)

- `models.transcript|audio|people|expression`: per-hardware model choice, like `models.main`.
- `signals.audio.mapping`: `{audio label: [AudioSet display name, ...]}`.
- `signals.expression.mapping`: `{expression label: [model class, ...]}`. Classes left out are dropped.
- `signals.people.person_conf`, `signals.expression.face_score`, and the existing `min_face_px`.
- `config/official_runs.yaml`: new lists `transcript`, `audio`, `people`, `expression`.

## labels.yaml change

- `guideline_version: "0.5.0"`.
- New label: `{name: singing_voice, group: audio, prompts: ["a person singing", "people singing together", "a person rapping into a microphone", "a choir singing"], color: <unused hex>}`.
- `music`, `speech`, `applause` and `silence` are unchanged. No other edits.

## Lifecycle

```
(video, stage) pending
  ├─ complete run exists with same stage+model+config_hash and no --force → skipped
  └─ run → write videos, segments, [predictions], [transcript.json] → write runs.parquet → complete
           └─ error before runs.parquet → incomplete folder, invisible to every read; the next run redoes it
```

The temp WAV exists only between "first audio stage starts" and "last audio stage ends" for one video (deleted in `finally`). Face crops exist only inside one frame's processing.
