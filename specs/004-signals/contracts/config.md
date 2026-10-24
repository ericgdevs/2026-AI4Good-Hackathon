# Contract: Config (004)

This extends `specs/001-foundation/contracts/config.md` and `specs/002-visual-pipeline/contracts/config.md`, and everything there still holds. `config.load_pipeline()` validates the new keys. Any error is a `ConfigError` (exit 3) raised before any video is processed.

## config/labels.yaml

- `guideline_version: "0.5.0"` (was 0.4.0).
- Add after `silence` in the audio block:

```yaml
  - name: singing_voice          # audio cue; not the activity label "singing"
    group: audio
    prompts: ["a person singing", "people singing together", "a person rapping into a microphone", "a choir singing"]
    color: "#c51b8a"
```

The prompts exist only because label validation requires 3–5 prompts for model-scored labels. The audio model never reads them.

## config/pipeline.yaml (additions)

```yaml
models:
  # ...main, motion, batch_windows unchanged...
  transcript:
    hpc: large-v3                # faster-whisper size; resolved as Systran/faster-whisper-<size> in HF_HOME
    mac: base
    hpc_device: cuda             # cpu = int8 fallback if ctranslate2 cannot use the node's CUDA
    compute_type: {cuda: float16, cpu: int8}
  audio:
    hpc: MIT/ast-finetuned-audioset-10-10-0.4593
    mac: MIT/ast-finetuned-audioset-10-10-0.4593
  people:                        # GitHub release assets (ultralytics/assets); file under weights_dir()
    hpc: {file: yolo11m.pt, sha256: null}    # sha256 filled when pre-downloading; null refuses to run
    mac: {file: yolo11n.pt, sha256: null}
  expression:
    face_detector: {repo: opencv/face_detection_yunet, file: face_detection_yunet_2023mar.onnx}   # HF
    model: {name: enet_b0_8_va_mtl, file: enet_b0_8_va_mtl.onnx, sha256: null}   # EmotiEffLib GitHub, ONNX, CPU

signals:
  people:
    person_conf: 0.4
  expression:
    face_score: 0.8              # YuNet score threshold; min face size is the existing min_face_px (48)
    mapping:                     # model class -> label; classes not listed are dropped (Fear, Surprise)
      positive: [Happiness]
      neutral: [Neutral]
      negative: [Anger, Contempt, Disgust]
      upset: [Sadness]
  audio:
    mapping:                     # exact AudioSet display names; label score = max sigmoid over its classes
      singing_voice: [Singing, Choir, Male singing, Female singing, Child singing, Rapping, Humming, Chant]
      music: [Music]
      speech: [Speech, "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking", Conversation, "Narration, monologue"]
      applause: [Applause, Clapping, Cheering]
      silence: [Silence]
```

### Validation rules (load time, no model needed)

- `models.transcript.hpc` and `.mac` are strings. `hpc_device` ∈ {cuda, cpu}. `compute_type` has `cuda` and `cpu`.
- `models.audio.hpc` and `.mac` are strings.
- `models.people.hpc` and `.mac` each have a `file` (a bare filename, no path separators). `sha256` is null or 64 hex characters.
- `models.expression.face_detector` has `repo` and `file`. `model` has `name`, `file` and `sha256` (null or 64 hex).
- `signals.people.person_conf` and `signals.expression.face_score` are in (0, 1).
- `signals.audio.mapping` keys must equal the set of `audio` group labels in labels.yaml. Each value is a non-empty list of strings, and no class appears under two labels.
- `signals.expression.mapping` keys must be `expression` labels. `confident` must not appear (it is derived later). Each value is a non-empty list, and no class appears twice.

### Validation rules (startup, after the model is loaded, before any video)

- Every audio mapping name is in the AST model's `id2label` values. Otherwise exit 3: `signals.audio.mapping: class 'X' not in model labels`.
- Every expression mapping class is in the recognizer's class list. Otherwise exit 3.
- GitHub-sourced files exist under `weights_dir()` and their sha256 is set and matches. Otherwise exit 3: `weights not pinned: set sha256 for yolo11m.pt` or `weights checksum mismatch: ...`.
- Hugging Face models load with `local_files_only=True`. Otherwise exit 3: `model not found locally: ...`.

## config/official_runs.yaml (additions)

```yaml
transcript: []
audio: []
people: []
expression: []
```

These are appended after the existing `moments` list, and the existing sections are left byte-identical. The rules are the same as 002: run IDs only, at most one run per video per stage, and the file is kept identical on HPC and in the repo.

## Environment overrides (tests and HPC)

| variable | default | used for |
|---|---|---|
| `TPA_SCRATCH_DIR` | `$TMPDIR`, else the system temp | temp WAV folder; refused if inside the repo |
| `TPA_WEIGHTS_DIR` | `$HF_HOME/tpa-weights` (HF_HOME defaults to `~/.cache/huggingface`) | GitHub-release weights |
| `YOLO_OFFLINE` | set to `1` by the loader and the Slurm script | stops ultralytics network checks |

## Stage config hash (resume key)

`stage_config_hash(stage, labels, pipeline, video)` reuses `config_hash` over only the keys listed in research R1. Changing any other key (prompts, thresholds, moments rules, app settings, other stages' sections) leaves that stage's hash unchanged.
