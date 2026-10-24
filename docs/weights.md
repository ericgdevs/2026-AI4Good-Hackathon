# Model weights (004 signal stages)

These are open weights. They are downloaded once (`scripts/download_weights.sh`, on the HPC login node and once on the Mac) and run locally and offline: jobs set `HF_HUB_OFFLINE=1` and `YOLO_OFFLINE=1`, and nothing is downloaded inside a job.

The weights not hosted on Hugging Face (YOLO11 and EmotiEffLib) come from the projects' official GitHub releases. This is a clarification of Constitution Principle II (open weights, downloaded, run locally), not an amendment (spec FR-009a). Those files are pinned by sha256 in `config/pipeline.yaml`, and a missing, unpinned or mismatched file stops the command before any video is processed.

The licences below are as published upstream. **Verify each one against the upstream page before the model card is final.**

| stage | weight | source | version / pin | licence | used on |
|---|---|---|---|---|---|
| transcript | faster-whisper large-v3 | HF `Systran/faster-whisper-large-v3` | HF snapshot in `~/hf-cache` | MIT (OpenAI Whisper weights) | HPC (CUDA, float16) |
| transcript | faster-whisper small | HF `Systran/faster-whisper-small` | HF snapshot | MIT | Mac (CPU, int8) |
| audio | AST AudioSet 10-10-0.4593 | HF `MIT/ast-finetuned-audioset-10-10-0.4593` | HF snapshot | BSD-3-Clause | HPC and Mac |
| expression (faces) | YuNet 2023mar | HF `opencv/face_detection_yunet`, `face_detection_yunet_2023mar.onnx` | HF snapshot | MIT | HPC and Mac (CPU) |
| expression | EmotiEffLib `enet_b0_8_va_mtl` (ONNX) | GitHub `sb-ai-lab/EmotiEffLib`, `models/affectnet_emotions/onnx/` | sha256: TBD (run `scripts/download_weights.sh`) | Apache-2.0 | HPC and Mac (CPU) |
| people | YOLO11m | GitHub `ultralytics/assets` release v8.3.0, `yolo11m.pt` | sha256: TBD | **AGPL-3.0** (Ultralytics) | HPC |
| people | YOLO11n | GitHub `ultralytics/assets` release v8.3.0, `yolo11n.pt` | sha256: TBD | **AGPL-3.0** (Ultralytics) | Mac |

**Licence note**: the Ultralytics weights and library are AGPL-3.0. Running them inside this project is fine. Handing the pipeline to TPA, or publishing it, means AGPL terms apply to the people stage. Check this with the organizers before the final hand-off.

## What each stage keeps

| stage | stored per segment | never stored |
|---|---|---|
| transcript | `has_speech`, `transcript_chars`. The full text stays only in the run's `transcript.json`, on HPC (never synced, printed or committed) | text in logs, Parquet or errors |
| audio | one score per audio label (`singing_voice`, `music`, `speech`, `applause`, `silence`) | raw AudioSet class scores |
| people | `person_count` (median over sampled frames) | boxes, tracks, per-frame counts |
| expression | `faces_detected` (most usable faces in one frame), mean score per expression label (`positive`, `neutral`, `negative`, `upset`) | faces, crops, landmarks, embeddings, per-face rows, valence and arousal |

## Known limitations (for the model card)

- **`person_count` includes the audience.** Every visible person is counted, not only participants.
- **`has_speech` can include sung lyrics.** It means "words were transcribed", not speech as opposed to singing.
- **Expression labels are unvalidated suggestions of observable expressions**, never feelings or mental state. Fear and Surprise are dropped, because both misfire on open mouths while singing. Scores are not renormalized after dropping. `confident` is not model-scored (it is derived later).
- **Expression needs a face of at least 48 px** (`min_face_px`) with a detector score of at least 0.8. Segments without one get `faces_detected = 0` and no expression scores.
- **Transcript peak memory on CUDA is approximate**: CTranslate2 allocates GPU memory outside torch's counters.
- **People and expression share one frame pass**, so their runs record the same runtime and peak memory. Do not sum them.
- **faster-whisper on HPC needs the CUDA 12 cuBLAS and cuDNN 9 libraries** (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, Linux only). `scripts/slurm/signals.sbatch` puts them on `LD_LIBRARY_PATH`.
