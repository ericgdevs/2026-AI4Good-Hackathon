# 2026-AI4Good-Hackathon Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-10-24

## Active Technologies
- Python 3.12, uv, package `tpa_index` (from 001) + transformers 5.20 (`AutoModel` and `AutoProcessor`, weights loaded from local files only), torch 2.14 (cu130 on Linux, MPS on Mac), and the 001 stack (PyAV, pyarrow, DuckDB, PyYAML). No new dependencies. (002-visual-pipeline)
- The existing 001 Parquet run folders and reviews DuckDB, with no schema changes. Adds `config/official_runs.yaml`. (002-visual-pipeline)
- Python 3.12 and uv. The package is `tpa_index` (001/002). + Gradio 6.30 (already locked), ffmpeg (system, macOS Homebrew, with libx264 and aac), and the 001/002 stack. No new Python dependencies. `psutil` (already locked) is used by one test. (003-review-app)
- The existing local `reviews.duckdb`, plus a new `blind_queue` table in the same file. Inputs are read through `read_official`. (003-review-app)
- Python 3.12, uv, package `tpa_index` (src layout) + all already locked in `pyproject.toml`. faster-whisper 1.2.1 (ctranslate2 4.8.3, bundled silero VAD), transformers 5.20 (`ASTForAudioClassification`), ultralytics 8.4.184, opencv-python 5.0.0.93 (`FaceDetectorYN`, YuNet), emotiefflib 1.1.1 with its ONNX engine (onnx 1.23.2, onnxruntime 1.31.0), soundfile, PyAV, psutil. **No new dependencies.** YuNet needs nothing beyond opencv. EmotiEffLib's torch engine would need `timm`, so the ONNX engine is used instead (research R8). (004-signals)
- existing per-run Parquet folders, plus `transcript.json` in transcript run folders (HPC or Mac scratch only). No schema change. `config/official_runs.yaml` gains four empty lists. (004-signals)

- Python 3.12, managed with uv; package `tpa_index` (src layout) (001-foundation)

## Project Structure

```text
src/
tests/
```

## Commands

uv run pytest

## Code Style

Python 3.12, managed with uv; package `tpa_index` (src layout): Follow standard conventions

## Recent Changes
- 004-signals: Added Python 3.12, uv, package `tpa_index` (src layout) + all already locked in `pyproject.toml`. faster-whisper 1.2.1 (ctranslate2 4.8.3, bundled silero VAD), transformers 5.20 (`ASTForAudioClassification`), ultralytics 8.4.184, opencv-python 5.0.0.93 (`FaceDetectorYN`, YuNet), emotiefflib 1.1.1 with its ONNX engine (onnx 1.23.2, onnxruntime 1.31.0), soundfile, PyAV, psutil. **No new dependencies.** YuNet needs nothing beyond opencv. EmotiEffLib's torch engine would need `timm`, so the ONNX engine is used instead (research R8).
- 003-review-app: Added Python 3.12 and uv. The package is `tpa_index` (001/002). + Gradio 6.30 (already locked), ffmpeg (system, macOS Homebrew, with libx264 and aac), and the 001/002 stack. No new Python dependencies. `psutil` (already locked) is used by one test.
- 002-visual-pipeline: Added Python 3.12, uv, package `tpa_index` (from 001) + transformers 5.20 (`AutoModel` and `AutoProcessor`, weights loaded from local files only), torch 2.14 (cu130 on Linux, MPS on Mac), and the 001 stack (PyAV, pyarrow, DuckDB, PyYAML). No new dependencies.


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
