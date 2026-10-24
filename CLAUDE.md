# 2026-AI4Good-Hackathon Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-10-23

## Active Technologies
- Python 3.12, uv, package `tpa_index` (from 001) + transformers 5.20 (`AutoModel` and `AutoProcessor`, weights loaded from local files only), torch 2.14 (cu130 on Linux, MPS on Mac), and the 001 stack (PyAV, pyarrow, DuckDB, PyYAML). No new dependencies. (002-visual-pipeline)
- The existing 001 Parquet run folders and reviews DuckDB, with no schema changes. Adds `config/official_runs.yaml`. (002-visual-pipeline)

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
- 002-visual-pipeline: Added Python 3.12, uv, package `tpa_index` (from 001) + transformers 5.20 (`AutoModel` and `AutoProcessor`, weights loaded from local files only), torch 2.14 (cu130 on Linux, MPS on Mac), and the 001 stack (PyAV, pyarrow, DuckDB, PyYAML). No new dependencies.

- 001-foundation: Added Python 3.12, managed with uv; package `tpa_index` (src layout)

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
