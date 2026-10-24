# Quickstart: Evaluation and Program Indicators (005)

## Before running on real data (team, after 004 runs are official)

1. Make sure `config/official_runs.yaml` lists the 004 stages (transcript, audio, people, expression) for every mapped video, identically on HPC and in the repo.
2. On HPC, calibrate on dev videos only, then sync:
   ```bash
   uv run tpa-index calibrate --stage audio --split dev
   uv run tpa-index calibrate --stage expression --split dev
   ```
   Then on the Mac, run `scripts/sync_outputs_from_hpc.sh`. It merges the thresholds into the local thresholds file.
3. If anything is missing, `evaluate` and `insights` stop with exit 3 and list every missing stage, video and threshold.

## Development (Mac, mock data only)

```bash
uv run pytest tests/test_view.py tests/test_rules.py tests/test_evaluate.py tests/test_indicators.py tests/test_report.py
```

## Run

```bash
uv run tpa-index evaluate --dry-run     # headline metrics with 95% CIs, nothing written
uv run tpa-index evaluate               # -> OUTPUT_DIR/results/evaluate-<ts>-<hex>/
uv run tpa-index insights --dry-run
uv run tpa-index insights               # -> OUTPUT_DIR/results/insights-<ts>-<hex>/
```

006 reads the newest complete folder of each kind (the one with `manifest.json`; see `contracts/results.md`).
