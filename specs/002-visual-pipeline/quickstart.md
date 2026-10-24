# Quickstart: Visual Pipeline

## Mac (benchmark, small model)

```bash
uv run tpa-index predict --video-id v01      # siglip2-base on MPS; one run per video
uv run tpa-index runs                        # find the run_id, hardware, runtime, peak memory
uv run pytest                                # fake embedder; slow model test skipped
uv run pytest -m slow                        # optional: real siglip2-base on a synthetic clip
```

## UNF HPC (official runs, large model)

```bash
# on the Mac: push code (never data, .env, or weights)
scripts/sync_code_to_hpc.sh --dry-run && scripts/sync_code_to_hpc.sh

# on HPC, in ~/tpa-work/repo (weights already in ~/hf-cache)
sbatch scripts/slurm/predict.sbatch                  # all mapped videos
sbatch scripts/slurm/predict.sbatch --video-id v03   # one video

# back on the Mac: pull finished runs (never overwrites local runs or reviews)
scripts/sync_outputs_from_hpc.sh --dry-run && scripts/sync_outputs_from_hpc.sh
uv run tpa-index runs
```

The sync scripts need these keys in `.env`:

```bash
HPC_USER=...
HPC_HOST=...
HPC_OUTPUT_DIR=/path/on/hpc/outputs
```

## Set thresholds from the dev split

```bash
uv run tpa-index calibrate --stage predict --split dev --top-pct 10 --dry-run   # review proposals
uv run tpa-index calibrate --stage predict --split dev --top-pct 10             # writes config/thresholds.local.yaml (gitignored)
uv run tpa-index moments                       # rebuild moments with the new thresholds, without re-running the model
```

Then list the new `moments` runs under `moments:` in `config/official_runs.yaml`.

## Mark which runs count

Edit `config/official_runs.yaml`, listing one predict run per video:

```yaml
predict: [predict-20261024T101500-ab12, predict-20261024T102210-cd34]
```

Result-facing code reads with `read_official("moments")`. Mock data for UI work: `predict: [mock-seed0]`.
