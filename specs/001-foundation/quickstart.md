# Quickstart: Shared Data Foundation

## 1. Install (Mac or UNF HPC)

```bash
uv sync                      # Linux pulls CUDA 13.0 torch wheels, macOS pulls default (MPS) wheels
```

On HPC, keep uv's cache off a small home quota: `export UV_CACHE_DIR=/path/to/scratch/uv-cache`.

## 2. Configure paths

Create `.env` in the repo root (it is gitignored):

```bash
DATA_DIR=/absolute/path/outside/repo/videos
OUTPUT_DIR=/absolute/path/outside/repo/outputs
```

Both paths must be outside the repo, or every command exits with code 2.

Real video metadata goes in `config/videos.local.yaml`, using the format of `config/videos.example.yaml`. It is gitignored.

Assistant read-deny rules are committed in `.claude/settings.json` (`permissions.deny`). They cover the data and output locations, `config/videos.local.yaml`, and media files. If you put `DATA_DIR` or `OUTPUT_DIR` somewhere new, add a matching `Read(//abs/path/**)` rule there, and do not create a separate local settings file.

## 3. Verify

```bash
uv run tpa-index check       # prints device, hardware, guard OK, configs OK
uv run pytest                # mock data + synthetic clips only
```

## 4. Build without footage

```bash
uv run tpa-index mock --seed 0
```

```python
from tpa_index.schema import read_table, read_reviews
segments = read_table("segments")     # all six tables are available the same way
```

## 5. Segment videos

```bash
uv run tpa-index segment                    # all videos listed in videos.local.yaml
uv run tpa-index segment --video-id v01
```

## 6. On UNF HPC

```bash
sbatch scripts/slurm/job.sbatch tpa-index check
sbatch scripts/slurm/job.sbatch tpa-index segment
```

Job logs go to `slurm-tpa-index-<jobid>.out` (gitignored). They are already redacted.
