# Contract: Python API and CLI

Later features depend on these names. Plain functions only; no class hierarchies.

## Python (`tpa_index`)

```python
# paths.py
repo_root() -> Path
data_dir() -> Path             # from .env; raises GuardError if unset or inside repo
output_dir() -> Path           # same rules
check_guard() -> None          # validates both; message names the setting, never the path
redact(text: str) -> str
install_redaction() -> None    # logging filter + sys.excepthook; every entry point calls it first

# config.py
load_labels(path=None) -> dict
load_pipeline(path=None) -> dict
load_videos(path=None) -> list[dict]    # local file if present, else example
effective_settings(pipeline, video) -> dict   # window_s, overlap_s, frames_per_window after overrides
config_hash(labels, pipeline) -> str    # 12 hex chars

# schema.py
SCHEMAS: dict[str, pa.Schema]           # videos, segments, predictions, moments, runs, reviews
validate(table: str, data, phases=None) -> pa.Table          # raises SchemaError(table, column, reason)
write_table(table: str, data, run_id: str) -> Path           # validate, then write {OUTPUT_DIR}/runs/{run_id}/{table}.parquet; refuses to overwrite
read_table(table: str, root=None) -> pa.Table                # all complete runs (dirs with runs.parquet), via DuckDB glob
append_reviews(rows) -> int                                  # validate + INSERT into reviews.duckdb
read_reviews() -> pa.Table
validate_dataset(root) -> None                               # cross-table reference checks

# segment.py
discover_videos(root=None) -> list[Path]       # skips hidden and ._ files
windows(duration_s, window_s, overlap_s) -> list[tuple[float, float]]
segment_video(video: dict, settings: dict) -> Iterator[Window]
#   Window = NamedTuple(row: dict, frames: np.ndarray[N, S, S, 3] uint8 letterboxed, audio: np.ndarray[float32] 16 kHz mono)
probe(path) -> dict                            # duration_s, fps, orientation, has_audio

# runs.py
track_run(stage, model=None, config_hash=...) -> ContextManager[RunContext]
#   ctx.run_id; ctx.n_segments = ...; on success writes runs.parquet (the run's "complete" marker);
#   on error writes no runs row and re-raises, so read_table skips the incomplete run

# moments.py
merge_moments(segments, predictions, pipeline, run_id) -> list[dict]

# mock.py
generate(out_root, seed=0, n_videos=4) -> str  # returns run_id
```

## CLI (`tpa-index`)

| command | does | exit codes |
|---|---|---|
| `tpa-index check` | guard, config validity, device (cuda/mps/cpu), hardware name; no paths printed | 0 ok, 2 guard failure, 3 config error |
| `tpa-index mock [--seed N] [--n-videos N]` | writes a full mock dataset under `OUTPUT_DIR`, then validates it | 0 / 2 / 3 |
| `tpa-index segment [--videos PATH] [--video-id ID ...]` | segments mapped videos (all if none given), writes videos + segments + runs | 0 / 2 / 3 / 1 on processing error |

Every command calls `install_redaction()` and then `check_guard()` before any other work.

## Slurm

`sbatch scripts/slurm/job.sbatch <command...>`. Example: `sbatch scripts/slurm/job.sbatch tpa-index check`.
