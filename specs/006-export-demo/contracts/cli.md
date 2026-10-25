# Contract: CLI additions

These are additive only. Existing commands keep their flags and behaviour. 005 adds `evaluate` and `insights` separately.

## `tpa-index export`

```
tpa-index export [--verified-only] [--sample]
```

- **Default**: official moments plus the local review log, written to `{OUTPUT_DIR}/exports/` (contract: export-format.md). It prints the record count per `review_decision` and the two file names (names only, no directory path).
- **`--verified-only`**: leaves out `rejected` records.
- **`--sample`**: ignores OUTPUT_DIR and the review log. It builds about 20 records from a temporary mock dataset with scripted reviews and writes `docs/samples/annotation_export_sample.{json,csv}`. It never reads real data.
- **Exit codes**: `0` ok, `2` guard error, `3` config or schema error (the same as the other commands). With no official moments it exits `3` with "no official moments: list a moments or predict run in config/official_runs.yaml".

## `tpa-index benchmark`

```
tpa-index benchmark --video-id ID [--videos PATH]
```

- Runs these steps for one video, on this machine, in order, each with `--video-id ID`:
  1. `segment`
  2. `predict`
  3. `signals --stages transcript,audio,people,expression --force`
  4. `moments`

  Each step writes its usual runs row. A failing step is recorded and the next steps still run.
- Never edits `config/official_runs.yaml`. The video ID is a run-time argument and is not stored in the repo.
- Writes `{OUTPUT_DIR}/benchmarks/benchmark-{YYYYmmddTHHMMSS}-{4hex}.json` and `.md`, and prints the summary table (data-model.md, Benchmark summary).
- **Requirement**: the video must already be in the official runs, because the 004 stages and moments use official segments and predict output. If it is not, the summary marks those stages "failed: no official segments".
- **Exit codes**: `0` if every step succeeded, `1` if any step failed (the summary is still written), `2` or `3` as above.

## `tpa-index check` (one added line)

After the config checks, it runs `check_guideline(labels)`:

- **Recorded and matching**: prints `guideline: ok (0.5.0)`.
- **Not recorded yet**: prints `guideline: 0.6.0 not recorded; add to config/guideline_history.yaml:  "0.6.0": "<fp>"` (exit 0).
- **Recorded but different**: exits 3 with `labels: vocabulary changed without bumping guideline_version (0.5.0)`.

## `tpa-index app` (behaviour change)

- At launch it calls `check_guideline` and refuses with the same message as `check`.
- New tabs: **Timeline** and **Insights** (ui.md). It is still bound to 127.0.0.1 only.
