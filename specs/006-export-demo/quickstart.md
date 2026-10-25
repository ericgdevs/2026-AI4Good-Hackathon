# Quickstart: 006 (validation walkthrough)

Steps 1–5 use mock data only and are safe for development and for assistants. Step 6 is run by the team on real data, after implementation, and is never run by an assistant.

## 1. Tests

```bash
uv run pytest tests/test_export.py tests/test_benchmark.py tests/test_theme.py tests/test_guideline.py tests/test_demo.py tests/test_insights.py
uv run pytest            # full suite; the slow marker stays off
```

## 2. Devpost sample export (mock only)

```bash
uv run tpa-index export --sample
git diff --stat docs/samples/          # regenerated sample; a test fails if it drifts from the code
```

Check:
- about 20 records;
- all five `review_decision` values;
- one staff-added label and one observation;
- `mock01`–`mock04` IDs only.

## 3. Export on a mock dataset

```bash
export OUTPUT_DIR=$(mktemp -d) DATA_DIR=$(mktemp -d) TPA_OFFICIAL_RUNS=$(mktemp)
uv run tpa-index mock --clips
# list the mock run IDs in $TPA_OFFICIAL_RUNS (moments/predict), as tests do
uv run tpa-index export
uv run tpa-index export --verified-only
ls "$OUTPUT_DIR/exports"               # two timestamped pairs + annotations-latest.*
```

## 4. Staff-defined observation label

On a copy of the label config (tests do this in a temp dir):

1. Add `- {name: interview, description: "a youth speaking to camera about their work"}` under `observation_types`, without changing `guideline_version`. `tpa-index check` refuses: "vocabulary changed without bumping guideline_version".
2. Bump `guideline_version` to `0.6.0`. `check` passes and prints the line to add to `config/guideline_history.yaml`.
3. Start the app on mock data. Review → Add observation now lists `interview`. Blind labeling looks exactly as before.

## 5. Demo screens on mock data

```bash
uv run tpa-index app                   # with the mock OUTPUT_DIR/DATA_DIR from step 3 only
```

- **Timeline**: pick "Video 1 · early". Signals per segment; the expression shows "AI suggestion, unvalidated: …"; staff-led rows are hidden by default. No transcript text, no filenames.
- **Insights**: with no `results/` folder it shows "not available yet". Copy the mock 005 fixture from `tests/fixtures/results/` into `$OUTPUT_DIR/results/` and reload: four charts, every number tagged.
- Switch the OS to dark mode: the app stays light.

## 6. Mac benchmark (team only, real promo, after implementation)

```bash
uv run tpa-index benchmark --video-id <promo id>   # id typed at run time, never committed
uv run tpa-index runs                               # seven new rows with runtime and peak memory
```

The summary is written to `$OUTPUT_DIR/benchmarks/`. Paste only the aggregate per-video-hour table into the deck. Never add benchmark runs to `config/official_runs.yaml`.
