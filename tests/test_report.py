"""005 result folders: manifest last, incomplete folders ignored, no leaks, summaries carry CIs and tags."""
import hashlib
import json
import re

import pytest

from tests.insights_fixtures import SECRET, SENTINEL, build_world
from tpa_index.evaluate import run_evaluate
from tpa_index.indicators import run_insights
from tpa_index.report import latest_result, write_result

FORBIDDEN_KEYS = re.compile(r"(?i)(filename|path|person_id|face_id|track|embedding|transcript_text)")


@pytest.fixture
def folders(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch)
    return (write_result("evaluate", run_evaluate(w["pipeline"], w["labels"])),
            write_result("insights", run_insights(w["pipeline"], w["labels"])))


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_no_filename_or_transcript_sentinel_in_any_output(folders):
    """SC-005: no planted filename, no transcript text, no per-person field in any result file."""
    for folder in folders:
        for f in folder.iterdir():
            data = f.read_bytes()
            assert SECRET.encode() not in data and SENTINEL.encode() not in data, f.name
            if f.suffix == ".csv":
                header = data.decode().splitlines()[0].split(",")
                assert not [c for c in header if FORBIDDEN_KEYS.search(c)], f.name
            if f.suffix == ".json" and f.name != "manifest.json":
                assert not [k for k in _keys(json.loads(data)) if FORBIDDEN_KEYS.search(str(k))], f.name


def test_summary_metric_lines_have_intervals(folders):
    """SC-008: every evaluate summary line with an accuracy, F1 or kappa value shows its 95% CI and n; every
    insights number line shows its tag."""
    evaluate, insights = folders
    for line in (evaluate / "summary.md").read_text().splitlines():
        if re.search(r"(accuracy|F1|Kappa|kappa).*\d", line) and "too few to score" not in line \
                and not line.startswith("| system") and "undefined" not in line:
            assert "95% CI" in line and "n=" in line, line
    body = (insights / "summary.md").read_text().split("## Limitations")[0]
    tags = ("staff-verified", "model estimate", "mixed (", "no tag:")
    for line in body.splitlines():
        if re.search(r"\d+(\.\d+)?( ?%| min| per min| pp)", line):
            assert any(t in line for t in tags), line
    assert "observable, unvalidated" in (insights / "summary.md").read_text()


def test_manifest_last_and_hashes_match(folders):
    for folder in folders:
        manifest = json.loads((folder / "manifest.json").read_text())
        newest = max(folder.iterdir(), key=lambda p: p.stat().st_mtime_ns)
        assert newest.name == "manifest.json"
        for name, digest in manifest["files"].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
        assert set(manifest["files"]) == {p.name for p in folder.iterdir()} - {"manifest.json"}


def test_incomplete_folder_ignored(folders, tmp_path):
    evaluate, _ = folders
    assert latest_result("evaluate") == evaluate
    broken = evaluate.parent / "evaluate-29991231T000000-ffff"
    broken.mkdir()
    (broken / "metrics.csv").write_text("x")
    assert latest_result("evaluate") == evaluate  # no manifest.json: not a result
    (broken / "manifest.json").write_text("{}")
    assert latest_result("evaluate") == broken


def test_chart_files_exist(folders):
    evaluate, insights = folders
    for name in ("ablation.png", "per_label_f1.png", "confusion.png", "coverage.png"):
        assert (evaluate / name).stat().st_size > 1000
    for name in ("headline.png", "time_breakdown.png"):
        assert (insights / name).stat().st_size > 1000
