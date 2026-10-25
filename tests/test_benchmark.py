"""006 US5: Mac benchmark command, with a fake stage runner (no models, no real video)."""
import functools
import json
import re
from datetime import datetime, timezone

import pytest

from tpa_index import benchmark, schema
from tpa_index.paths import repo_root
from tests.test_schema import row

VIDEO = "mock05"
MAC_RUNTIME = {"segment": 3.0, "predict": 30.0, "transcript": 12.0, "audio": 6.0, "people": 9.0, "expression": 15.0,
               "moments": 0.5}
MAC_WALL = {s: rt + (20.0 if s not in ("segment", "moments") else 0.2) for s, rt in MAC_RUNTIME.items()}  # + loading


def _write_run(run_id, stage, video_id, duration_s, runtime_s, hardware, model, peak=1.5):
    schema.write_table("videos", [row("videos", video_id=video_id, filename=f"mock/{video_id}.mp4",
                                      duration_s=duration_s)], run_id)
    schema.write_table("runs", [row("runs", run_id=run_id, stage=stage, model=model, hardware=hardware,
                                    runtime_s=runtime_s, peak_mem_gb=peak,
                                    started_at=datetime(2026, 10, 24, tzinfo=timezone.utc))], run_id)


def fake_runner(fail=()):
    calls = []

    def run(argv):
        calls.append(argv)
        stage = argv[2] if argv[0] == "signals" else argv[0]
        video = argv[argv.index("--video-id") + 1]
        if stage in fail:
            raise RuntimeError(f"{stage} broke on /Users/someone/secret/{video}.mp4")
        _write_run(f"{stage}-mac-{len(calls)}", stage, video, 300.0, MAC_RUNTIME[stage], "Apple M-test",
                   f"mac-{stage}-model", peak=2.0 + len(calls))
        return 0

    run.calls = calls
    return run


def fake_clock():
    ticks = []
    for i, stage in enumerate(benchmark.STAGES):
        ticks += [1000.0 * i, 1000.0 * i + MAC_WALL[stage]]
    it = iter(ticks)
    return lambda: next(it)


@pytest.fixture
def hpc_official(tmp_path, monkeypatch):
    """Official HPC runs for every stage but expression: two videos of 1 h and 0.5 h."""
    official = {}
    for stage in benchmark.STAGES:
        if stage == "expression":
            continue
        ids = []
        for vid, hours, rt in (("v01", 1.0, 60.0), ("v02", 0.5, 30.0)):
            rid = f"{stage}-hpc-{vid}"
            _write_run(rid, stage, vid, hours * 3600, rt, "NVIDIA test GPU", f"hpc-{stage}-model", peak=8.0)
            ids.append(rid)
        official[stage] = ids
    path = tmp_path / "official.yaml"
    path.write_text("".join(f"{s}: [{', '.join(ids)}]\n" for s, ids in official.items()))
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)
    return path


def test_steps_order_and_arguments():
    got = benchmark.steps(VIDEO)
    assert [s for s, _ in got] == list(benchmark.STAGES)
    assert got[2][1] == ["signals", "--stages", "transcript", "--video-id", VIDEO, "--force"]
    assert all(argv[argv.index("--video-id") + 1] == VIDEO for _, argv in got)
    assert benchmark.steps(VIDEO, "v.yaml")[0][1] == ["segment", "--video-id", VIDEO, "--videos", "v.yaml"]


def test_benchmark_summary(hpc_official):
    """SC-007: seven stages recorded, loading separated from processing, per-video-hour on both machines."""
    shipped = (repo_root() / "config/official_runs.yaml").read_bytes()
    official_before = hpc_official.read_bytes()
    runner = fake_runner()
    results = benchmark.run_benchmark(VIDEO, runner=runner, clock=fake_clock())
    assert len(runner.calls) == 7 and all(len(r["run_ids"]) == 1 and r["error"] is None for r in results)

    summary = benchmark.summarize(results)
    rows = {r["stage"]: r for r in summary["rows"]}
    assert list(rows) == [*benchmark.STAGES, "total"]
    for stage in benchmark.STAGES:
        r = rows[stage]
        assert r["mac_processing_s"] == MAC_RUNTIME[stage]
        assert r["mac_load_s"] == pytest.approx(MAC_WALL[stage] - MAC_RUNTIME[stage])
        assert r["mac_s_per_video_hour"] == pytest.approx(MAC_RUNTIME[stage] / (300 / 3600))
        assert r["mac_peak_mem_gb"] > 0 and r["mac_model"] == f"mac-{stage}-model" and r["mac_hardware"] == "Apple M-test"
    assert rows["predict"]["hpc_s_per_video_hour"] == pytest.approx(90.0 / 1.5)  # summed runtime / summed hours
    assert rows["predict"]["hpc_peak_mem_gb"] == 8.0 and rows["predict"]["hpc_hardware"] == "NVIDIA test GPU"
    assert rows["predict"]["status"] == "ok"
    assert rows["expression"]["status"] == "no HPC run" and rows["expression"]["hpc_s_per_video_hour"] is None

    total = rows["total"]
    assert total["mac_s_per_video_hour"] == pytest.approx(sum(MAC_RUNTIME.values()) / (300 / 3600))
    assert total["mac_load_s"] == pytest.approx(sum(MAC_WALL.values()) - sum(MAC_RUNTIME.values()))
    assert total["mac_peak_mem_gb"] == max(rows[s]["mac_peak_mem_gb"] for s in benchmark.STAGES)
    assert total["status"] == "incomplete: missing expression"
    assert any("excludes model loading" in n for n in summary["notes"])

    json_path, md_path = benchmark.write_summary(summary)
    for text in (json_path.read_text(), md_path.read_text()):
        assert ".mp4" not in text and not re.search(r"/(Users|home|private|tmp|Volumes)\b", text)
    assert json.loads(json_path.read_text())["rows"][0]["stage"] == "segment"
    assert "Mac s / video-hour (processing)" in md_path.read_text()
    assert (repo_root() / "config/official_runs.yaml").read_bytes() == shipped
    assert hpc_official.read_bytes() == official_before


def test_failure_is_recorded_and_later_steps_run(hpc_official, monkeypatch, capsys):
    from tpa_index import cli

    runner = fake_runner(fail=("audio",))
    monkeypatch.setattr(benchmark, "run_benchmark",
                        functools.partial(benchmark.run_benchmark, runner=runner, clock=fake_clock()))
    assert cli.main(["benchmark", "--video-id", VIDEO]) == 1
    assert len(runner.calls) == 7  # people, expression and moments still ran after audio failed
    out = capsys.readouterr().out
    assert "failed: RuntimeError" in out and "/Users/someone" not in out
    assert "incomplete: missing audio, expression" in out
    summaries = list((schema._runs_root(None).parent / "benchmarks").glob("benchmark-*.json"))
    assert len(summaries) == 1
