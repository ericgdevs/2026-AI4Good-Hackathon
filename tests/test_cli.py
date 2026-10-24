import os
import subprocess
import sys

from tpa_index.paths import repo_root

RUN = "import sys; from tpa_index.cli import main; sys.exit(main(sys.argv[1:]))"


def run_cli(*args, **env):
    return subprocess.run([sys.executable, "-c", RUN, *args], capture_output=True, text=True,
                          env={**os.environ, **env})


def test_guard_failure_exits_2_without_path():
    inside = str(repo_root() / "outputs")
    result = run_cli("mock", OUTPUT_DIR=inside)
    assert result.returncode == 2
    assert "OUTPUT_DIR" in result.stderr
    assert inside not in result.stdout + result.stderr


def test_mock_command_writes_dataset():
    result = run_cli("mock", "--seed", "0", "--n-videos", "3")
    assert result.returncode == 0, result.stderr
    assert "videos: 3" in result.stdout
    assert os.environ["OUTPUT_DIR"] not in result.stdout + result.stderr


def _clip_yaml(tmp_path, clips):
    path = tmp_path / "videos.yaml"
    entries = "\n".join(
        f"  - {{video_id: {vid}, filename: {vid}.mp4, orientation: landscape, kind: {kind}, phase: early, split: dev}}"
        for vid, kind in (("t01", "live"), ("t02", "clip"), ("t03", "live")))
    path.write_text(f"videos:\n{entries}\n")
    return path


def test_segment_command_is_deterministic(tmp_path, clips):
    from collections import Counter

    import pyarrow.compute as pc

    from tpa_index.schema import read_table

    yaml_path = _clip_yaml(tmp_path, clips)
    for _ in range(2):
        result = run_cli("segment", "--videos", str(yaml_path), DATA_DIR=str(clips))
        assert result.returncode == 0, result.stderr
        assert str(clips) not in result.stdout + result.stderr
    segments, runs = read_table("segments"), read_table("runs")
    assert runs.num_rows == 6 and set(runs["stage"].to_pylist()) == {"segment"}  # one run per video
    counts = Counter(tuple(sorted(r.items())) for r in segments.to_pylist())
    assert set(counts.values()) == {2}  # both invocations produced identical rows
    assert sum(runs["n_segments"].to_pylist()) == segments.num_rows
    videos = read_table("videos")
    assert videos.filter(pc.equal(videos["video_id"], "t02"))["orientation"][0].as_py() == "portrait"  # probed


def test_one_bad_video_keeps_the_others(tmp_path, clips):
    from tpa_index.schema import read_table

    path = tmp_path / "mixed.yaml"
    path.write_text("videos:\n"
                    "  - {video_id: t01, filename: t01.mp4, orientation: landscape, kind: live, phase: early, split: dev}\n"
                    "  - {video_id: t09, filename: missing clip.mp4, orientation: landscape, kind: live, phase: early, split: dev}\n")
    result = run_cli("segment", "--videos", str(path), DATA_DIR=str(clips))
    assert result.returncode == 1
    assert "failed videos: t09" in result.stderr and "missing clip" not in result.stderr
    assert set(read_table("segments")["video_id"].to_pylist()) == {"t01"}
    assert read_table("runs").num_rows == 1


def test_segment_without_local_videos_file_exits_3(monkeypatch, capsys):
    from tpa_index import cli

    monkeypatch.setattr(cli, "install_redaction", lambda: None)
    assert cli.main(["segment"]) == 3
    assert "videos.local.yaml" in capsys.readouterr().err


def test_check_counts_unmapped_videos(monkeypatch, capsys):
    from tpa_index import cli
    from tpa_index.paths import data_dir

    monkeypatch.setattr(cli, "install_redaction", lambda: None)
    (data_dir() / "mock").mkdir()
    (data_dir() / "mock" / "mock_live_01.mp4").write_bytes(b"")  # listed in the example config
    (data_dir() / "Spring Show.mp4").write_bytes(b"")  # not listed
    (data_dir() / "._Spring Show.mp4").write_bytes(b"")  # hidden: ignored
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "unmapped videos: 1" in out and "Spring" not in out


def test_segment_unknown_video_id_exits_3(tmp_path, clips):
    result = run_cli("segment", "--videos", str(_clip_yaml(tmp_path, clips)), "--video-id", "nope",
                     DATA_DIR=str(clips))
    assert result.returncode == 3 and "nope" in result.stderr


def test_segment_processing_error_exits_1_redacted(tmp_path, clips):
    bad = tmp_path / "bad.yaml"
    bad.write_text("videos:\n  - {video_id: t09, filename: missing_clip.mp4, orientation: landscape, kind: live, "
                   "phase: early, split: dev}\n")
    result = run_cli("segment", "--videos", str(bad), DATA_DIR=str(clips))
    assert result.returncode == 1
    assert str(clips) not in result.stderr and "<DATA_DIR>" in result.stderr


def test_check_reports_device_without_paths():
    result = run_cli("check")
    assert result.returncode == 0, result.stderr
    lines = dict(line.split(": ", 1) for line in result.stdout.splitlines())
    assert lines["device"] in {"cuda", "mps", "cpu"} and lines["hardware"] and lines["guard"] == "ok"
    for name in ("DATA_DIR", "OUTPUT_DIR"):
        assert os.environ[name] not in result.stdout + result.stderr


def test_slurm_template():
    path = repo_root() / "scripts/slurm/job.sbatch"
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    text = path.read_text()
    for directive in ("--partition=gpu", "--gres=gpu:6KSE:1", "--cpus-per-task=8", "--mem=32G", "--time=",
                      "--output=slurm-%x-%j.out"):
        assert f"#SBATCH {directive}" in text, directive
    assert 'uv sync --frozen' in text and 'uv run "$@"' in text


def _predict_yaml(tmp_path, extra=""):
    path = tmp_path / "predict.yaml"
    rows = "".join(
        f"  - {{video_id: {vid}, filename: {vid}.mp4, orientation: landscape, kind: {kind}, phase: early, split: dev}}\n"
        for vid, kind in (("t01", "live"), ("t02", "clip"), ("t03", "live")))
    path.write_text("videos:\n" + rows + extra)
    return path


def _patched_cli(monkeypatch):
    from tests.fakes import FakeEmbedder
    from tpa_index import cli

    monkeypatch.setattr(cli, "install_redaction", lambda: None)
    monkeypatch.setattr(cli, "_make_embedder", lambda model_id, dev: FakeEmbedder())
    return cli


def test_predict_one_run_per_video_and_failures(tmp_path, clip_videos, monkeypatch, capsys):
    import re

    from tpa_index.schema import read_table

    cli = _patched_cli(monkeypatch)
    bad = "  - {video_id: t09, filename: missing clip.mp4, orientation: landscape, kind: live, phase: early, split: dev}\n"
    assert cli.main(["predict", "--videos", str(_predict_yaml(tmp_path, bad))]) == 1
    err = capsys.readouterr().err
    assert "failed videos: t09" in err and "missing clip" not in err
    runs = read_table("runs").to_pylist()
    assert len(runs) == 3 and {r["stage"] for r in runs} == {"predict"} and {r["model"] for r in runs} == {"fake"}
    assert all(r["hardware"] and r["n_segments"] > 0 and r["runtime_s"] > 0 for r in runs)
    assert all(re.fullmatch(r"([0-9a-f]{7}(\+dirty)?|nogit)", r["git_commit"]) for r in runs)
    assert set(read_table("moments")["video_id"].to_pylist()) <= {"t01", "t02", "t03"}
    assert read_table("predictions").num_rows > 0


def test_predict_kind_filter(tmp_path, clip_videos, monkeypatch):
    from tpa_index.schema import read_table

    cli = _patched_cli(monkeypatch)
    assert cli.main(["predict", "--videos", str(_predict_yaml(tmp_path)), "--kind", "clip"]) == 0
    assert set(read_table("videos")["video_id"].to_pylist()) == {"t02"}


def test_predict_missing_model_exits_3_before_any_video(tmp_path, clip_videos, monkeypatch, capsys):
    from tpa_index import cli
    from tpa_index.config import ConfigError
    from tpa_index.schema import read_table

    monkeypatch.setattr(cli, "install_redaction", lambda: None)

    def missing(model_id, dev):
        raise ConfigError(f"model not found locally: {model_id}")

    monkeypatch.setattr(cli, "_make_embedder", missing)
    assert cli.main(["predict", "--videos", str(_predict_yaml(tmp_path)), "--model", "google/nope"]) == 3
    assert "model not found locally: google/nope" in capsys.readouterr().err
    assert read_table("runs").num_rows == 0


def test_runs_command_lists_runs_without_paths():
    mock = run_cli("mock", "--n-videos", "1")
    assert mock.returncode == 0, mock.stderr
    result = run_cli("runs")
    assert result.returncode == 0, result.stderr
    assert "mock-seed0" in result.stdout and "mock01" in result.stdout
    assert os.environ["OUTPUT_DIR"] not in result.stdout + result.stderr


# ---- calibrate and moments rebuild (mock data only) ----------------------------------------------------------

def _official_mock(tmp_path, monkeypatch, extra=""):
    """Mock dataset listed as the official predict run; returns the in-process cli."""
    from tpa_index import cli, schema
    from tpa_index.mock import generate

    monkeypatch.setattr(cli, "install_redaction", lambda: None)
    generate(seed=0)
    path = tmp_path / "official.yaml"
    path.write_text("predict: [mock-seed0]\n" + extra)
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)
    return cli


def _crafted_run(label="acting", score=0.001):
    """Copy of the mock run in which one label never scores above `score`."""
    from tpa_index.schema import read_table, write_table

    for table in ("videos", "segments", "predictions", "moments", "runs"):
        rows = read_table(table, run_ids=["mock-seed0"]).to_pylist()
        for r in rows:
            if "run_id" in r:
                r["run_id"] = "crafted"
            if table == "predictions" and r["label"] == label:
                r["score"] = score
        write_table(table, rows if table != "runs" else rows[:1], "crafted")


def test_calibrate_rejects_eval(tmp_path, monkeypatch, capsys):
    cli = _official_mock(tmp_path, monkeypatch)
    assert cli.main(["calibrate", "--split", "eval"]) == 3
    assert "dev split only" in capsys.readouterr().err


def test_calibrate_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    from tpa_index import config

    cli = _official_mock(tmp_path, monkeypatch)
    assert cli.main(["calibrate", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "labels at the floor (min_threshold): none" in out and "idle score_threshold:" in out
    assert not config.THRESHOLDS_LOCAL.exists()


def test_calibrate_writes_labels_idle_and_reports_floor(tmp_path, monkeypatch, capsys):
    import yaml

    from tpa_index import config

    cli = _official_mock(tmp_path, monkeypatch)
    _crafted_run("acting")
    (tmp_path / "official.yaml").write_text("predict: [crafted]\n")
    assert cli.main(["calibrate"]) == 0
    out = capsys.readouterr().out
    assert "labels at the floor (min_threshold): acting" in out
    data = yaml.safe_load(config.THRESHOLDS_LOCAL.read_text())
    assert data["label_thresholds"]["mock-model"]["acting"] == 0.05
    assert set(data["idle_thresholds"]) == {"mock-model"}
    before = data["idle_thresholds"]["mock-model"]
    assert cli.main(["calibrate", "--idle-top-pct", "60"]) == 0
    after = yaml.safe_load(config.THRESHOLDS_LOCAL.read_text())
    assert after["idle_thresholds"]["mock-model"] < before  # wider idle share -> lower threshold
    assert after["label_thresholds"] == data["label_thresholds"]  # labels unchanged


def _hashes(run_id):
    import hashlib

    from tpa_index.paths import output_dir

    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (output_dir() / "runs" / run_id).iterdir()}


def test_moments_rebuild_follows_thresholds_and_keeps_predict_run(tmp_path, monkeypatch):
    from collections import Counter

    from tpa_index import config
    from tpa_index.schema import read_official, read_table

    cli = _official_mock(tmp_path, monkeypatch)
    before_files = _hashes("mock-seed0")
    before = read_official("moments").to_pylist()
    top = Counter(m["primary_label"] for m in before).most_common(1)[0][0]
    config.THRESHOLDS_LOCAL.write_text(f"label_thresholds:\n  mock-model: {{{top}: 0.99}}\n")
    assert cli.main(["moments"]) == 0
    (rebuilt,) = [r["run_id"] for r in read_table("runs").to_pylist() if r["stage"] == "moments"]
    after = read_table("moments", run_ids=[rebuilt]).to_pylist()
    assert after != before and top not in {m["primary_label"] for m in after}
    assert _hashes("mock-seed0") == before_files  # predict run untouched
    (tmp_path / "official.yaml").write_text(f"predict: [mock-seed0]\nmoments: [{rebuilt}]\n")
    assert read_official("moments").to_pylist() == after
    assert read_official("segments").num_rows == read_table("segments", run_ids=[rebuilt]).num_rows


def test_moments_rebuild_recomputes_idle(tmp_path, monkeypatch):
    from tpa_index import config
    from tpa_index.schema import read_table

    cli = _official_mock(tmp_path, monkeypatch)
    idle_before = sum(bool(s["is_idle"]) for s in read_table("segments", run_ids=["mock-seed0"]).to_pylist())
    config.THRESHOLDS_LOCAL.write_text("idle_thresholds: {mock-model: 0.0}\n")  # every low-change segment is idle
    assert cli.main(["moments"]) == 0
    (rebuilt,) = [r["run_id"] for r in read_table("runs").to_pylist() if r["stage"] == "moments"]
    segs = read_table("segments", run_ids=[rebuilt]).to_pylist()
    low = [s for s in segs if s["visual_change"] < config.load_pipeline()["idle"]["change_threshold"]]
    assert all(s["is_idle"] for s in low) and sum(s["is_idle"] for s in segs) > idle_before
    idle_spans = [(s["video_id"], s["start_s"], s["end_s"]) for s in segs if s["is_idle"]]
    for m in read_table("moments", run_ids=[rebuilt]).to_pylist():  # no moment contains an idle segment
        assert not any(v == m["video_id"] and m["start_s"] <= a and b <= m["end_s"] for v, a, b in idle_spans)
