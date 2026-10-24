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
