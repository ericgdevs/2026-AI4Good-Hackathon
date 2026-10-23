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
    import pyarrow.compute as pc

    from tpa_index.schema import read_table

    yaml_path = _clip_yaml(tmp_path, clips)
    for _ in range(2):
        result = run_cli("segment", "--videos", str(yaml_path), DATA_DIR=str(clips))
        assert result.returncode == 0, result.stderr
        assert str(clips) not in result.stdout + result.stderr
    segments, runs = read_table("segments"), read_table("runs")
    run_ids = sorted(set(runs["run_id"].to_pylist()))
    assert len(run_ids) == 2 and set(runs["stage"].to_pylist()) == {"segment"}
    rows = segments.to_pylist()
    assert rows[: len(rows) // 2] == rows[len(rows) // 2:]  # same rows from both runs
    assert runs["n_segments"].to_pylist() == [len(rows) // 2] * 2
    assert read_table("videos").filter(pc.equal(read_table("videos")["video_id"], "t02"))["orientation"][0].as_py() \
        == "portrait"  # probed, not taken from the YAML


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
