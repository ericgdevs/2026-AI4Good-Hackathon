import os
import subprocess
import sys

import pytest

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


def _crafted_run(label="acting_theater", score=0.001):
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
    assert "labels at the null floor: none" in out and "labels at min_threshold: none" in out
    assert "null floor: 0." in out and "idle score_threshold:" in out
    assert "null_food_plate" not in out  # null labels never get a threshold row
    assert not config.THRESHOLDS_LOCAL.exists()


def test_calibrate_writes_labels_idle_and_reports_floor(tmp_path, monkeypatch, capsys):
    import yaml

    from tpa_index import config

    cli = _official_mock(tmp_path, monkeypatch)
    _crafted_run("acting_theater")
    (tmp_path / "official.yaml").write_text("predict: [crafted]\n")
    assert cli.main(["calibrate"]) == 0
    out = capsys.readouterr().out
    assert "labels at the null floor: acting_theater" in out
    floor = float(next(line for line in out.splitlines() if line.startswith("null floor:")).split()[2])
    data = yaml.safe_load(config.THRESHOLDS_LOCAL.read_text())
    assert data["label_thresholds"]["mock-model"]["acting_theater"] == pytest.approx(floor, abs=1e-4)
    assert not any(k.startswith("null_") for k in data["label_thresholds"]["mock-model"])
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


def test_calibrate_refuses_mixed_models_unless_named(tmp_path, monkeypatch, capsys):
    cli = _official_mock(tmp_path, monkeypatch)
    _crafted_run("acting_theater")
    from tpa_index.schema import read_table, write_table

    rows = read_table("predictions", run_ids=["crafted"]).to_pylist()
    other = [dict(r, model="other-model") for r in rows if r["model"] == "mock-model"][:50]
    for r in other:
        r["segment_id"] = r["segment_id"].replace("mock01", "mock99")
    write_table("videos", [dict(read_table("videos", run_ids=["crafted"]).to_pylist()[0], video_id="mock99")], "mixed")
    write_table("predictions", other, "mixed")
    write_table("runs", [dict(read_table("runs", run_ids=["crafted"]).to_pylist()[0], run_id="mixed")], "mixed")
    (tmp_path / "official.yaml").write_text("predict: [crafted, mixed]\n")
    assert cli.main(["calibrate", "--dry-run"]) == 3
    assert "official runs use several models (mock-model, other-model); pass --model" in capsys.readouterr().err
    assert cli.main(["calibrate", "--dry-run", "--model", "mock-model"]) == 0
    out = capsys.readouterr().out
    assert "model: mock-model" in out and "dev videos used: mock01, mock02, mock03" in out
    assert cli.main(["calibrate", "--dry-run", "--model", "nope"]) == 3
    assert "model nope not in official predict runs" in capsys.readouterr().err


def test_calibrate_moments_stage_has_no_model_rows(tmp_path, monkeypatch, capsys):
    cli = _official_mock(tmp_path, monkeypatch)
    assert cli.main(["moments"]) == 0
    from tpa_index.schema import list_runs

    rebuilt = [r["run_id"] for r in list_runs() if r["stage"] == "moments"]
    (tmp_path / "official.yaml").write_text(f"predict: [mock-seed0]\nmoments: [{rebuilt[0]}]\n")
    capsys.readouterr()
    assert cli.main(["calibrate", "--stage", "moments", "--dry-run"]) == 3
    assert "no model-scored predictions in official moments runs" in capsys.readouterr().err


def test_moments_unknown_video_id_exits_3(tmp_path, monkeypatch, capsys):
    cli = _official_mock(tmp_path, monkeypatch)
    assert cli.main(["moments", "--video-id", "typo1"]) == 3
    assert "no official predict run covers typo1" in capsys.readouterr().err


def test_mock_clips_cover_every_segment(tmp_path):
    import av

    from tpa_index.paths import data_dir
    from tpa_index.schema import read_table

    result = run_cli("mock", "--n-videos", "1", "--clips")
    assert result.returncode == 0, result.stderr
    (video,) = read_table("videos").to_pylist()
    last_end = max(read_table("segments")["end_s"].to_pylist())
    path = data_dir() / video["filename"]
    assert path.exists()
    with av.open(str(path)) as c:
        assert c.duration / 1e6 >= last_end - 0.5
    assert str(data_dir()) not in result.stdout



def test_calibrate_lists_labels_and_idle_at_min_threshold(tmp_path, monkeypatch, capsys):
    """Real-run case: null floor ~0, a barely-seen label and a ~0 idle score must stop at min_threshold."""
    import yaml

    from tpa_index import config
    from tpa_index.schema import read_table, write_table

    cli = _official_mock(tmp_path, monkeypatch)
    null_group = config.load_pipeline()["calibrate"]["null_group"]
    nulls = {lab["name"] for lab in config.load_labels()["labels"] if lab["group"] == null_group}
    for table in ("videos", "segments", "predictions", "moments", "runs"):
        rows = read_table(table, run_ids=["mock-seed0"]).to_pylist()
        for r in rows:
            if "run_id" in r:
                r["run_id"] = "lowfloor"
            if table == "predictions" and (r["label"] in nulls or r["label"] == "acting_theater"):
                r["score"] = 0.00001  # null floor ~0; acting_theater barely seen
            if table == "segments" and r["idle_score"] is not None:
                r["idle_score"] = 0.00001  # idle percentile ~0
        write_table(table, rows if table != "runs" else rows[:1], "lowfloor")
    (tmp_path / "official.yaml").write_text("predict: [lowfloor]\n")
    assert cli.main(["calibrate"]) == 0
    out = capsys.readouterr().out
    assert "labels at min_threshold: acting_theater" in out
    assert "idle score_threshold: 0.0010 (at min_threshold)" in out
    data = yaml.safe_load(config.THRESHOLDS_LOCAL.read_text())
    assert data["label_thresholds"]["mock-model"]["acting_theater"] == 0.001
    assert data["idle_thresholds"]["mock-model"] == 0.001


def test_missing_official_run_exits_with_message_not_traceback(tmp_path, monkeypatch, capsys):
    from tpa_index import clips

    cli = _official_mock(tmp_path, monkeypatch)
    (tmp_path / "official.yaml").write_text("predict: [mock-seed0, not-on-disk]\n")
    assert cli.main(["calibrate", "--dry-run"]) == 3
    assert "run not-on-disk not found or incomplete" in capsys.readouterr().err
    # the app too: startup reads the official runs before it serves anything
    monkeypatch.setattr(clips, "register_cleanup", lambda cutter: None)  # no signal handlers inside pytest
    monkeypatch.setattr(clips, "session_dir", lambda: tmp_path / "session")
    (tmp_path / "session").mkdir()
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert cli.main(["app", "--port", str(port)]) == 3
    err = capsys.readouterr().err
    assert "run not-on-disk not found or incomplete" in err and "Traceback" not in err


# ---- signals (004): fakes only ---------------------------------------------------------------------------------
def signals_cli(monkeypatch, models, clip_videos, pipeline, tmp_path, ids=("t01", "t03")):
    """In-process cli with official segment runs for the clips and `models` in place of real weights."""
    from tests.test_signals import official_clip_runs
    from tpa_index import cli

    monkeypatch.setattr(cli, "install_redaction", lambda: None)
    monkeypatch.setattr(cli, "_make_signal_models", lambda stages, pipeline, dev: {s: models[s] for s in stages})
    monkeypatch.setenv("TPA_SCRATCH_DIR", str(tmp_path / "scratch"))
    official_clip_runs([clip_videos[i] for i in ids], pipeline, tmp_path, monkeypatch)
    return cli, ["--videos", str(_predict_yaml(tmp_path)), *[x for i in ids for x in ("--video-id", i)]]


def _job_a_models(**kw):
    from tests.fakes import FakeAudioTagger, FakeTranscriber

    return {"transcript": FakeTranscriber(**kw), "audio": FakeAudioTagger(
        ["singing_voice", "music", "speech", "applause", "silence"])}


def test_signals_unknown_stage_exits_3(monkeypatch, clip_videos, pipeline, tmp_path, capsys):
    cli, sel = signals_cli(monkeypatch, _job_a_models(), clip_videos, pipeline, tmp_path)
    assert cli.main(["signals", "--stages", "bogus", *sel]) == 3
    assert "unknown stage bogus" in capsys.readouterr().err


def test_signals_model_load_failure_exits_3_before_any_video(monkeypatch, clip_videos, pipeline, tmp_path, capsys):
    from tpa_index import config, schema

    cli, sel = signals_cli(monkeypatch, _job_a_models(), clip_videos, pipeline, tmp_path)

    def broken(stages, pipeline, dev):
        raise config.ConfigError("model not found locally: faster-whisper-base")

    monkeypatch.setattr(cli, "_make_signal_models", broken)
    assert cli.main(["signals", "--stages", "transcript,audio", *sel]) == 3
    assert "model not found locally" in capsys.readouterr().err
    assert {r["stage"] for r in schema.read_table("runs").to_pylist()} == {"segment"}


def test_signals_done_skipped_and_failed_lines(monkeypatch, clip_videos, pipeline, tmp_path, capsys):
    models = _job_a_models(fail_for={"t03"})
    cli, sel = signals_cli(monkeypatch, models, clip_videos, pipeline, tmp_path)
    assert cli.main(["signals", "--stages", "transcript,audio", *sel]) == 1
    out, err = capsys.readouterr()
    assert "transcript t01: done transcript-" in out and "audio t03: done audio-" in out
    import re

    assert re.search(r"audio t03: done audio-\S+ \d+ segs \d+s", out)
    assert "transcript t03: failed TranscriptError(RuntimeError)" in out
    assert err.strip().splitlines()[-1] == "failed: transcript t03"
    models["transcript"].fail_for = set()
    assert cli.main(["signals", "--stages", "transcript,audio", *sel]) == 0
    out = capsys.readouterr().out
    assert "transcript t03: done" in out and "audio t01: skipped (done: audio-" in out


# ---- evaluate (005): mock world only ---------------------------------------------------------------------------
EVALUATE_FILES = {"metrics.csv", "metrics.json", "confusion.csv", "signal_indicators.csv", "ablation.csv",
                  "topup.csv", "failure_gallery.csv", "learned_predictions.csv", "summary.md", "manifest.json",
                  "ablation.png", "per_label_f1.png", "confusion.png", "coverage.png"}


def _evaluate_cli(tmp_path, monkeypatch, **kw):
    from tests.insights_fixtures import build_world
    from tpa_index import cli

    monkeypatch.setattr(cli, "install_redaction", lambda: None)
    build_world(tmp_path, monkeypatch, **kw)
    return cli


def test_evaluate_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    cli = _evaluate_cli(tmp_path, monkeypatch)
    assert cli.main(["evaluate", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "primary accuracy" in out and "95% CI" in out and "kappa" in out and "dry run: nothing written" in out
    assert not (tmp_path / "out/results").exists()


def test_evaluate_writes_one_complete_folder(tmp_path, monkeypatch, capsys):
    import json

    cli = _evaluate_cli(tmp_path, monkeypatch)
    assert cli.main(["evaluate"]) == 0
    (folder,) = list((tmp_path / "out/results").iterdir())
    assert folder.name.startswith("evaluate-")
    assert {p.name for p in folder.iterdir()} == EVALUATE_FILES
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["schema_version"] == 1 and set(manifest["files"]) == EVALUATE_FILES - {"manifest.json"}
    assert f"written: <OUTPUT_DIR>/results/{folder.name}" in capsys.readouterr().out


def test_evaluate_missing_audio_exits_3(tmp_path, monkeypatch, capsys):
    cli = _evaluate_cli(tmp_path, monkeypatch, overrides={"skip_stages": [("audio", "e02")]})
    assert cli.main(["evaluate"]) == 3
    assert "official audio runs missing for: e02" in capsys.readouterr().err
    assert not (tmp_path / "out/results").exists()


# ---- insights (005): mock world only ---------------------------------------------------------------------------
INSIGHTS_FILES = {"indicators.csv", "indicators.json", "headline.csv", "time_breakdown.csv", "summary.md",
                  "manifest.json", "headline.png", "time_breakdown.png"}


def test_insights_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    cli = _evaluate_cli(tmp_path, monkeypatch)
    assert cli.main(["insights", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "performance_share [early]" in out and ("model estimate (learned)" in out or "staff-verified" in out)
    assert "dry run: nothing written" in out and not (tmp_path / "out/results").exists()


def test_insights_writes_one_complete_folder(tmp_path, monkeypatch):
    cli = _evaluate_cli(tmp_path, monkeypatch)
    assert cli.main(["insights"]) == 0
    (folder,) = list((tmp_path / "out/results").iterdir())
    assert folder.name.startswith("insights-")
    assert INSIGHTS_FILES == {p.name for p in folder.iterdir()}


def test_insights_missing_expression_exits_3(tmp_path, monkeypatch, capsys):
    cli = _evaluate_cli(tmp_path, monkeypatch, overrides={"skip_stages": [("expression", "e01")]})
    assert cli.main(["insights"]) == 3
    assert "official expression runs missing for: e01" in capsys.readouterr().err


def test_evaluate_and_insights_fast_on_mock_scale(tmp_path, monkeypatch):
    """SC-007: 12 videos x 300 segments with 205 eval labels; each command finishes in under 30 s."""
    import time

    cli = _evaluate_cli(tmp_path, monkeypatch, n_extra_eval=8, n_segments=300, n_labels=205)
    for command in ("evaluate", "insights"):
        t0 = time.perf_counter()
        assert cli.main([command]) == 0
        assert time.perf_counter() - t0 < 30, command
