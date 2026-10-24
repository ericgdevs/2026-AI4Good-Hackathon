"""004 signal stages: pure functions and orchestration with fakes (no model loaded, synthetic clips only)."""
import os

import pytest

from tpa_index import schema, signals
from tpa_index.config import ConfigError, effective_settings
from tpa_index.paths import GuardError, repo_root
from tpa_index.schema import write_table


def official_clip_runs(videos, pipeline, tmp_path, monkeypatch, stage="segment"):
    """One official segment run per synthetic clip video (windows from the clip's real duration)."""
    from tpa_index.cli import _video_row
    from tpa_index.paths import data_dir
    from tpa_index.segment import probe, windows

    ids = []
    for v in videos:
        run_id = f"{stage}-20261024T000000-{v['video_id']}"
        info = probe(data_dir() / v["filename"])
        write_table("videos", [_video_row(v, info)], run_id)
        s = effective_settings(pipeline, v)
        write_table("segments", [
            {"segment_id": f"{v['video_id']}_{round(a * 1000)}", "video_id": v["video_id"], "start_s": a, "end_s": b,
             "person_count": None, "faces_detected": None, "has_speech": None, "transcript_chars": None,
             "is_idle": None, "idle_score": None, "visual_change": None}
            for a, b in windows(info["duration_s"], s["window_s"], s["overlap_s"])], run_id)
        write_table("runs", [{"run_id": run_id, "stage": stage, "model": None, "config_hash": "0" * 12,
                              "hardware": "test", "n_segments": 1, "runtime_s": 1.0, "peak_mem_gb": None,
                              "started_at": __import__("datetime").datetime(2026, 10, 24,
                                                                            tzinfo=__import__("datetime").timezone.utc),
                              "git_commit": None}], run_id)
        ids.append(run_id)
    path = tmp_path / "official.yaml"
    path.write_text(f"{stage}:\n" + "".join(f"  - {i}\n" for i in ids))
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)
    return ids


# ---- foundation (T012) --------------------------------------------------------------------------------------------
def test_scratch_dir_override_and_repo_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("TPA_SCRATCH_DIR", str(tmp_path / "s"))
    assert signals.scratch_dir() == (tmp_path / "s").resolve() and (tmp_path / "s").is_dir()
    monkeypatch.setenv("TPA_SCRATCH_DIR", str(repo_root() / "tmp-scratch"))
    with pytest.raises(GuardError, match="outside the repo"):
        signals.scratch_dir()
    assert not (repo_root() / "tmp-scratch").exists()


def test_clean_stale_scratch_keeps_live_pids(tmp_path):
    import psutil

    tmp_path = tmp_path / "scratch"
    tmp_path.mkdir()

    dead = 999_999
    while psutil.pid_exists(dead):
        dead += 1
    (tmp_path / f"tpa-audio-{dead}-v01.wav").write_bytes(b"x")
    (tmp_path / f"tpa-audio-{os.getpid()}-v02.wav").write_bytes(b"x")
    (tmp_path / "other.wav").write_bytes(b"x")
    assert signals.clean_stale_scratch(tmp_path) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["other.wav", f"tpa-audio-{os.getpid()}-v02.wav"]


def test_official_segments_fallback_and_errors(clip_videos, pipeline, tmp_path, monkeypatch):
    with pytest.raises(ConfigError, match="no official segments for t01"):
        signals.official_segments("t01")
    official_clip_runs([clip_videos["t01"]], pipeline, tmp_path, monkeypatch)
    segs = signals.official_segments("t01")
    assert [s["segment_id"] for s in segs][:2] == ["t01_0", "t01_6000"]
    assert signals.official_video("t01")["duration_s"] > 19


def test_official_segments_prefer_predict(clip_videos, pipeline, tmp_path, monkeypatch):
    official_clip_runs([clip_videos["t01"]], pipeline, tmp_path, monkeypatch, stage="predict")
    assert signals.official_segments("t01")[0]["segment_id"] == "t01_0"


def test_check_windows(clip_videos, pipeline, tmp_path, monkeypatch):
    v = clip_videos["t02"]  # clip kind: 4 s windows
    official_clip_runs([v], pipeline, tmp_path, monkeypatch)
    segs, row = signals.official_segments("t02"), signals.official_video("t02")
    signals.check_windows(row, effective_settings(pipeline, v), segs)
    shifted = [{**segs[0], "segment_id": "t02_1"}, *segs[1:]]
    with pytest.raises(ValueError, match="windows do not match"):
        signals.check_windows(row, effective_settings(pipeline, v), shifted)
    with pytest.raises(ValueError):
        signals.check_windows(row, pipeline, segs)  # default 8 s windows are not this clip's windows


def test_base_rows_empty_optional_columns(clip_videos, pipeline, tmp_path, monkeypatch):
    official_clip_runs([clip_videos["t01"]], pipeline, tmp_path, monkeypatch)
    rows = signals.base_rows(signals.official_segments("t01"))
    assert all(r[c] is None for r in rows for c in signals.SIGNAL_COLUMNS)
    schema.validate("segments", rows)


# ---- transcript and audio helpers (T017-T019) ----------------------------------------------------------------------
SEGS = [{"segment_id": f"v_{a * 1000}", "start_s": float(a), "end_s": float(a + 8)} for a in (0, 6, 12)]
SEGS[-1]["end_s"] = 19.5


def _w(start, end, word):
    return {"start": start, "end": end, "word": word}


def test_assign_words_midpoint_rule():
    out = signals.assign_words([_w(1.0, 1.4, " ab"), _w(6.8, 7.2, " cde ")], SEGS)
    assert out["v_0"] == (True, 5)  # 1.2 and 7.0 both in [0, 8)
    assert out["v_6000"] == (True, 3)  # 7.0 also in [6, 14): overlap counts twice
    assert out["v_12000"] == (False, 0)


def test_assign_words_boundaries():
    out = signals.assign_words([_w(5.9, 6.1, "x"), _w(19.4, 19.6, "yz")], SEGS)
    assert out["v_0"] == (True, 1) and out["v_6000"] == (True, 1)  # midpoint 6.0: start inclusive, end exclusive
    assert out["v_12000"] == (True, 2)  # midpoint exactly at the video's end counts in the last segment


def test_assign_words_none():
    assert set(signals.assign_words([], SEGS).values()) == {(False, 0)}


def test_transcript_rows_fill_only_speech_columns():
    rows = signals.transcript_rows([{**s, "video_id": "v"} for s in SEGS],
                                   [{"words": [_w(1.0, 1.4, " ab")]}])
    assert [(r["has_speech"], r["transcript_chars"]) for r in rows] == [(True, 2), (False, 0), (False, 0)]
    assert all(r["person_count"] is None and r["faces_detected"] is None for r in rows)


def test_extract_audio_and_read_clip(clip_videos, tmp_path):
    import soundfile as sf

    wav = signals.extract_audio(clip_videos["t01"], 20.0, tmp_path / "a.wav")
    info = sf.info(str(wav))
    assert (info.samplerate, info.channels) == (16000, 1) and 19.5 < info.duration < 20.5
    clip = signals.read_clip(wav, 0.0, 8.0, 163840)
    assert clip.shape == (163840,) and abs(clip[:128000]).max() > 0.1 and not clip[128000:].any()  # padded
    assert not signals.read_clip(wav, 30.0, 38.0, 1000).any()  # past the end: silence
    assert signals.read_clip(wav, 0.0, 8.0, 1000).shape == (1000,)  # cut to the model input


def test_extract_audio_no_track_is_silence(clip_videos, tmp_path):
    import soundfile as sf

    wav = signals.extract_audio(clip_videos["t03"], 20.0, tmp_path / "b.wav")
    data, rate = sf.read(str(wav))
    assert rate == 16000 and len(data) == 320000 and not data.any()


def test_map_scores_max_rule():
    import numpy as np

    probs = np.array([[0.1, 0.9, 0.3], [0.5, 0.2, 0.4]], np.float32)
    out = signals.map_scores(probs, {"a": [0, 1], "b": [2]})
    assert np.allclose(out, [[0.9, 0.3], [0.5, 0.4]])


# ---- orchestration: transcript + audio with fakes (T022) ----------------------------------------------------------
AUDIO_LABELS = ["singing_voice", "music", "speech", "applause", "silence"]


@pytest.fixture
def job_a(clip_videos, pipeline, labels, tmp_path, monkeypatch):
    """Official segment runs for t01 (audio) and t03 (no audio track); scratch under tmp."""
    from tests.fakes import FakeAudioTagger, FakeTranscriber

    monkeypatch.setenv("TPA_SCRATCH_DIR", str(tmp_path / "scratch"))
    vids = [clip_videos["t01"], clip_videos["t03"]]
    official_clip_runs(vids, pipeline, tmp_path, monkeypatch)
    models = {"transcript": FakeTranscriber(), "audio": FakeAudioTagger(AUDIO_LABELS)}

    def run(stages=("transcript", "audio"), **kw):
        return signals.run_signals(vids, stages, models, pipeline, labels, **kw)
    return run, models, vids


def _runs(stage=None):
    rows = schema.read_table("runs").to_pylist()
    return [r for r in rows if stage is None or r["stage"] == stage]


def test_one_run_per_video_and_stage(job_a, labels, pipeline):
    from tpa_index.config import stage_config_hash

    run, models, vids = job_a
    results = run()
    assert set(results) == {(v["video_id"], s) for v in vids for s in ("transcript", "audio")}
    assert all(r.startswith("done:") for r in results.values())
    for v in vids:
        for stage in ("transcript", "audio"):
            run_id = results[(v["video_id"], stage)].split(":", 1)[1]
            (row,) = schema.read_table("runs", run_ids=[run_id]).to_pylist()
            assert (row["stage"], row["model"]) == (stage, models[stage].name)
            assert row["config_hash"] == stage_config_hash(stage, labels, pipeline, v)
            assert row["n_segments"] == len(signals.official_segments(v["video_id"]))


def test_transcript_and_audio_rows(job_a):
    run, _, _ = job_a
    results = run()
    t01 = results[("t01", "transcript")].split(":")[1]
    segs = schema.read_table("segments", run_ids=[t01]).to_pylist()
    assert [s["has_speech"] for s in segs][:2] == [True, True] and segs[-1]["has_speech"] is False
    assert all(s["person_count"] is None for s in segs)
    audio = results[("t01", "audio")].split(":")[1]
    preds = schema.read_table("predictions", run_ids=[audio]).to_pylist()
    assert {p["label_group"] for p in preds} == {"audio"} and {p["label"] for p in preds} == set(AUDIO_LABELS)
    assert len(preds) == len(segs) * len(AUDIO_LABELS)


def test_no_audio_track_video(job_a):
    import soundfile  # noqa: F401  (WAV reads)

    run, models, _ = job_a
    models["transcript"].words = []  # silence: VAD finds nothing
    results = run()
    t03 = results[("t03", "transcript")].split(":")[1]
    assert {(s["has_speech"], s["transcript_chars"]) for s in schema.read_table("segments", run_ids=[t03]).to_pylist()} \
        == {(False, 0)}
    assert results[("t03", "audio")].startswith("done:")


def test_failure_isolated_and_rerun_only_failed(job_a):
    """SC-006: one failing pair; the rest complete; a rerun creates only the missing run."""
    run, models, _ = job_a
    models["transcript"].fail_for = {"t01"}
    first = run()
    assert [k for k, v in first.items() if v.startswith("failed:")] == [("t01", "transcript")]
    assert first[("t01", "transcript")] == "failed:TranscriptError(RuntimeError)"
    assert len(_runs()) - 2 == 3  # 2 official segment runs + 3 stage runs
    models["transcript"].fail_for = set()
    second = run()
    assert second[("t01", "transcript")].startswith("done:")
    assert all(v.startswith("skipped:") for k, v in second.items() if k != ("t01", "transcript"))
    assert len(_runs("transcript")) == 2 and len(_runs("audio")) == 2


def test_force_reruns(job_a):
    run, _, _ = job_a
    run()
    again = run(force=True)
    assert all(v.startswith("done:") for v in again.values())
    assert len(_runs("audio")) == 4


def test_window_mismatch_writes_nothing(job_a, tmp_path):
    from tpa_index import schema as sch

    run, _, _ = job_a
    folder = next((tmp_path / "out/runs").glob("segment-*-t01"))
    tbl = sch.read_table("segments", run_ids=[folder.name]).to_pylist()
    tbl[0]["segment_id"] = "t01_1"
    (folder / "segments.parquet").unlink()
    sch.write_table("segments", tbl, folder.name)
    results = run()
    assert results[("t01", "transcript")] == results[("t01", "audio")] == "failed:ValueError"
    assert results[("t03", "audio")].startswith("done:")
    assert not [r for r in _runs() if r["stage"] in ("transcript", "audio") and "t01" in
                schema.read_table("videos", run_ids=[r["run_id"]])["video_id"].to_pylist()]


def test_audio_extracted_once_per_video(job_a, monkeypatch):
    run, _, _ = job_a
    calls = []
    real = signals.extract_audio
    monkeypatch.setattr(signals, "extract_audio", lambda *a, **k: calls.append(a[0]["video_id"]) or real(*a, **k))
    run()
    assert calls == ["t01", "t03"]


def test_audio_clips_padded_to_model_input(job_a):
    run, models, _ = job_a
    run(stages=("audio",))
    assert set(models["audio"].shapes) == {(models["audio"].input_samples,)}


# ---- model wrappers without weights (T024) --------------------------------------------------------------------------
def test_whisper_cuda_failure_names_library_and_does_not_fall_back(monkeypatch):
    import faster_whisper

    from tpa_index.signal_models import WhisperTranscriber

    calls = []

    def fake_model(size, device, compute_type, local_files_only):
        calls.append(device)
        raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")

    monkeypatch.setattr(faster_whisper, "WhisperModel", fake_model)
    with pytest.raises(ConfigError) as err:
        WhisperTranscriber("large-v3", "cuda", "float16")
    assert "libcublas.so.12" in str(err.value) and "hpc_device" in str(err.value)
    assert calls == ["cuda"]  # never retried on CPU


def test_whisper_cuda_failure_at_first_inference_is_caught_at_startup(monkeypatch):
    import faster_whisper

    from tpa_index.signal_models import WhisperTranscriber

    class Lazy:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            raise RuntimeError("Library libcudnn_ops.so.9.1.0 is not found")

    monkeypatch.setattr(faster_whisper, "WhisperModel", Lazy)
    with pytest.raises(ConfigError, match=r"missing libcudnn_ops\.so\.9\.1\.0"):
        WhisperTranscriber("large-v3", "cuda", "float16")


def test_whisper_missing_model_is_config_error(monkeypatch):
    import faster_whisper
    from huggingface_hub.errors import LocalEntryNotFoundError

    from tpa_index.signal_models import WhisperTranscriber

    def missing(*a, **k):
        raise LocalEntryNotFoundError("no snapshot")

    monkeypatch.setattr(faster_whisper, "WhisperModel", missing)
    with pytest.raises(ConfigError, match="model not found locally: faster-whisper-base"):
        WhisperTranscriber("base", "cpu", "int8")


# ---- people and expression helpers (T034) ---------------------------------------------------------------------------
def test_median_count_rounds_halves_down():
    assert signals.median_count([1, 2]) == 1
    assert signals.median_count([1, 2, 3, 4]) == 2
    assert signals.median_count([3]) == 3
    assert signals.median_count([0, 5, 5, 0]) == 2


def test_aggregate_expression():
    import numpy as np

    labels = ["positive", "neutral", "negative", "upset"]
    assert signals.aggregate_expression([np.zeros((0, 4))] * 3, labels) == (0, None)
    frames = [np.zeros((0, 4)), np.array([[0.6, 0.2, 0.1, 0.0], [0.2, 0.2, 0.1, 0.1]]),
              np.array([[0.4, 0.5, 0.0, 0.0]])]
    faces, scores = signals.aggregate_expression(frames, labels)
    assert faces == 2  # most faces in one frame, not the total (no tracking, no double counting)
    assert scores == pytest.approx({"positive": 0.4, "neutral": 0.3, "negative": 0.2 / 3, "upset": 0.1 / 3})
    assert sum(scores.values()) < 1  # dropped classes are not renormalized


# ---- weight loading never downloads (T037) --------------------------------------------------------------------------
def test_missing_weights_raise_our_error_and_never_download(tmp_path, monkeypatch, pipeline):
    import urllib.request

    import emotiefflib.utils
    import ultralytics

    from tpa_index.signal_models import FaceExpression, YoloPersonCounter

    def forbidden(*a, **k):
        raise AssertionError("a download was attempted")

    monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
    monkeypatch.setattr(emotiefflib.utils, "download_model", forbidden)
    monkeypatch.setattr(ultralytics, "YOLO", forbidden)
    monkeypatch.setenv("TPA_WEIGHTS_DIR", str(tmp_path / "empty"))
    monkeypatch.delenv("YOLO_OFFLINE", raising=False)
    m = pipeline["models"]
    with pytest.raises(ConfigError, match="weights not found: yolo11n.pt"):
        YoloPersonCounter(m["people"]["mac"], "cpu", 0.4)
    with pytest.raises(ConfigError, match="weights not found: enet_b0_8_va_mtl.onnx"):
        FaceExpression(m["expression"]["face_detector"], m["expression"]["model"],
                       pipeline["signals"]["expression"]["mapping"], 0.8, 48)


def test_unpinned_weights_refused(tmp_path, monkeypatch, pipeline):
    from tpa_index.signal_models import YoloPersonCounter

    monkeypatch.setenv("TPA_WEIGHTS_DIR", str(tmp_path))
    (tmp_path / "yolo11n.pt").write_bytes(b"not really weights")
    with pytest.raises(ConfigError, match="weights not pinned: set sha256 for yolo11n.pt"):
        YoloPersonCounter(pipeline["models"]["people"]["mac"], "cpu", 0.4)


def test_cli_sets_yolo_offline(monkeypatch, clip_videos, pipeline, tmp_path):
    from tests.test_cli import _job_a_models, signals_cli

    monkeypatch.delenv("YOLO_OFFLINE", raising=False)
    cli, sel = signals_cli(monkeypatch, _job_a_models(), clip_videos, pipeline, tmp_path)
    cli.main(["signals", "--stages", "audio", *sel])
    assert os.environ["YOLO_OFFLINE"] == "1"


def test_usable_faces_filters_score_size_and_clips():
    import numpy as np

    from tpa_index.signal_models import usable_faces

    def row(x, y, w, h, score):
        return [x, y, w, h, *range(10), score]  # 10 landmark values, never returned

    dets = np.array([row(10, 10, 60, 60, 0.95), row(100, 10, 60, 60, 0.5), row(200, 10, 30, 30, 0.99),
                     row(-20, 300, 80, 80, 0.9)], np.float32)
    boxes = usable_faces(dets, (400, 400, 3), 0.8, 48)
    assert boxes == [(10, 10, 70, 70), (0, 300, 60, 380)]  # low score, small face dropped; edge face clipped
    assert usable_faces(None, (400, 400, 3), 0.8, 48) == []
    assert usable_faces(np.array([row(-40, 0, 80, 80, 0.9)], np.float32), (400, 400, 3), 0.8, 48) == []


# ---- orchestration: people + expression with fakes (T039) ---------------------------------------------------------
EXPR_LABELS = ["positive", "neutral", "negative", "upset"]


@pytest.fixture
def job_b(clip_videos, pipeline, labels, tmp_path, monkeypatch):
    from tests.fakes import FakeExpressionScorer, FakePersonCounter

    monkeypatch.setenv("TPA_SCRATCH_DIR", str(tmp_path / "scratch"))
    vids = [clip_videos["t01"], clip_videos["t02"]]  # live 8 s windows and clip 4 s windows
    official_clip_runs(vids, pipeline, tmp_path, monkeypatch)
    models = {"people": FakePersonCounter(count=3), "expression": FakeExpressionScorer(EXPR_LABELS, faces_per_frame=2)}

    def run(stages=("people", "expression"), **kw):
        return signals.run_signals(vids, stages, models, pipeline, labels, **kw)
    return run, models, vids


def test_job_b_one_run_per_pair_with_rows(job_b, labels, pipeline):
    from tpa_index.config import stage_config_hash

    run, models, vids = job_b
    results = run()
    assert all(v.startswith("done:") for v in results.values()) and len(results) == 4
    for v in vids:
        n = len(signals.official_segments(v["video_id"]))
        people = results[(v["video_id"], "people")].split(":")[1]
        expr = results[(v["video_id"], "expression")].split(":")[1]
        (prow,) = schema.read_table("runs", run_ids=[people]).to_pylist()
        assert prow["config_hash"] == stage_config_hash("people", labels, pipeline, v)
        assert {s["person_count"] for s in schema.read_table("segments", run_ids=[people]).to_pylist()} == {3}
        segs = schema.read_table("segments", run_ids=[expr]).to_pylist()
        assert {s["faces_detected"] for s in segs} == {2} and all(s["person_count"] is None for s in segs)
        preds = schema.read_table("predictions", run_ids=[expr]).to_pylist()
        assert len(preds) == n * 4 and {p["label_group"] for p in preds} == {"expression"}
        assert "confident" not in {p["label"] for p in preds}


def test_job_b_frames_decoded_once(job_b, monkeypatch):
    from tpa_index import segment

    run, _, vids = job_b
    calls = []
    real = segment._sample_frames
    monkeypatch.setattr(segment, "_sample_frames", lambda *a, **k: calls.append(1) or real(*a, **k))
    run()
    assert len(calls) == sum(len(signals.official_segments(v["video_id"])) for v in vids)


def test_job_b_expression_failure_keeps_people(job_b):
    run, models, _ = job_b
    models["expression"].fail_on_call = 3  # mid-pass, first video
    results = run()
    assert results[("t01", "expression")] == "failed:RuntimeError"
    assert results[("t01", "people")].startswith("done:")
    assert results[("t02", "expression")].startswith("done:")
    expr_runs = [r for r in schema.read_table("runs").to_pylist() if r["stage"] == "expression"]
    assert len(expr_runs) == 1


def test_job_b_resume_and_force(job_b):
    run, _, _ = job_b
    run()
    again = run()
    assert all(v.startswith("skipped:") for v in again.values())
    forced = run(force=True)
    assert all(v.startswith("done:") for v in forced.values())


def test_job_b_shared_runtime(job_b):
    run, _, _ = job_b
    results = run()
    rows = {r["run_id"]: r for r in schema.read_table("runs").to_pylist()}
    t01 = [rows[results[("t01", s)].split(":")[1]] for s in ("people", "expression")]
    assert t01[0]["runtime_s"] == t01[1]["runtime_s"] and t01[0]["peak_mem_gb"] == t01[1]["peak_mem_gb"]
