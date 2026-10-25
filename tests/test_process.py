"""Process a video tab (007): background run, progress, failures, the local-runs registry, transcripts and the live
preview. Synthetic clips and fakes only; no real model is loaded."""
import base64
import json
import logging

import numpy as np
import pytest
import yaml

from tests.fakes import FakeAudioTagger, FakeEmbedder, FakeExpressionScorer, FakePersonCounter, FakeTranscriber
from tests.test_cli import _predict_yaml
from tpa_index import config, process, schema
from tpa_index.preview import BLOCKS, HIDDEN, HIDDEN_PEOPLE, FaceBlur, Live, to_data_url

AUDIO_LABELS = ["singing_voice", "music", "speech", "applause", "silence"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".npy", ".webp", ".gif", ".tif", ".tiff"}


class FakeDetector:
    """YuNet stand-in: fixed rows [x, y, w, h, 10 landmark values, score] for every frame."""

    def __init__(self, rows=(), fail=False):
        self.rows, self.fail, self.calls = [list(r) for r in rows], fail, 0

    def setInputSize(self, size):
        pass

    def detect(self, img):
        self.calls += 1
        if self.fail:
            raise RuntimeError("detector broke")
        return 1, (np.array(self.rows, np.float32) if self.rows else None)


def _face(x, y, w, h, score=0.9):
    return [x, y, w, h, *([0.0] * 10), score]


def _models(pipeline, **over):
    expr = list(pipeline["signals"]["expression"]["mapping"])
    models = {"transcript": FakeTranscriber(), "audio": FakeAudioTagger(AUDIO_LABELS),
              "people": FakePersonCounter(), "expression": FakeExpressionScorer(expr)}
    models.update(over)
    return models


@pytest.fixture
def world(clip_videos, pipeline, tmp_path, monkeypatch):
    """t01-t03 in a videos config, scratch under tmp, fake model loaders."""
    monkeypatch.setattr(config, "LOCAL_VIDEOS", _predict_yaml(tmp_path))
    monkeypatch.setenv("TPA_SCRATCH_DIR", str(tmp_path / "scratch"))
    models = _models(pipeline)
    makers = {"make_embedder": lambda model_id, dev: FakeEmbedder(),
              "make_models": lambda stages, pipe, dev: {s: models[s] for s in stages},
              "make_blur": lambda pipe: FaceBlur(FakeDetector())}
    return makers, models


def _run(video_id, makers, timeout=120):
    job = process.start(video_id, **makers)
    assert not isinstance(job, str), job
    assert job.done.wait(timeout), "process run did not finish"
    return job


def _local():
    return yaml.safe_load(schema.local_runs_path().read_text()) or {}


def _official(tmp_path, **stages):
    schema.OFFICIAL_RUNS.write_text(yaml.safe_dump(stages))


# ---- progress and the end-to-end run -----------------------------------------------------------------------------
def test_run_reaches_every_stage_and_registers_local_runs(world):
    makers, _ = world
    job = _run("t01", makers)
    snap = job.snapshot()
    assert snap["done"] and set(snap["status"].values()) == {"done"}, snap["errors"]
    assert snap["n_segments"] == 3 and snap["counts"] == {"predict": 3, "frames": 3}
    local = _local()
    assert set(local) == {"predict", "transcript", "audio", "people", "expression"}
    assert all(len(ids) == 1 for ids in local.values())
    assert local == job.run_ids
    assert schema.load_official() == {}  # official_runs.yaml is never written
    assert "Search moments and Review queue" in job.message
    assert job.message.endswith(process.LAPTOP_NOTE) and "never counted" not in job.message


def test_stages_advance_in_order_and_frame_stages_run_together(world):
    job = process.Job({"video_id": "t01"}, None)
    job.set("predict", "done")
    with job._lock:
        job._advance()
    assert job.status["transcript"] == "running" and job.status["audio"] == "waiting"
    job.report(("t01", "transcript"), "done:transcript-x")
    assert job.status["audio"] == "running"
    job.report(("t01", "audio"), "done:audio-x")
    assert job.status["people"] == job.status["expression"] == "running"
    job.report(("t01", "people"), "failed:RuntimeError")
    assert job.status["people"] == "failed" and "stopped unexpectedly (RuntimeError)" in job.errors["people"]


def test_estimate_from_earlier_laptop_runs(world):
    makers, _ = world
    assert process.estimate(60.0) is None  # first run: no estimate
    _run("t01", makers)
    est = process.estimate(60.0)
    assert est is not None and est > 0
    assert process.estimate(120.0) == pytest.approx(2 * est)


def test_app_generator_streams_progress_then_the_result(world, labels):
    from tpa_index.app import process_video

    makers, _ = world
    out = list(process_video("t01", labels=labels, poll=0.05, **makers))
    status, preview, message, table = out[-1]
    assert "Finished in" in status and "Live preview" in preview and "found" in message
    first = list(process_video("", labels=labels))
    assert first[-1][2] == "Pick a video first."


def test_one_run_at_a_time(world):
    makers, _ = world
    assert process._LOCK.acquire(blocking=False)
    try:
        assert process.start("t01", **makers) == process.BUSY
    finally:
        process._LOCK.release()


def test_choices_show_plain_names_phase_and_official_status_only(world, tmp_path):
    choices, note = process.video_choices()
    assert note == "" and [v for _, v in choices] == ["t01", "t02", "t03"]
    assert all(".mp4" not in name for name, _ in choices) and choices[0][0] == "Video 1 · early · new"
    assert process.display_name("t01") == "Video 1 · early" and process.display_name("zz9") == "zz9"


def test_progress_header_shows_the_plain_name(world):
    from tpa_index import ui_html

    makers, _ = world
    snap = _run("t01", makers).snapshot()
    html = ui_html.process_status(snap)
    assert snap["name"] == "Video 1 · early" and "Progress · Video 1 · early" in html


def test_deleted_local_run_folder_is_skipped(world):
    import shutil

    makers, _ = world
    job = _run("t01", makers)
    shutil.rmtree(schema._runs_root(None) / job.run_ids["predict"][0])  # someone tidies the runs folder
    assert "t01" not in schema.read_official("moments", include_local=True)["video_id"].to_pylist()
    assert "t01" not in schema.read_official("videos", include_local=True)["video_id"].to_pylist()
    choices, note = process.video_choices()
    assert note == "" and [v for _, v in choices] == ["t01", "t02", "t03"]


# ---- failures ----------------------------------------------------------------------------------------------------
def test_missing_model_fails_only_its_step_in_plain_english(world, pipeline):
    makers, models = world

    def make_models(stages, pipe, dev):
        if stages == ["people"]:
            raise config.ConfigError("weights not found: yolo11n.pt; run scripts/download_weights.sh on the login node")
        return {s: models[s] for s in stages}

    job = _run("t01", {**makers, "make_models": make_models})
    snap = job.snapshot()
    assert snap["status"]["people"] == "failed"
    assert {snap["status"][s] for s in ("predict", "transcript", "audio", "expression")} == {"done"}
    assert "not on this laptop yet" in snap["errors"]["people"] and "yolo11n" not in snap["errors"]["people"]
    assert "people" not in _local() and "Some steps did not finish" in job.message


def test_unreadable_video_stops_with_a_message_and_registers_nothing(world, clips, tmp_path):
    makers, _ = world
    (clips / "t02.mp4").rename(clips / "t02.bak")
    try:
        job = _run("t02", makers)
    finally:
        (clips / "t02.bak").rename(clips / "t02.mp4")
    assert job.status["predict"] == "failed" and set(job.status.values()) == {"failed", "skipped"}
    assert job.message == "Processing stopped: Activity: the video file could not be opened or read."
    assert not schema.local_runs_path().exists()
    assert "t02.mp4" not in job.message


def test_signal_step_failure_keeps_the_others(world, pipeline):
    makers, models = world
    models["audio"] = FakeAudioTagger(AUDIO_LABELS, fail_on_call=1)
    job = _run("t01", makers)
    assert job.status["audio"] == "failed" and job.status["transcript"] == job.status["people"] == "done"
    assert "audio" not in _local()


# ---- registry: local runs never replace official results ---------------------------------------------------------
def _official_predict(video):
    from tpa_index import cli
    from tpa_index.runs import track_run

    pipeline, labels = config.load_pipeline(), config.load_labels()
    with track_run("predict", model="fake") as ctx:
        cli._predict_run(video, ctx.run_id, pipeline, labels, FakeEmbedder(), "cpu")
    return ctx.run_id


def test_local_runs_never_change_official_reads(world, clip_videos, tmp_path):
    makers, _ = world
    official = _official_predict(clip_videos["t01"])
    _official(tmp_path, predict=[official])
    before = {t: schema.read_official(t) for t in ("videos", "segments", "predictions", "moments")}
    _run("t01", makers)  # reprocess a video that has official results
    _run("t03", makers)  # and a new one
    for t, tbl in before.items():
        assert schema.read_official(t).equals(tbl), t  # the default (every result) is unchanged
    assert set(schema.local_only()["predict"]) == {"t03"}  # official videos never take a local run
    vids = set(schema.read_official("videos", include_local=True)["video_id"].to_pylist())
    assert vids == {"t01", "t03"}
    t01 = [r for r in schema.read_official("moments", include_local=True).to_pylist() if r["video_id"] == "t01"]
    assert {r["run_id"] for r in t01} <= {official}
    job = _run("t01", makers)
    assert job.message.endswith(process.LAPTOP_NOTE) and process.OFFICIAL_NOTE in job.message
    assert "Search moments and Review queue" not in job.message


def test_registry_replaces_older_local_run_and_refuses_official(world, clip_videos, tmp_path):
    makers, _ = world
    first = _run("t01", makers).run_ids["predict"]
    second = _run("t01", makers).run_ids["predict"]
    assert _local()["predict"] == second != first
    official = _official_predict(clip_videos["t03"])
    _official(tmp_path, predict=[official])
    with pytest.raises(schema.SchemaError):
        schema.register_local({"predict": [official]})


def test_search_and_review_badge_local_videos_and_refuse_reviews(world, labels, pipeline):
    from tpa_index.app import LOCAL_BADGE, LOCAL_READ_ONLY, ReviewController
    from tpa_index.demo import DemoController
    from tpa_index.schema import read_reviews

    makers, _ = world
    _run("t01", makers)
    tl = DemoController(labels=labels, pipeline=pipeline)
    names = dict((v, n) for n, v in tl.video_choices()[0])
    assert LOCAL_BADGE in names["t01"]
    rv = ReviewController(labels=labels, pipeline=pipeline)
    moment = {"moment_id": "t01_0_dance", "video_id": "t01", "start_s": 0.0, "end_s": 8.0, "primary_label": "dance",
              "secondary_labels": [], "indicators": [], "confidence": 0.6, "needs_review": False, "run_id": "x"}
    assert rv.decide(moment, "ana", "accept", ["dance"])[3] == LOCAL_READ_ONLY
    assert rv.act(moment, "ana", "dance", "accept")[3] == LOCAL_READ_ONLY
    assert read_reviews().num_rows == 0
    assert LOCAL_BADGE in rv.suggestion(moment)
    assert tl.strength(moment) is None  # no calibrated cut-off for the laptop model


def test_scorecard_shows_no_laptop_note_and_still_leaves_laptop_runs_out(world):
    from tpa_index.app import build_app

    makers, _ = world
    _run("t01", makers)
    scorecard = [c for c in build_app().config["components"]
                 if str(c["props"].get("elem_id") or "").startswith("scorecard-")]
    assert scorecard and not any(c["props"].get("elem_id") == "scorecard-local" for c in scorecard)
    assert "processed on this laptop" not in json.dumps(scorecard, default=str)
    assert schema.read_official("videos").num_rows == 0  # the default reads (Scorecard, insights) skip laptop runs
    assert "t01" in schema.read_official("videos", include_local=True)["video_id"].to_pylist()


# ---- transcripts: never written on a laptop run ------------------------------------------------------------------
def test_no_transcript_text_is_written(world, pipeline, tmp_path):
    sentinel = "ZQXJ-SENTINEL-4412"
    makers, models = world
    models["transcript"] = FakeTranscriber(sentinel=sentinel)
    job = _run("t01", makers)
    assert job.status["transcript"] == "done"
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert not any(p.name == "transcript.json" for p in files)
    assert not any(sentinel.encode() in p.read_bytes() for p in files)
    segs = schema.read_table("segments", run_ids=job.run_ids["transcript"]).to_pylist()
    assert any(s["has_speech"] for s in segs) and all(s["transcript_chars"] is not None for s in segs)


def test_run_signals_keep_transcript_false(clip_videos, pipeline, labels, tmp_path, monkeypatch):
    from tests.test_signals import official_clip_runs
    from tpa_index import signals

    monkeypatch.setenv("TPA_SCRATCH_DIR", str(tmp_path / "scratch"))
    official_clip_runs([clip_videos["t01"]], pipeline, tmp_path, monkeypatch)
    models = {"transcript": FakeTranscriber()}
    signals.run_signals([clip_videos["t01"]], ["transcript"], models, pipeline, labels, keep_transcript=False)
    assert not list(tmp_path.rglob("transcript.json"))
    signals.run_signals([clip_videos["t01"]], ["transcript"], models, pipeline, labels, force=True)
    assert len(list(tmp_path.rglob("transcript.json"))) == 1  # HPC default unchanged


# ---- live preview: blurred, fail closed, memory only -------------------------------------------------------------
def _noise(h=120, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_preview_frame_is_pixelated_inside_face_and_head_boxes_only():
    frame = _noise()
    blur = FaceBlur(FakeDetector([_face(60, 30, 40, 40)]))
    person = (0.0, 0.5, 0.25, 1.0)  # x 0-40, y 60-120 in pixels; head area y 60-73
    out = blur.apply(frame, [person])
    assert out is not None and out.shape == frame.shape and out is not frame
    mask = np.zeros(frame.shape[:2], bool)
    mask[20:80, 50:110] = True  # face box padded by 25% on each side
    mask[60:74, 0:40] = True  # top 22% of the person box
    assert np.array_equal(out[~mask], frame[~mask])  # nothing else changes
    face = out[20:80, 50:110].reshape(-1, 3)
    assert len(np.unique(face, axis=0)) <= BLOCKS * BLOCKS  # coarse blocks, no detail left
    assert out[20:80, 50:110].std() < frame[20:80, 50:110].std() / 2
    assert not np.array_equal(out[60:74, 0:40], frame[60:74, 0:40])
    assert np.array_equal(frame, _noise())  # the input frame is never modified


def test_preview_uses_looser_face_settings():
    frame = _noise()
    small_low_score = FaceBlur(FakeDetector([_face(10, 10, 12, 12, score=0.55)])).apply(frame)
    assert not np.array_equal(small_low_score[10:22, 10:22], frame[10:22, 10:22])  # below 48 px and score 0.8


def test_preview_fails_closed():
    frame = _noise()
    assert FaceBlur(FakeDetector(fail=True)).apply(frame) is None
    for blur in (FaceBlur(FakeDetector(fail=True)), None):
        live = Live(blur, interval=0.0, people=FakePersonCounter())
        live.window({"segment_id": "t01_0", "start_s": 0.0, "end_s": 8.0}, frame[None], {"dance": 0.4})
        view = live.view()
        assert view["image"] is None and view["hidden"] == HIDDEN


def test_shown_image_is_the_blurred_frame():
    import cv2

    class Grey:
        def apply(self, frame, boxes=None):
            return np.full_like(frame, 128)

    class Nobody:
        def boxes(self, frames):
            return [[] for _ in frames]  # no rectangles drawn: the image is only the blurred frame

    live = Live(Grey(), interval=0.0, people=Nobody())
    live.window({"segment_id": "t01_0", "start_s": 0.0, "end_s": 8.0}, _noise()[None], {"dance": 0.4, "singing": 0.2})
    url = live.view()["image"]
    assert url.startswith("data:image/jpeg;base64,")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert abs(int(img.mean()) - 128) <= 2 and img.std() < 2  # the raw noise frame never reaches the page
    view = live.view()
    assert view["activity"] == "dance" and view["others"] == [("singing", pytest.approx(0.5))]


class OnePerson:
    """Person counter stand-in: one person filling the left quarter of every frame, bottom half."""

    def __init__(self):
        self.calls = []

    def boxes(self, frames):
        self.calls.append(len(frames))
        return [[(0.0, 0.5, 0.25, 1.0)] for _ in frames]


def test_preview_covers_the_head_area_when_the_stage_passes_no_boxes():
    """The activity stage passes no person boxes: the preview runs person detection on the one shown frame, so a face
    the detector misses (none here) is still covered by the head area."""
    frame = _noise()
    people = OnePerson()

    class Keep:
        def apply(self, img, boxes=None):
            self.out = FaceBlur(FakeDetector()).apply(img, boxes)  # no face found
            return self.out

    blur = Keep()
    live = Live(blur, interval=0.0, people=people)
    live.window({"segment_id": "t01_0", "start_s": 0.0, "end_s": 8.0}, frame[None], {"dance": 0.4})
    assert people.calls == [1] and live.view()["image"] and live.view()["shown"]["people_boxes"]
    assert not np.array_equal(blur.out[60:74, 0:40], frame[60:74, 0:40])  # top 22% of the person box
    assert np.array_equal(blur.out[80:, 60:], frame[80:, 60:])


def test_preview_shows_no_image_without_person_detection():
    calls = []

    class Spy:
        def apply(self, frame, boxes=None):
            calls.append(1)
            return frame

    class Broken:
        def boxes(self, frames):
            raise RuntimeError("yolo broke")

    seg = {"segment_id": "t01_0", "start_s": 0.0, "end_s": 8.0}
    for people in (None, Broken()):
        live = Live(Spy(), interval=0.0, people=people)
        live.window(seg, _noise()[None], {"dance": 0.4})
        view = live.view()
        assert view["image"] is None and view["hidden"] == HIDDEN_PEOPLE
    assert calls == []  # face-only blur is never shown
    live = Live(Spy(), interval=0.0)
    live.frames(seg, _noise()[None], [[(0.0, 0.5, 0.25, 1.0)]], None)  # the frame pass brings its own boxes
    assert live.view()["image"] and calls == [1]


def test_preview_boxes_are_plain_rectangles():
    img = np.full((120, 160, 3), 255, np.uint8)
    url = to_data_url(img, [(0.1, 0.1, 0.5, 0.9)])
    assert url.startswith("data:image/jpeg;base64,")


def test_preview_throttles_to_about_once_a_second():
    t = [0.0]
    calls = []

    class Spy:
        def apply(self, frame, boxes=None):
            calls.append(1)
            return frame

    live = Live(Spy(), interval=1.0, clock=lambda: t[0], people=FakePersonCounter())
    seg = {"segment_id": "s", "start_s": 0.0, "end_s": 8.0}
    for step in (0.0, 0.3, 0.6, 1.1, 1.5, 2.2):
        t[0] = step
        live.window(seg, _noise()[None], {"dance": 0.5})
    assert len(calls) == 3 and live.view()["counts"]["predict"] == 6


def test_preview_callbacks_never_stop_the_pipeline():
    live = Live(None)
    live.window(None, None, None)  # nonsense input: swallowed
    live.frames({"segment_id": "s", "start_s": 0, "end_s": 1}, None, None, None)
    assert live.view()["counts"]["predict"] == 0


def test_full_run_writes_no_image_and_logs_no_frame(world, tmp_path, monkeypatch, caplog):
    """Nothing image-like is written anywhere under the sandbox (data, outputs, scratch, Gradio and system temp, cwd),
    and no frame reaches a log line."""
    import tempfile

    import cv2
    import PIL.Image

    makers, _ = world
    for name in ("gradio", "tmp"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    monkeypatch.chdir(tmp_path)

    def refuse(*a, **k):
        raise AssertionError("an image was written to disk")

    monkeypatch.setattr(cv2, "imwrite", refuse)
    monkeypatch.setattr(PIL.Image.Image, "save", refuse)
    caplog.set_level(logging.DEBUG)
    shown = []
    blur = FaceBlur(FakeDetector([_face(10, 10, 30, 30)]))
    orig = Live._show

    def spy(self, seg, frames, boxes):
        orig(self, seg, frames, boxes)
        shown.append(self.image)

    monkeypatch.setattr(Live, "_show", spy)
    job = _run("t01", {**makers, "make_blur": lambda pipe: blur})
    assert set(job.status.values()) == {"done"} and any(s for s in shown)
    assert 1 <= len({s for s in shown if s}) <= blur.detector.calls  # every new image went through face blurring
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    magic = (b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"RIFF")
    assert not [p.name for p in files if p.suffix.lower() in IMAGE_EXTS or p.read_bytes()[:4].startswith(magic)]
    assert "base64" not in caplog.text and "data:image" not in caplog.text
