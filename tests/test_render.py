"""006 render-overlay: synthetic clips and fakes only (no real footage, no real model)."""
import tempfile
from pathlib import Path

import av
import numpy as np
import pytest

from tests.fakes import FakeAudioTagger, FakeEmbedder, FakeExpressionScorer, FakePersonCounter, FakeTranscriber
from tests.test_cli import _predict_yaml
from tests.test_process import AUDIO_LABELS, IMAGE_EXTS
from tpa_index import config, render, schema
from tpa_index.config import ConfigError, load_theme
from tpa_index.paths import GuardError, repo_root

MARK = (255, 0, 255)  # what the fake blur paints: never a colour of the synthetic clips
SENTINEL = "ZQXJ-RENDER-SENTINEL"


class MarkBlur:
    """Blur stand-in: returns a magenta copy (so blurred pixels are recognisable), or None to test failing closed."""

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def apply(self, frame, person_boxes=None):
        self.calls += 1
        return None if self.fail else np.full_like(frame, MARK)


@pytest.fixture
def world(clip_videos, pipeline, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_VIDEOS", _predict_yaml(tmp_path))
    expr = list(pipeline["signals"]["expression"]["mapping"])
    models = {"transcript": FakeTranscriber(sentinel=SENTINEL), "audio": FakeAudioTagger(AUDIO_LABELS),
              "people": FakePersonCounter(), "expression": FakeExpressionScorer(expr)}
    return {"make_embedder": lambda model_id, dev: FakeEmbedder(),
            "make_models": lambda stages, pipe, dev: {s: models[s] for s in stages}}


def _frames(path):
    with av.open(str(path)) as c:
        assert not c.streams.audio  # video only
        v = c.streams.video[0]
        assert (v.width, v.height) == (render.WIDTH, render.HEIGHT)
        return [f.to_ndarray(format="rgb24") for f in c.decode(v)]


def _panel_centre(frame, x):
    y = render.TOP + render.IMG_H // 2
    return frame[y - 5:y + 5, x + render.PANEL_W // 2 - 5:x + render.PANEL_W // 2 + 5].reshape(-1, 3).mean(axis=0)


def _files(*roots):
    return {p for r in roots for p in r.rglob("*") if p.is_file()}


def test_both_panels_blurred_video_only_and_nothing_else_written(world, tmp_path, monkeypatch):
    import cv2
    from PIL import Image

    def no_image_write(*a, **k):
        raise AssertionError("an image was written")

    monkeypatch.setattr(cv2, "imwrite", no_image_write)
    monkeypatch.setattr(Image.Image, "save", no_image_write)
    from PIL import ImageDraw

    drawn, real_text = [], ImageDraw.ImageDraw.text

    def record(self, xy, text, *a, **k):
        drawn.append(text)
        return real_text(self, xy, text, *a, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    tmp = Path(tempfile.gettempdir())
    before = _files(tmp_path) | {p for p in tmp.iterdir() if p.is_file()}

    blur = MarkBlur()
    path, secs = render.render_overlay("t01", 2.0, 3, make_blur=lambda pipe: blur, **world)
    assert path.parent.name == "renders" and path.suffix == ".mp4" and secs == 3
    frames = _frames(path)
    assert len(frames) == 3 * render.FPS and blur.calls == len(frames)
    left, right = render.PAD, render.PAD + render.PANEL_W + render.GAP
    for f in frames[::6]:  # the blurred (magenta) frame fills both panels; no raw clip colour shows
        assert np.allclose(_panel_centre(f, left), MARK, atol=30)
        assert np.allclose(_panel_centre(f, right), MARK, atol=30)

    after = _files(tmp_path) | {p for p in tmp.iterdir() if p.is_file()}
    new = after - before
    assert new == {path}  # only the MP4: no frames, crops, audio or partial files
    assert not any(p.suffix.lower() in IMAGE_EXTS | {".wav"} for p in new)
    assert not schema.local_runs_path().exists() and schema.load_official() == {}  # no run registered
    assert not (path.parent.parent / "runs").exists()

    text = " | ".join(drawn)
    assert SENTINEL not in text  # transcript text is never drawn
    assert "Speech: yes" in text and "AI suggestion (uncalibrated): " in text and "Expression (unvalidated): " in text
    assert "person detection (counts only): 2 people" in text and "What the AI notices" in text


def test_fails_closed_to_a_grey_frame(world):
    path, _ = render.render_overlay("t01", 0.0, 1, make_blur=lambda pipe: MarkBlur(fail=True), **world)
    grey = np.array(render._rgb(load_theme()["muted"]))
    for f in _frames(path):
        left, right = render.PAD, render.PAD + render.PANEL_W + render.GAP
        assert np.allclose(_panel_centre(f, left), grey, atol=12) and np.allclose(_panel_centre(f, right), grey, atol=12)
        corners = np.array([f[20, 20], f[-20, -20], f[20, -20], f[-20, 20]], float)
        assert np.allclose(corners, grey, atol=12)  # the whole frame is grey: nothing from the video


def test_default_blur_is_the_preview_blur_and_refuses_without_it(world, monkeypatch):
    from tpa_index import preview

    calls = []
    monkeypatch.setattr(preview.FaceBlur, "load", classmethod(lambda cls, pipe: calls.append(1) or MarkBlur()))
    render.render_overlay("t01", 0.0, 1, **world)
    assert calls == [1]

    def broken(cls, pipe):
        raise RuntimeError("no YuNet")

    monkeypatch.setattr(preview.FaceBlur, "load", classmethod(broken))
    with pytest.raises(ConfigError, match="face blurring is not available"):
        render.render_overlay("t01", 0.0, 1, **world)


def test_output_refused_inside_the_repo(world):
    for out in (repo_root() / "x.mp4", repo_root() / "docs" / "renders" / "y.mp4"):
        with pytest.raises(GuardError, match="outside the repo"):
            render.render_overlay("t01", 0.0, 1, out=str(out), make_blur=lambda pipe: MarkBlur(), **world)
        assert not out.exists()


def test_missing_models_show_not_available_and_inputs_are_checked(world, tmp_path):
    def none_load(stages, pipe, dev):
        raise ConfigError("weights not found")

    def no_embedder(model_id, dev):
        raise ConfigError("weights not found")

    out = tmp_path / "elsewhere" / "clip.mp4"
    path, secs = render.render_overlay("t01", 18.0, 10, out=str(out), make_blur=lambda pipe: MarkBlur(),
                                       make_embedder=no_embedder, make_models=none_load)
    assert path == out.resolve() and secs == pytest.approx(2.0, abs=0.1)  # clipped to the end of the 20 s clip
    assert render.parse_time("01:30") == 90 and render.parse_time("1:02:03") == 3723 and render.parse_time("45") == 45
    with pytest.raises(ConfigError):
        render.parse_time("ninety")
    with pytest.raises(ConfigError, match="past the end"):
        render.render_overlay("t01", 999.0, 5, make_blur=lambda pipe: MarkBlur(), **world)
    with pytest.raises(ConfigError, match="unknown video_id"):
        render.render_overlay("zz9", 0.0, 5, make_blur=lambda pipe: MarkBlur(), **world)


def test_cli_prints_only_path_and_duration(world, monkeypatch, capsys):
    from tpa_index import cli, preview

    monkeypatch.setattr(cli, "_make_embedder", world["make_embedder"])
    monkeypatch.setattr(cli, "_make_signal_models", world["make_models"])
    monkeypatch.setattr(preview.FaceBlur, "load", classmethod(lambda cls, pipe: MarkBlur()))
    assert cli.main(["render-overlay", "--video-id", "t01", "--start", "00:01", "--duration", "2"]) == 0
    out = capsys.readouterr()
    lines = out.out.strip().splitlines()
    assert len(lines) == 1 and lines[0].endswith(".mp4 (2 s)") and out.err == ""


def test_fonts_draw_letters_not_boxes():
    for bold in (False, True):
        assert render.has_letters(render._font(30, bold))
