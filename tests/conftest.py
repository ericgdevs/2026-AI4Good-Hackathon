import pytest

from tpa_index import config, paths


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Point data/output at tmp dirs and never let tests see real local video metadata."""
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    no_local = tmp_path / "no-local-videos.yaml"
    monkeypatch.setenv("TPA_LOCAL_VIDEOS", str(no_local))  # subprocesses inherit this
    monkeypatch.setattr(config, "LOCAL_VIDEOS", no_local)
    no_thresholds = tmp_path / "no-thresholds.local.yaml"
    monkeypatch.setenv("TPA_THRESHOLDS_FILE", str(no_thresholds))
    monkeypatch.setattr(config, "THRESHOLDS_LOCAL", no_thresholds)
    paths._reset_redaction_cache()
    yield
    paths._reset_redaction_cache()


@pytest.fixture
def labels():
    return config.load_labels()


@pytest.fixture
def pipeline():
    return config.load_pipeline()


@pytest.fixture
def example_videos():
    return config.load_videos(config.EXAMPLE_VIDEOS)


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    """Synthetic clips only: t01 landscape, t02 portrait, t03 no audio, t04 static (20 s each)."""
    from tests.clips import make_clip

    root = tmp_path_factory.mktemp("clips")
    make_clip(root / "t01.mp4", 160, 90)
    make_clip(root / "t02.mp4", 90, 160)
    make_clip(root / "t03.mp4", 160, 90, audio=False)
    make_clip(root / "t04.mp4", 160, 90, static=True)
    return root


@pytest.fixture
def clip_videos(clips, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(clips))
    base = {"phase": "early", "cohort": None, "split": "dev"}
    return {
        "t01": {**base, "video_id": "t01", "filename": "t01.mp4", "orientation": "landscape", "kind": "live"},
        "t02": {**base, "video_id": "t02", "filename": "t02.mp4", "orientation": "portrait", "kind": "clip"},
        "t03": {**base, "video_id": "t03", "filename": "t03.mp4", "orientation": "landscape", "kind": "live"},
        "t04": {**base, "video_id": "t04", "filename": "t04.mp4", "orientation": "landscape", "kind": "live"},
    }
