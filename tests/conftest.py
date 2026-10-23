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
    monkeypatch.setattr(config, "LOCAL_VIDEOS", tmp_path / "no-local-videos.yaml")
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
