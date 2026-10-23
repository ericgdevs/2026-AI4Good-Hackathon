import logging
import os
import subprocess
import sys
import textwrap

import av
import pytest

from tpa_index import paths
from tpa_index.paths import GuardError, check_guard, data_dir, redact, repo_root


def _guard_fails(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(GuardError) as err:
        check_guard()
    assert name in str(err.value) and value not in str(err.value)


@pytest.mark.parametrize("name", ["DATA_DIR", "OUTPUT_DIR"])
def test_inside_repo_rejected(monkeypatch, name):
    _guard_fails(monkeypatch, name, str(repo_root() / "data"))


def test_relative_path_into_repo_rejected(monkeypatch):
    monkeypatch.chdir(repo_root())
    _guard_fails(monkeypatch, "OUTPUT_DIR", "outputs")


def test_home_relative_symlink_into_repo_rejected(monkeypatch, tmp_path):
    (tmp_path / "link").symlink_to(repo_root())
    monkeypatch.setenv("HOME", str(tmp_path))
    _guard_fails(monkeypatch, "DATA_DIR", "~/link/data")


def test_symlink_into_repo_rejected(monkeypatch, tmp_path):
    (tmp_path / "sneaky").symlink_to(repo_root() / "config")
    _guard_fails(monkeypatch, "DATA_DIR", str(tmp_path / "sneaky"))


def test_unset_rejected(monkeypatch):
    monkeypatch.delenv("DATA_DIR")
    with pytest.raises(GuardError, match="DATA_DIR is not set"):
        check_guard()


def test_outside_repo_passes():
    check_guard()


def test_guard_does_not_import_torch():
    code = "import sys, tpa_index.paths, tpa_index.cli; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_redact_paths_and_filenames():
    root = str(data_dir())
    full = redact(f"failed on {root}/mock/mock_live_01.mp4: bad header")
    assert root not in full and "mock_live_01" not in full and full.startswith("failed on <DATA_DIR>/<file>")
    # everything after an unquoted data path is hidden (names may contain spaces): safe over-redaction
    assert "bad header" not in full
    assert redact("retrying mock_live_01.mp4 and mock_live_01") == "retrying <video:mock01> and <video:mock01>"
    assert redact(f"out={os.environ['OUTPUT_DIR']}") == "out=<OUTPUT_DIR>"


def test_redact_third_party_error():
    target = str(data_dir() / "mock" / "mock_live_01.mp4")
    with pytest.raises(Exception) as err:
        av.open(target)
    assert target in str(err.value)  # PyAV leaks the path...
    out = redact(str(err.value))
    assert str(data_dir()) not in out and "mock_live_01" not in out  # ...redaction removes it


def test_log_filter_redacts():
    record = logging.LogRecord("x", logging.ERROR, __file__, 1, "reading %s", (str(data_dir() / "a.mp4"),), None)
    paths._RedactFilter().filter(record)
    assert str(data_dir()) not in record.getMessage() and "<DATA_DIR>" in record.getMessage()


def test_excepthook_redacts_uncaught_errors():
    script = textwrap.dedent("""
        import os
        from tpa_index.paths import install_redaction
        install_redaction()
        install_redaction()  # idempotent
        raise FileNotFoundError(os.path.join(os.environ["DATA_DIR"], "secret_clip.mp4"))
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    assert "<DATA_DIR>" in result.stderr and os.environ["DATA_DIR"] not in result.stderr
    assert "secret_clip" not in result.stderr


def test_redact_unlisted_filename_with_spaces():
    out = redact(f"cannot open '{data_dir()}/Spring Show 2025.mp4': error")
    assert "Spring" not in out and "Show" not in out and "<DATA_DIR>/<file>" in out
