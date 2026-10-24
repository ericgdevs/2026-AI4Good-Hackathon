import subprocess

import av
import pytest

from tpa_index.clips import ClipCutter, session_dir
from tpa_index.paths import repo_root


@pytest.fixture
def cutter():
    c = ClipCutter(session_dir(), clip_height=240)
    yield c
    c.cleanup()


def _video(clips, vid="t01", duration=20.0):
    return {"video_id": vid, "filename": f"{vid}.mp4", "duration_s": duration}


def _info(path):
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        return v.codec_context.name, float(c.duration / 1e6)


def test_cut_is_h264_with_the_requested_length(clips, clip_videos, cutter):
    path = cutter.cut(_video(clips), 4.0, 12.0)
    codec, duration = _info(path)
    assert codec == "h264" and duration == pytest.approx(8.0, abs=0.4)
    assert path.parent == cutter.session


def test_padding_is_clipped_to_video_bounds(clips, clip_videos, cutter):
    start = cutter.cut(_video(clips), 0.0, 8.0, pad_s=3)
    assert _info(start)[1] == pytest.approx(11.0, abs=0.4)  # 0 .. 11 (no negative start)
    end = cutter.cut(_video(clips), 14.0, 20.0, pad_s=3)
    assert _info(end)[1] == pytest.approx(9.0, abs=0.4)  # 11 .. 20 (stops at the video end)


def test_previous_clip_is_deleted_and_prefetch_is_reused(clips, clip_videos, cutter):
    first = cutter.cut(_video(clips), 0.0, 8.0)
    cutter.prefetch(_video(clips), 6.0, 14.0)
    second = cutter.cut(_video(clips), 6.0, 14.0)
    assert not first.exists() and second.exists()
    assert len(list(cutter.session.glob("*.mp4"))) == 1  # prefetched file used, nothing left over


def test_cleanup_removes_session_outside_repo(cutter):
    assert not cutter.session.is_relative_to(repo_root())
    cutter.cleanup()
    assert not cutter.session.exists()


def test_missing_file_error_is_redacted(clips, clip_videos, cutter):
    from tpa_index.paths import data_dir

    with pytest.raises(RuntimeError) as err:
        cutter.cut({"video_id": "t09", "filename": "secret_take_9.mp4", "duration_s": 20.0}, 0, 8)
    assert str(data_dir()) not in str(err.value) and "secret_take_9" not in str(err.value)


def test_no_files_written_to_repo(clips, clip_videos, cutter):
    before = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root(), capture_output=True, text=True).stdout
    cutter.cut(_video(clips), 0.0, 8.0)
    after = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root(), capture_output=True, text=True).stdout
    assert before == after
