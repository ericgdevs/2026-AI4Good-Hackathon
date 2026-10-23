import shutil
import subprocess

import av
import numpy as np
import pytest

from tpa_index.config import effective_settings
from tpa_index.paths import repo_root
from tpa_index.schema import validate
from tpa_index.segment import _to_rgb, discover_videos, probe, segment_video

DEFAULT = {"window_s": 8, "overlap_s": 2, "frames_per_window": 4}


def run(video, settings=DEFAULT):
    return list(segment_video(video, settings))


def test_landscape_segments_ids_and_frames(clip_videos):
    out = run(clip_videos["t01"])
    assert [w.row["segment_id"] for w in out] == ["t01_0", "t01_6000", "t01_12000"]
    assert out[-1].row["end_s"] == pytest.approx(20, abs=0.15)
    for w in out:
        assert w.frames.shape == (4, 160, 160, 3) and w.frames.dtype == np.uint8


def test_landscape_letterbox_pads_top_bottom_without_crop(clip_videos):
    frame = run(clip_videos["t01"])[0].frames[0]
    assert frame[:30].max() < 20 and frame[-30:].max() < 20  # (160-90)/2 = 35 px padding each side
    assert frame[37:123, 1].mean() > 200  # white left border kept (mean: lossy codec blurs edges): nothing cropped


def test_portrait_letterbox_pads_left_right(clip_videos):
    frame = run(clip_videos["t02"])[0].frames[0]
    assert frame.shape == (160, 160, 3)
    assert frame[:, :30].max() < 20 and frame[:, -30:].max() < 20
    assert frame[:, 80].max() > 100


def test_frames_are_spread_across_the_window(clip_videos):
    frames = run(clip_videos["t01"])[0].frames  # window 0-8 s; samples at 1, 3, 5, 7 s
    centres = [tuple(f[80, 80]) for f in frames]
    assert len(set(centres)) >= 3


def test_audio_length_and_content(clip_videos):
    for w in run(clip_videos["t01"]):
        assert len(w.audio) == round((w.row["end_s"] - w.row["start_s"]) * 16000)
        assert np.sqrt(np.mean(w.audio ** 2)) > 0.05
    for w in run(clip_videos["t03"]):
        assert len(w.audio) > 0 and not w.audio.any()


def test_kind_override_changes_windows(clip_videos, pipeline):
    settings = effective_settings(pipeline, clip_videos["t02"])  # kind clip: 4 s / 1 s
    assert (settings["window_s"], settings["overlap_s"]) == (4, 1)
    assert len(run(clip_videos["t02"], settings)) > len(run(clip_videos["t02"]))


def test_by_video_override_wins(pipeline):
    p = {**pipeline, "overrides": {"by_kind": {"clip": {"window_s": 4, "overlap_s": 1}},
                                   "by_video": {"t02": {"frames_per_window": 8, "window_s": 6}}}}
    assert effective_settings(p, {"video_id": "t02", "kind": "clip"}) == \
        {"window_s": 6, "overlap_s": 1, "frames_per_window": 8}


def test_rows_validate_with_null_signals(clip_videos):
    rows = [w.row for w in run(clip_videos["t01"])]
    assert validate("segments", rows).num_rows == 3
    assert all(r["person_count"] is None for r in rows)


def test_probe(clip_videos, clips):
    info = probe(clips / "t02.mp4")
    assert info["orientation"] == "portrait" and info["has_audio"] and info["fps"] == pytest.approx(10)
    assert probe(clips / "t03.mp4")["has_audio"] is False


def test_discover_skips_hidden(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "._a.mp4").write_bytes(b"")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in discover_videos(tmp_path)] == ["a.mp4"]


def test_no_files_written_to_repo(clip_videos):
    before = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root(), capture_output=True, text=True).stdout
    run(clip_videos["t01"])
    after = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root(), capture_output=True, text=True).stdout
    assert before == after


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg CLI needed to set a display rotation")
def test_rotated_phone_video_is_upright(clips, tmp_path):
    rotated = tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-display_rotation", "90", "-i", str(clips / "t01.mp4"),
                    "-c", "copy", str(rotated)], check=True)
    assert probe(rotated)["orientation"] == "portrait"
    with av.open(str(rotated)) as c:
        assert _to_rgb(next(c.decode(video=0))).shape[:2] == (160, 90)
