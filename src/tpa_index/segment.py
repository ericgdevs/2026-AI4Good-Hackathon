"""Split a video into fixed windows and sample frames and audio per window."""
import logging
from pathlib import Path
from typing import Iterator, NamedTuple

import av
import numpy as np

from tpa_index.paths import data_dir

log = logging.getLogger(__name__)
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}


class Window(NamedTuple):
    row: dict
    frames: np.ndarray  # (N, S, S, 3) uint8, letterboxed to square
    audio: np.ndarray  # float32 mono at the configured sample rate


def windows(duration_s: float, window_s: float, overlap_s: float) -> list[tuple[float, float]]:
    stride = window_s - overlap_s
    out: list[tuple[float, float]] = []
    i = 0
    while (start := i * stride) < duration_s:
        end = min(start + window_s, duration_s)
        if out and end - start < window_s / 2:
            out[-1] = (out[-1][0], duration_s)  # fold a short trailing window into the previous one
            break
        out.append((start, end))
        if end >= duration_s:
            break
        i += 1
    return out


def discover_videos(root=None) -> list[Path]:
    """Video files under DATA_DIR, skipping hidden files and folders (incl. macOS ._ files)."""
    root = Path(root or data_dir())
    return [p for p in sorted(root.rglob("*"))
            if p.suffix.lower() in VIDEO_EXTS and p.is_file()
            and not any(part.startswith(".") for part in p.relative_to(root).parts)]


def _to_rgb(frame: av.VideoFrame) -> np.ndarray:
    img = frame.to_ndarray(format="rgb24")
    k = round((frame.rotation or 0) / 90) % 4  # display-matrix rotation, counterclockwise degrees
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def probe(path) -> dict:
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        duration = float(vs.duration * vs.time_base) if vs.duration else (c.duration or 0) / 1e6
        rate = vs.average_rate or vs.guessed_rate
        first = next(c.decode(vs))
        h, w = _to_rgb(first).shape[:2]
        return {"duration_s": duration, "fps": float(rate) if rate else 0.0,
                "orientation": "portrait" if h > w else "landscape", "has_audio": bool(c.streams.audio)}


def _letterbox(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    out[top:top + h, left:left + w] = img
    return out


def _sample_frames(container, stream, start: float, end: float, n: int) -> np.ndarray:
    offset = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
    frames: list[np.ndarray | None] = []
    for i in range(n):
        target = offset + start + (i + 0.5) * (end - start) / n
        img = None
        try:
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                img = frame
                if frame.time >= target - 1e-6:
                    break
            frames.append(_to_rgb(img) if img is not None else None)
        except av.error.FFmpegError:
            frames.append(None)
    good = [i for i, f in enumerate(frames) if f is not None]
    if not good:
        raise RuntimeError("no decodable frames in window")
    frames = [f if f is not None else frames[min(good, key=lambda g: abs(g - i))] for i, f in enumerate(frames)]
    return np.stack([_letterbox(f) for f in frames])


def _load_audio(path, rate: int) -> np.ndarray | None:
    """Whole audio track as float32 mono at `rate`, or None if the video has no audio."""
    with av.open(str(path)) as c:
        if not c.streams.audio:
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks = [out.to_ndarray().reshape(-1) for frame in c.decode(audio=0) for out in resampler.resample(frame)]
        chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(None)]
    # shortcut: assumes audio starts at t=0; use the audio stream's start_time if A/V offsets show up
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


def segment_video(video: dict, settings: dict, sample_rate: int = 16000) -> Iterator[Window]:
    """Yield one Window per segment; frames and audio stay in memory."""
    path = data_dir() / video["filename"]
    info = probe(path)
    audio = _load_audio(path, sample_rate)
    if audio is None:
        log.info("video %s has no audio track; using silence", video["video_id"])
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        for start, end in windows(info["duration_s"], settings["window_s"], settings["overlap_s"]):
            n = int(round((end - start) * sample_rate))
            clip = np.zeros(n, np.float32)
            if audio is not None:
                part = audio[int(round(start * sample_rate)):][:n]
                clip[:len(part)] = part
            row = {"segment_id": f"{video['video_id']}_{round(start * 1000)}", "video_id": video["video_id"],
                   "start_s": float(start), "end_s": float(end), "person_count": None, "faces_detected": None,
                   "has_speech": None, "transcript_chars": None, "is_idle": None}
            yield Window(row, _sample_frames(c, stream, start, end, settings["frames_per_window"]), clip)
