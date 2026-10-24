"""On-demand playback clips. The only module that touches footage: cuts go to one session temp folder outside
the repo, which is removed on exit."""
import atexit
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from tpa_index.paths import data_dir, redact, repo_root

PREFIX = "tpa-clips-"


def session_dir() -> Path:
    """A fresh temp folder for this app session; leftovers older than 12 h (crashed sessions) are removed."""
    root = Path(tempfile.gettempdir())
    for old in root.glob(PREFIX + "*"):
        try:
            if old.is_dir() and time.time() - old.stat().st_mtime > 12 * 3600:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    path = Path(tempfile.mkdtemp(prefix=PREFIX)).resolve()
    if path.is_relative_to(repo_root()):
        raise RuntimeError("clip folder must be outside the repo")
    return path


class ClipCutter:
    def __init__(self, session: Path, clip_height: int = 480):
        self.session, self.clip_height = Path(session), clip_height
        self._current: Path | None = None
        self._cache: dict[tuple, Path] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _cut_to(self, video: dict, start: float, end: float) -> Path:
        src = data_dir() / video["filename"]
        out = self.session / f"{uuid.uuid4().hex}.mp4"
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src),
               "-t", f"{end - start:.3f}", "-vf", f"scale=-2:{self.clip_height}", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "26", "-c:a", "aac", "-movflags", "+faststart", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or not out.exists():
            out.unlink(missing_ok=True)
            raise RuntimeError(redact("clip could not be cut: " + (result.stderr.strip().splitlines() or ["?"])[-1]))
        return out

    @staticmethod
    def _bounds(video: dict, start_s: float, end_s: float, pad_s: float) -> tuple[float, float]:
        return max(0.0, start_s - pad_s), min(float(video["duration_s"]), end_s + pad_s)

    def cut(self, video: dict, start_s: float, end_s: float, pad_s: float = 0) -> Path:
        """Clip of [start - pad, end + pad] clipped to the video; the previously shown clip is deleted."""
        start, end = self._bounds(video, start_s, end_s, pad_s)
        key = (video["video_id"], round(start, 3), round(end, 3))
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            path = self._cache.pop(key, None)
        path = path if path is not None and path.exists() else self._cut_to(video, start, end)
        if self._current is not None and self._current != path:
            self._current.unlink(missing_ok=True)
        self._current = path
        return path

    def prefetch(self, video: dict, start_s: float, end_s: float, pad_s: float = 0) -> None:
        """Cut the next clip in the background (one at a time); errors surface when it is shown."""
        start, end = self._bounds(video, start_s, end_s, pad_s)
        key = (video["video_id"], round(start, 3), round(end, 3))

        def work():
            try:
                path = self._cut_to(video, start, end)
            except Exception:
                return
            with self._lock:
                for stale in self._cache.values():
                    stale.unlink(missing_ok=True)
                self._cache = {key: path}

        if self._thread is not None:
            self._thread.join()
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def cleanup(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=30)
        shutil.rmtree(self.session, ignore_errors=True)


def register_cleanup(cutter: ClipCutter) -> None:
    """Remove the clip folder on normal exit and on SIGINT/SIGTERM."""
    atexit.register(cutter.cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(sig)

        def handler(signum, frame, previous=previous):
            cutter.cleanup()
            if callable(previous):
                previous(signum, frame)
            else:
                raise SystemExit(128 + signum)

        signal.signal(sig, handler)
