"""Process tab live preview, in memory only. Every frame is pixelated inside detected faces (padded) and over the head
area of each person box before it is shown; when blurring is unavailable or fails, no image is shown at all (fail
closed). An image leaves this module only as a base64 JPEG string for an HTML <img>: nothing is written to disk or
logged. Boxes are plain rectangles: no IDs, no numbers, no tracking."""
import base64
import threading
import time

import numpy as np

from tpa_index.config import load_theme

FACE_SCORE = 0.5  # looser than the expression stage (0.8): a missed face here would be shown
FACE_PAD = 0.25  # each side, as a share of the face box
HEAD_FRAC = 0.22  # top share of a person box pixelated as well, for faces the detector misses
BLOCKS = 6  # pixelation blocks across the longer side of a region
HEIGHT = 360  # preview height in pixels
SOUNDS = ("singing_voice", "music", "speech", "applause")
HIDDEN = "Preview image hidden: face blurring is not available on this laptop."


def pixelate(img: np.ndarray, box) -> None:
    """Replace a region (x0, y0, x1, y1 in pixels, clipped to the image) with coarse blocks, in place."""
    import cv2

    h, w = img.shape[:2]
    x0, y0 = max(int(box[0]), 0), max(int(box[1]), 0)
    x1, y1 = min(int(np.ceil(box[2])), w), min(int(np.ceil(box[3])), h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return
    region = img[y0:y1, x0:x1]
    side = max(x1 - x0, y1 - y0)
    small = cv2.resize(region, (max(1, round(BLOCKS * (x1 - x0) / side)), max(1, round(BLOCKS * (y1 - y0) / side))),
                       interpolation=cv2.INTER_AREA)
    img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


class FaceBlur:
    """YuNet (the expression stage's pinned file) with looser settings, used only to hide faces in the preview."""

    def __init__(self, detector, max_side: int = 1280):
        self.detector, self.max_side = detector, max_side

    @classmethod
    def load(cls, pipeline: dict) -> "FaceBlur":
        import cv2

        from tpa_index.signal_models import yunet_file

        path = yunet_file(pipeline["models"]["expression"]["face_detector"])
        detector = cv2.FaceDetectorYN.create(path, "", (320, 320), FACE_SCORE)
        return cls(detector, pipeline["signals"]["expression"]["detect_max_side"])

    def apply(self, frame: np.ndarray, person_boxes=None) -> np.ndarray | None:
        """A pixelated copy of an RGB frame, or None if anything fails (the caller then shows no image)."""
        from tpa_index.signal_models import detect_faces, usable_faces

        try:
            faces = usable_faces(detect_faces(self.detector, frame, self.max_side), frame.shape, FACE_SCORE, 1)
            out = np.array(frame, copy=True)
            h, w = out.shape[:2]
            for x0, y0, x1, y1 in faces:
                px, py = (x1 - x0) * FACE_PAD, (y1 - y0) * FACE_PAD
                pixelate(out, (x0 - px, y0 - py, x1 + px, y1 + py))
            for x0, y0, x1, y1 in person_boxes or []:
                pixelate(out, (x0 * w, y0 * h, x1 * w, (y0 + HEAD_FRAC * (y1 - y0)) * h))
            return out
        except Exception:  # noqa: BLE001 - fail closed: no blur, no image
            return None


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def to_data_url(img: np.ndarray, person_boxes=None, box_colour: str | None = None) -> str:
    """Plain person rectangles on an already blurred RGB frame, scaled to HEIGHT, as a JPEG data URL (memory only)."""
    import cv2

    h, w = img.shape[:2]
    scale = min(1.0, HEIGHT / h)
    small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    colour = _rgb(box_colour or load_theme()["accent"])[::-1]  # theme accent unless the caller picks one
    for x0, y0, x1, y1 in person_boxes or []:
        cv2.rectangle(bgr, (round(x0 * sw), round(y0 * sh)), (round(x1 * sw), round(y1 * sh)), colour, 2)
    ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")


def _safe(fn):
    """Preview callbacks run inside the pipeline: they must never stop it."""
    def wrapper(*a, **k):
        try:
            fn(*a, **k)
        except Exception:  # noqa: BLE001
            pass
    return wrapper


class Live:
    """Callbacks for predict_video and run_signals, and the latest values the Process tab shows. Thread-safe. Keeps
    only the newest preview image (a string) and per-clip numbers; frames are dropped as soon as they are encoded."""

    def __init__(self, blur: FaceBlur | None, interval: float = 1.0, clock=time.monotonic, box_colour=None):
        self.blur, self.interval, self.clock, self.box_colour = blur, interval, clock, box_colour
        self._lock = threading.Lock()
        self._last = float("-inf")
        self.info: dict[str, dict] = {}
        self.counts = {"predict": 0, "frames": 0}
        self.image, self.shown = None, None  # data URL, {"segment_id", "start_s", "end_s", "people_boxes"}

    def _seg(self, sid):
        return self.info.setdefault(sid, {"activity": None, "others": [], "sounds": None, "speech": None,
                                          "expression": None})

    @_safe
    def window(self, seg: dict, frames: np.ndarray, activity_scores: dict[str, float]) -> None:
        ranked = sorted(activity_scores.items(), key=lambda kv: -kv[1])
        with self._lock:
            self.counts["predict"] += 1
            if ranked:
                info = self._seg(seg["segment_id"])
                top = ranked[0][1]
                info["activity"] = ranked[0][0]
                info["others"] = [(lab, s / top) for lab, s in ranked[1:3]] if top > 0 else []
        self._show(seg, frames, None)

    @_safe
    def audio(self, segment_id: str, scores: dict[str, float]) -> None:
        with self._lock:
            self._seg(segment_id)["sounds"] = {k: float(scores[k]) for k in SOUNDS if k in scores}

    @_safe
    def speech(self, segment_id: str, has_speech) -> None:
        with self._lock:
            self._seg(segment_id)["speech"] = None if has_speech is None else bool(has_speech)

    @_safe
    def frames(self, seg: dict, frames: np.ndarray, boxes, expression) -> None:
        """One clip of the people/expression pass. expression: (faces, {label: score} | None) or None."""
        with self._lock:
            self.counts["frames"] += 1
            if expression is not None:
                faces, scores = expression
                self._seg(seg["segment_id"])["expression"] = (
                    "no faces" if not faces else max(scores, key=scores.get) if scores else None)
        self._show(seg, frames, boxes)

    def _show(self, seg, frames, boxes) -> None:
        now = self.clock()
        with self._lock:
            if now - self._last < self.interval:
                return
            self._last = now
        i = len(frames) // 2
        frame_boxes = boxes[i] if boxes else []
        blurred = self.blur.apply(frames[i], frame_boxes) if self.blur is not None else None
        url = None if blurred is None else to_data_url(blurred, frame_boxes, self.box_colour)
        del blurred
        with self._lock:
            self.image = url
            self.shown = {"segment_id": seg["segment_id"], "start_s": seg["start_s"], "end_s": seg["end_s"],
                          "people_boxes": boxes is not None}

    def view(self) -> dict:
        """What the panel shows: the latest image (or why there is none) and the shown clip's values."""
        with self._lock:
            info = dict(self.info.get(self.shown["segment_id"], {})) if self.shown else {}
            return {"image": self.image, "hidden": None if self.image or not self.shown else HIDDEN,
                    "shown": dict(self.shown) if self.shown else None, "counts": dict(self.counts), **info}
