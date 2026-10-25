"""006: `tpa-index render-overlay`: a side-by-side "models at work" MP4 of one stretch of one video, for an in-room
slide. Laptop models, this machine only.

Privacy: every frame is blurred with the Process preview's blur (padded YuNet faces plus the top of each person box)
before anything else uses it, and only blurred frames go, from memory, to the encoder. If blurring fails for a frame,
that frame is plain grey ("blur unavailable"): the raw frame is never written. No frame images, crops, audio or other
temp files are written, no run is registered (local_runs.yaml and official results are untouched), nothing is logged,
and transcript text is dropped as soon as speech yes/no is known. The output must be outside the repo."""
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tpa_index.config import ConfigError, load_labels, load_pipeline, load_theme
from tpa_index.paths import GuardError, repo_root

FPS, WIDTH, HEIGHT = 12, 1920, 1080
PAD, GAP, TOP = 40, 40, 100  # outer margin, gap between panels, top of the panel images
PANEL_W = (WIDTH - 2 * PAD - GAP) // 2
IMG_H = round(PANEL_W * 9 / 16)
SOUND_NAMES = {"singing_voice": "Singing", "music": "Music", "speech": "Speech", "applause": "Applause"}
BLUR_UNAVAILABLE = "blur unavailable"
NA = "not available"


def parse_time(text: str) -> float:
    """`MM:SS`, `H:MM:SS` or plain seconds -> seconds."""
    if not re.fullmatch(r"\d+(\.\d+)?|(\d+:)?\d{1,2}:\d{2}(\.\d+)?", text.strip()):
        raise ConfigError(f"--start must look like MM:SS (got {text!r})")
    secs = 0.0
    for part in text.strip().split(":"):
        secs = secs * 60 + float(part)
    return secs


def output_path(out, video_id: str, start_s: float, duration_s: float) -> Path:
    """The MP4 to write: --out, or OUTPUT_DIR/renders/. Never inside the repo."""
    from tpa_index.paths import output_dir

    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        mm, ss = divmod(int(start_s), 60)
        path = output_dir() / "renders" / f"overlay-{video_id}-{mm:02d}{ss:02d}-{int(duration_s)}s-{stamp}.mp4"
    else:
        path = Path(out).expanduser().resolve()
    if path.resolve().is_relative_to(repo_root()):
        raise GuardError("--out must be outside the repo (renders hold footage)")
    if path.suffix.lower() != ".mp4":
        raise ConfigError("--out must end in .mp4")
    return path


def has_letters(font) -> bool:
    """False for a font that would draw boxes: some bundled web fonts are subsets without basic Latin letters."""
    return font.getmask("V").getbbox() != font.getmask("W").getbbox()


def _font(size: int, bold: bool = False):
    """IBM Plex Sans from Gradio's bundled fonts when it can be read and has the letters, else Pillow's own font."""
    from PIL import ImageFont

    try:
        import gradio

        folder = Path(gradio.__file__).parent / "templates/frontend/static/fonts/IBMPlexSans"
        for name in (["IBMPlexSans-Bold.woff2"] if bold else []) + ["IBMPlexSans-Regular.woff2"]:
            font = ImageFont.truetype(str(folder / name), size)
            if has_letters(font):
                return font
    except Exception:  # noqa: BLE001 - a plain font is fine
        pass
    return ImageFont.load_default(size)


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---- models (laptop sizes; each optional except the blur) ----
def load_models(pipeline: dict, labels: dict, dev: str, make_embedder=None, make_models=None, make_blur=None) -> dict:
    """The blur is required (refuse rather than risk an unblurred frame); a model that fails to load shows as
    "not available" on the right panel."""
    from tpa_index import cli
    from tpa_index.preview import FaceBlur

    try:
        blur = (make_blur or FaceBlur.load)(pipeline)
    except Exception:  # noqa: BLE001
        raise ConfigError("face blurring is not available on this computer; nothing was rendered") from None
    models = {"blur": blur}
    try:
        models["activity"] = (make_embedder or cli._make_embedder)(
            pipeline["models"]["main"]["hpc" if dev == "cuda" else "mac"], dev)
    except Exception:  # noqa: BLE001
        models["activity"] = None
    for stage in ("transcript", "audio", "people", "expression"):
        try:
            models[stage] = (make_models or cli._make_signal_models)([stage], pipeline, dev)[stage]
        except Exception:  # noqa: BLE001
            models[stage] = None
    return models


# ---- per 8-second clip: activity, sounds, speech yes/no, expression ----
def clip_info(video: dict, path: Path, clips: list[tuple[float, float]], pipeline: dict, labels: dict,
              models: dict) -> list[dict]:
    """One dict per clip: activity (top label and the next two as ratios), sounds, speech, expression. Each part is
    None when its model is missing or fails. Frames, audio and transcript stay in memory."""
    import av

    from tpa_index.config import effective_settings, model_labels
    from tpa_index.score import label_embeddings, sigmoid_scores, window_embedding
    from tpa_index.segment import _load_audio, _sample_frames
    from tpa_index.signals import _score_segment_faces, aggregate_expression, assign_words

    info = [{"start_s": a, "end_s": b, "activity": None, "others": [], "sounds": None, "speech": None,
             "expression": None} for a, b in clips]
    n_frames = effective_settings(pipeline, video)["frames_per_window"]
    embedder, scorer = models.get("activity"), models.get("expression")
    names, label_matrix = [], None
    if embedder is not None:
        try:
            primary = model_labels(labels, {pipeline["moments"]["primary_group"]})
            emb = label_embeddings(embedder, primary)
            names = [lab["name"] for lab in primary]
            label_matrix = np.stack([emb[n] for n in names])
        except Exception:  # noqa: BLE001
            embedder = None
    if embedder is not None or scorer is not None:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for item in info:
                try:
                    frames = _sample_frames(container, stream, item["start_s"], item["end_s"], n_frames)
                except Exception:  # noqa: BLE001
                    continue
                if embedder is not None:
                    try:
                        window = window_embedding(embedder.embed_images(frames))[None, :]
                        scores = sigmoid_scores(window, label_matrix, embedder.logit_scale, embedder.logit_bias)[0]
                        ranked = sorted(zip(names, map(float, scores)), key=lambda kv: -kv[1])
                        item["activity"] = ranked[0][0]
                        top = ranked[0][1]
                        item["others"] = [(lab, s / top) for lab, s in ranked[1:3]] if top > 0 else []
                    except Exception:  # noqa: BLE001
                        pass
                if scorer is not None:
                    try:
                        faces, scores = aggregate_expression(_score_segment_faces(scorer, frames), scorer.labels)
                        item["expression"] = "no faces" if not faces else max(scores, key=scores.get) if scores else None
                    except Exception:  # noqa: BLE001
                        pass
                del frames

    tagger, transcriber = models.get("audio"), models.get("transcript")
    if tagger is None and transcriber is None:
        return info
    rate = pipeline["audio_sample_rate"]
    try:
        audio = _load_audio(path, rate)
    except Exception:  # noqa: BLE001
        return info
    if audio is None:
        audio = np.zeros(int(round(clips[-1][1] * rate)), np.float32)
    if tagger is not None:
        try:
            batch = []
            for a, _ in clips:
                clip = np.zeros(tagger.input_samples, np.float32)
                part = audio[int(round(a * rate)):][:tagger.input_samples]
                clip[:len(part)] = part
                batch.append(clip)
            for item, row in zip(info, tagger.score(batch)):
                item["sounds"] = {k: float(np.clip(v, 0, 1)) for k, v in zip(tagger.labels, row) if k in SOUND_NAMES}
        except Exception:  # noqa: BLE001
            pass
    if transcriber is not None:
        try:
            offset = clips[0][0]
            stretch = audio[int(round(offset * rate)):int(round(clips[-1][1] * rate))]
            # keep word timings only; the text is dropped right here and never stored, drawn or written
            words = [{"start": w["start"] + offset, "end": w["end"] + offset, "word": "w"}
                     for s in transcriber.transcribe_audio(stretch) for w in s.get("words") or []]
            segs = [{"segment_id": str(i), "start_s": it["start_s"], "end_s": it["end_s"]} for i, it in enumerate(info)]
            for i, (speech, _) in assign_words(words, segs).items():
                info[int(i)]["speech"] = speech
        except Exception:  # noqa: BLE001
            pass
    del audio
    return info


# ---- drawing ----
class Painter:
    """Lays out one 1920x1080 frame: blurred video left, the same blurred frame with person boxes and the current
    clip's AI notes right. Colours from config/theme.yaml."""

    def __init__(self, labels: dict):
        self.labels, self.theme = labels, load_theme()
        self.h1, self.body, self.small, self.bold = _font(34, True), _font(26), _font(21), _font(26, True)

    def _fit(self, img: np.ndarray) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        scale = min(PANEL_W / w, IMG_H / h)
        return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)

    def grey(self) -> np.ndarray:
        """The fail-closed frame: plain grey with the reason, nothing from the video."""
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", (WIDTH, HEIGHT), _rgb(self.theme["muted"]))
        draw = ImageDraw.Draw(canvas)
        draw.text((WIDTH // 2, HEIGHT // 2), BLUR_UNAVAILABLE, fill=_rgb(self.theme["text"]), font=self.h1,
                  anchor="mm")
        return np.asarray(canvas)

    def frame(self, blurred: np.ndarray, boxes, item: dict | None) -> np.ndarray:
        import cv2
        from PIL import Image, ImageDraw

        from tpa_index.demo import display_label

        t = self.theme
        canvas = Image.new("RGB", (WIDTH, HEIGHT), _rgb(t["background"]))
        draw = ImageDraw.Draw(canvas)
        left_x, right_x = PAD, PAD + PANEL_W + GAP
        for x, title in ((left_x, "Video"), (right_x, "What the AI notices")):
            draw.text((x, TOP - 54), title, fill=_rgb(t["accent_deep"]), font=self.h1)
            draw.rectangle((x, TOP, x + PANEL_W - 1, TOP + IMG_H - 1), fill=_rgb(t["text"]))
        small = self._fit(blurred)
        h, w = small.shape[:2]
        oy = TOP + (IMG_H - h) // 2
        canvas.paste(Image.fromarray(small), (left_x + (PANEL_W - w) // 2, oy))
        boxed = np.array(small, copy=True)
        for x0, y0, x1, y1 in boxes or []:
            cv2.rectangle(boxed, (round(x0 * w), round(y0 * h)), (round(x1 * w), round(y1 * h)), _rgb(t["accent"]), 3)
        canvas.paste(Image.fromarray(boxed), (right_x + (PANEL_W - w) // 2, oy))
        del small, boxed

        below = TOP + IMG_H + 24
        draw.text((left_x, below), "Faces blurred · nothing saved", fill=_rgb(t["text_secondary"]), font=self.small)
        n = len(boxes or [])
        draw.text((right_x, below), f"person detection (counts only): {n} {'person' if n == 1 else 'people'}",
                  fill=_rgb(t["text_secondary"]), font=self.small)
        self._notes(draw, right_x, below + 44, item, display_label)
        return np.asarray(canvas)

    def _notes(self, draw, x, y, item, display_label) -> None:
        t, name = self.theme, (lambda lab: display_label(lab, self.labels))  # noqa: E731
        item = item or {}
        act = item.get("activity")
        draw.text((x, y), "AI suggestion (uncalibrated): " + (name(act) if act else NA), fill=_rgb(t["text"]),
                  font=self.bold)
        others = " · ".join(f"{name(lab)} {r:.2f}×" for lab, r in item.get("others") or [])
        if others:
            draw.text((x, y + 36), "next: " + others, fill=_rgb(t["text_secondary"]), font=self.small)
        y += 84
        sounds = item.get("sounds")
        for i, (key, label) in enumerate(SOUND_NAMES.items()):
            row = y + i * 34
            draw.text((x, row), label, fill=_rgb(t["text"]), font=self.small)
            draw.rectangle((x + 130, row + 6, x + 130 + 300, row + 24), fill=_rgb(t["track"]))
            if sounds is not None and key in sounds:
                draw.rectangle((x + 130, row + 6, x + 130 + round(300 * sounds[key]), row + 24),
                               fill=_rgb(t["chart_series"][1]))
        if sounds is None:
            draw.text((x + 450, y), "sounds " + NA, fill=_rgb(t["text_secondary"]), font=self.small)
        y += 4 * 34 + 16
        speech = item.get("speech")
        draw.text((x, y), "Speech: " + (NA if speech is None else "yes" if speech else "no"), fill=_rgb(t["text"]),
                  font=self.body)
        expr = item.get("expression")
        draw.text((x, y + 40), "Expression (unvalidated): " + (NA if expr is None else expr if expr == "no faces"
                                                              else name(expr)), fill=_rgb(t["text"]), font=self.body)
        if item:
            mmss = lambda s: f"{int(s // 60)}:{int(s % 60):02d}"  # noqa: E731
            draw.text((x, y + 84), f"8-second clip {mmss(item['start_s'])}–{mmss(item['end_s'])}",
                      fill=_rgb(t["text_secondary"]), font=self.small)


def _decode(path: Path, start_s: float, n: int):
    """RGB frames at FPS from start_s (a frame repeats when the source is slower), straight from the decoder."""
    import av

    from tpa_index.segment import _to_rgb

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        offset = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
        container.seek(int((offset + start_s) / stream.time_base), stream=stream, backward=True)
        k, last = 0, None
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = frame.time - offset
            while k < n and t >= start_s + k / FPS:
                yield start_s + k / FPS, _to_rgb(frame)
                k += 1
            last = frame
            if k >= n:
                return
        while k < n and last is not None:  # the stretch ends after the last decodable frame
            yield start_s + k / FPS, _to_rgb(last)
            k += 1


def render_overlay(video_id: str, start_s: float, duration_s: float = 60, out=None, make_embedder=None,
                   make_models=None, make_blur=None, batch: int = FPS) -> tuple[Path, float]:
    """Write the MP4 and return (path, seconds rendered). Nothing else is written."""
    import av

    from tpa_index.config import effective_settings, load_videos
    from tpa_index.paths import data_dir
    from tpa_index.runs import device
    from tpa_index.segment import probe, windows

    video = next((v for v in load_videos() if v["video_id"] == video_id), None)
    if video is None:
        raise ConfigError(f"unknown video_id: {video_id}")
    if duration_s <= 0:
        raise ConfigError("--duration must be more than 0 seconds")
    dest = output_path(out, video_id, start_s, duration_s)
    path = data_dir() / video["filename"]
    total = probe(path)["duration_s"]
    if start_s >= total:
        raise ConfigError(f"--start is past the end of the video ({int(total // 60)}:{int(total % 60):02d})")
    duration_s = min(duration_s, total - start_s)
    n = max(1, round(duration_s * FPS))

    pipeline, labels = load_pipeline(), load_labels()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # laptop weights from the local cache only
    os.environ["YOLO_OFFLINE"] = "1"
    models = load_models(pipeline, labels, device(), make_embedder, make_models, make_blur)
    s = effective_settings(pipeline, video)
    end_s = start_s + duration_s
    clips = [(a, b) for a, b in windows(total, s["window_s"], s["overlap_s"]) if a < end_s and b > start_s]
    info = clip_info(video, path, clips, pipeline, labels, models)
    painter = Painter(labels)

    def current(t):  # the latest 8-second clip that has started by time t
        return next((it for it in reversed(info) if it["start_s"] <= t), info[0] if info else None)

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.stem + ".partial.mp4")
    try:
        with av.open(str(partial), "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=FPS)  # video only: no audio stream is ever added
            stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
            stream.options = {"crf": "20", "preset": "medium"}

            def emit(rgb):
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                    container.mux(packet)

            pending: list[tuple[float, np.ndarray]] = []

            def flush():
                people = models.get("people")
                try:
                    boxes = people.boxes(np.stack([f for _, f in pending])) if people is not None else None
                except Exception:  # noqa: BLE001 - no boxes; faces are still blurred
                    boxes = None
                for i, (t, frame) in enumerate(pending):
                    frame_boxes = boxes[i] if boxes else []
                    blurred = models["blur"].apply(frame, frame_boxes)
                    emit(painter.grey() if blurred is None else painter.frame(blurred, frame_boxes, current(t)))
                    del blurred
                pending.clear()

            for t, frame in _decode(path, start_s, n):
                pending.append((t, frame))
                if len(pending) >= batch:
                    flush()
            if pending:
                flush()
            for packet in stream.encode():
                container.mux(packet)
        os.replace(partial, dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return dest, n / FPS
