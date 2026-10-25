"""006: `tpa-index render-overlay`: a side-by-side "models at work" MP4 of one stretch of one video, for an in-room
slide. Laptop models, this machine only.

Privacy: every frame is blurred with the Process preview's blur (padded YuNet faces plus the top of each person box)
before anything else uses it, and only blurred frames go, from memory, to the encoder. Face blurring and person
detection are both required. If either fails for a frame, that frame is plain grey ("blur unavailable"): the raw frame
is never written. No frame images, crops, audio or other
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
PAD, GAP, TOP = 40, 40, 96  # outer margin, gap between panels, top of the panel images
PANEL_W = (WIDTH - 2 * PAD - GAP) // 2
IMG_H = 440  # leaves room under the image for the three sections at sizes readable from the back of a room
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
    return bytes(font.getmask("V")) != bytes(font.getmask("W"))  # a missing glyph draws the same box for both


def _font(size: int, bold: bool = False, family: str = "IBMPlexSans"):
    """IBM Plex (Sans or Mono) from Gradio's bundled fonts when it can be read and has the letters, else Pillow's
    own font."""
    from PIL import ImageFont

    try:
        import gradio

        folder = Path(gradio.__file__).parent / "templates/frontend/static/fonts" / family
        for name in ([f"{family}-Bold.woff2"] if bold else []) + [f"{family}-Regular.woff2"]:
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
    """The blur and the person detector (its head-area cover catches faces YuNet misses) are required: refuse rather
    than risk an unblurred face. Any other model that fails to load shows as "not available" on the right panel."""
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
    try:
        models["people"] = (make_models or cli._make_signal_models)(["people"], pipeline, dev)["people"]
    except Exception:  # noqa: BLE001
        raise ConfigError("person detection is not available on this computer; nothing was rendered") from None
    for stage in ("transcript", "audio", "expression"):
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
def _mix(a: str, b: str, f: float) -> tuple[int, int, int]:
    """Theme colour a moved a share f towards theme colour b (a lighter accent without a new colour)."""
    return tuple(round(x + (y - x) * f) for x, y in zip(_rgb(a), _rgb(b)))


def _mmss(secs: float) -> str:
    return f"{int(secs // 60)}:{int(secs % 60):02d}"


class Painter:
    """Lays out one 1920x1080 frame: blurred video left; right, the same blurred frame with plain person boxes and
    the current clip's AI notes in three sections. Colours from config/theme.yaml, IBM Plex."""

    BAR_W = 420  # activity bar length

    def __init__(self, labels: dict):
        self.labels, self.theme = labels, load_theme()
        self.h1, self.mono = _font(34, True), _font(26, family="IBMPlexMono")
        self.label, self.label_b, self.head = _font(26), _font(26, True), _font(19)
        self.value, self.note = _font(32, True), _font(19)

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

        t = self.theme
        canvas = Image.new("RGB", (WIDTH, HEIGHT), _rgb(t["background"]))
        draw = ImageDraw.Draw(canvas)
        left_x, right_x = PAD, PAD + PANEL_W + GAP
        for x, title in ((left_x, "Video"), (right_x, "What the AI notices")):
            draw.text((x, TOP - 18), title, fill=_rgb(t["accent_deep"]), font=self.h1, anchor="ls")
            draw.rectangle((x, TOP, x + PANEL_W - 1, TOP + IMG_H - 1), fill=_rgb(t["text"]))
        if item:
            draw.text((right_x + PANEL_W, TOP - 18), f"clip {_mmss(item['start_s'])}–{_mmss(item['end_s'])}",
                      fill=_rgb(t["text_secondary"]), font=self.mono, anchor="rs")
        small = self._fit(blurred)
        h, w = small.shape[:2]
        oy = TOP + (IMG_H - h) // 2
        canvas.paste(Image.fromarray(small), (left_x + (PANEL_W - w) // 2, oy))
        boxed = np.array(small, copy=True)
        for x0, y0, x1, y1 in boxes or []:  # plain rectangles: no IDs, no per-person labels
            cv2.rectangle(boxed, (round(x0 * w), round(y0 * h)), (round(x1 * w), round(y1 * h)), _rgb(t["accent"]), 3)
        canvas.paste(Image.fromarray(boxed), (right_x + (PANEL_W - w) // 2, oy))
        del small, boxed
        draw.text((left_x, TOP + IMG_H + 22), "Faces blurred · nothing saved", fill=_rgb(t["text_secondary"]),
                  font=self.head)
        self._notes(draw, right_x, TOP + IMG_H + 20, item or {}, len(boxes or []))
        return np.asarray(canvas)

    def _section(self, draw, x: int, y: int, title: str) -> int:
        """A thin divider and a small grey heading; returns where the content starts."""
        draw.line((x, y, x + PANEL_W, y), fill=_rgb(self.theme["border"]), width=2)
        draw.text((x, y + 12), title, fill=_rgb(self.theme["text_secondary"]), font=self.head)
        return y + 48

    def _notes(self, draw, x: int, y: int, item: dict, n_people: int) -> None:
        from tpa_index.demo import display_label

        t = self.theme
        name = lambda lab: display_label(lab, self.labels)  # noqa: E731
        grey, text = _rgb(t["text_secondary"]), _rgb(t["text"])

        # 1. activity: the top suggestion and the next two, as bars relative to the top (never percentages)
        y = self._section(draw, x, y, "Activity (SigLIP2 · uncalibrated)")
        bar_x = x + 330
        if item.get("activity") is None:
            draw.text((x, y), NA, fill=grey, font=self.label)
        else:
            rows = [(item["activity"], 1.0, True)] + [(lab, r, False) for lab, r in item.get("others") or []]
            for i, (lab, ratio, top) in enumerate(rows[:3]):
                row = y + i * 40
                draw.text((x, row + 15), name(lab), fill=text if top else grey, font=self.label_b if top else self.label,
                          anchor="lm")
                draw.rectangle((bar_x, row + 5, bar_x + self.BAR_W, row + 25), fill=_rgb(t["track"]))
                colour = _rgb(t["accent"]) if top else _mix(t["accent"], t["surface"], 0.55)
                draw.rectangle((bar_x, row + 5, bar_x + round(self.BAR_W * max(0.0, min(ratio, 1.0))), row + 25),
                               fill=colour)
                draw.text((x + PANEL_W, row + 15), "top" if top else f"{ratio:.2f}×", fill=text if top else grey,
                          font=self.label_b if top else self.label, anchor="rm")
        y += 3 * 40 + 8

        # 2. sounds: four bars, labels in one column
        y = self._section(draw, x, y, "Sounds (AST)")
        sounds, sx = item.get("sounds"), x + 170
        for i, (key, label) in enumerate(SOUND_NAMES.items()):
            row = y + i * 36
            draw.text((x, row + 14), label, fill=text, font=self.label, anchor="lm")
            draw.rectangle((sx, row + 5, x + PANEL_W, row + 23), fill=_rgb(t["track"]))
            if sounds is not None and key in sounds:
                draw.rectangle((sx, row + 5, sx + round((x + PANEL_W - sx) * sounds[key]), row + 23),
                               fill=_rgb(t["chart_series"][1]))
        if sounds is None:
            draw.text((x + PANEL_W, y - 36), NA, fill=grey, font=self.head, anchor="ra")
        y += 4 * 36 + 8

        # 3. three equal blocks: speech present, people, expression
        draw.line((x, y, x + PANEL_W, y), fill=_rgb(t["border"]), width=2)
        speech, expr = item.get("speech"), item.get("expression")
        blocks = [("Speech present", "Whisper · no text", NA if speech is None else "Yes" if speech else "No", None),
                  ("People", "YOLO11 · counts only", f"{n_people} {'person' if n_people == 1 else 'people'}", None),
                  ("Expression", "EmotiEffLib", NA if expr is None else expr if expr == "no faces" else name(expr),
                   "unvalidated")]
        width = PANEL_W // 3
        for i, (title, model, value, note) in enumerate(blocks):
            bx = x + i * width
            draw.text((bx, y + 12), title, fill=grey, font=self.head)
            draw.text((bx, y + 34), model, fill=grey, font=self.note)
            draw.text((bx, y + 64), value, fill=text, font=self.value)
            if note:
                draw.text((bx, y + 104), note, fill=grey, font=self.note)


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
                try:
                    boxes = models["people"].boxes(np.stack([f for _, f in pending]))
                except Exception:  # noqa: BLE001 - no head-area cover: these frames go out grey
                    boxes = None
                for i, (t, frame) in enumerate(pending):
                    frame_boxes = boxes[i] if boxes is not None else None
                    blurred = None if frame_boxes is None else models["blur"].apply(frame, frame_boxes)
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
