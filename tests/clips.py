"""Tiny synthetic test clips (no real footage)."""
from fractions import Fraction

import av
import numpy as np

COLORS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30), (30, 200, 200)]


def make_clip(path, w, h, seconds=20, fps=10, audio=True, rate=16000, static=False):
    """Solid colour that changes every second (one constant colour if static), 4 px white border,
    optional 440 Hz mono AAC track."""
    with av.open(str(path), "w") as out:
        video = out.add_stream("mpeg4", rate=fps)
        video.width, video.height, video.pix_fmt = w, h, "yuv420p"
        sound = out.add_stream("aac", rate=rate, layout="mono") if audio else None
        for i in range(seconds * fps):
            img = np.full((h, w, 3), COLORS[0 if static else (i // fps) % len(COLORS)], np.uint8)
            img[:4], img[-4:], img[:, :4], img[:, -4:] = 255, 255, 255, 255
            for packet in video.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        if sound:
            t = np.arange(seconds * rate) / rate
            wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            for start in range(0, len(wave), 1024):
                frame = av.AudioFrame.from_ndarray(wave[None, start:start + 1024], format="fltp", layout="mono")
                frame.sample_rate, frame.pts, frame.time_base = rate, start, Fraction(1, rate)
                for packet in sound.encode(frame):
                    out.mux(packet)
            for packet in sound.encode():
                out.mux(packet)
    return path
