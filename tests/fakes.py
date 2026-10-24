"""Deterministic stand-ins for model embedders (no downloads)."""
import zlib

import numpy as np


def _unit(v):
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


class FakeEmbedder:
    """Images point mostly along BASE (plus a little colour); texts are random unless steered."""

    def __init__(self, d=8, scale=10.0, bias=-5.0, name="fake"):
        self.d, self.name, self.logit_scale, self.logit_bias = d, name, scale, bias
        rng = np.random.default_rng(0)
        self.base = _unit(rng.normal(size=d))
        self.colour = rng.normal(size=(3, d)).astype(np.float32)
        self.steered: dict[str, np.ndarray] = {}
        self.seen_shapes: list[tuple] = []

    def steer(self, text, vec):
        self.steered[text.lower()] = _unit(np.asarray(vec, np.float32))

    def embed_texts(self, texts):
        out = []
        for t in texts:
            if t in self.steered:
                out.append(self.steered[t])
            else:
                out.append(_unit(np.random.default_rng(zlib.crc32(t.encode())).normal(size=self.d)))
        return np.stack(out)

    def embed_images(self, images):
        self.seen_shapes += [img.shape for img in images]
        mean_rgb = images.reshape(len(images), -1, 3).mean(axis=1) / 255.0
        return _unit(self.base + 0.05 * (mean_rgb @ self.colour))


class FakeMotion:
    """Fixed-score motion model stand-in."""

    def __init__(self, labels, score=0.7, name="fake-motion", frames_per_window=8):
        self.labels, self.score_value, self.name, self.frames_per_window = labels, score, name, frames_per_window

    def score(self, windows_frames):
        return np.full((len(windows_frames), len(self.labels)), self.score_value, np.float32)
