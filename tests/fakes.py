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


# ---- 004 signal stages: same Protocols as tpa_index.signal_models, no weights ----------------------------------
class FakeTranscriber:
    """Words every 2 s (text = `sentinel` if given). Fails for videos in `fail_for` (matched on the WAV name)."""

    def __init__(self, words=None, sentinel=None, fail_for=(), raise_with="transcriber broke", name="fake-whisper"):
        self.name, self.calls = name, []
        self.words = words if words is not None else [(t, t + 0.4, sentinel or "la") for t in (1.0, 3.0, 9.0)]
        self.fail_for, self.raise_with = set(fail_for), raise_with

    def transcribe(self, wav_path):
        self.calls.append(str(wav_path))
        if any(f"-{v}.wav" in str(wav_path) for v in self.fail_for):
            raise RuntimeError(self.raise_with)
        return self._segments()

    def transcribe_audio(self, audio):
        self.calls.append(f"<{len(audio)} samples in memory>")
        return self._segments()

    def _segments(self):
        ws = [{"start": a, "end": b, "word": f" {w}"} for a, b, w in self.words]
        return [{"start": ws[0]["start"], "end": ws[-1]["end"], "text": "".join(w["word"] for w in ws),
                 "words": ws}] if ws else []


class FakeAudioTagger:
    """Constant scores; small input so tests stay light. `fail_on_call` raises on that call (1-based)."""

    def __init__(self, labels, value=0.6, input_samples=32000, fail_on_call=None, name="fake-ast"):
        self.labels, self.value, self.input_samples, self.name = list(labels), value, input_samples, name
        self.fail_on_call, self.calls, self.shapes = fail_on_call, 0, []

    def score(self, clips):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("tagger broke")
        self.shapes += [c.shape for c in clips]
        return np.full((len(clips), len(self.labels)), self.value, np.float32)


class FakePersonCounter:
    def __init__(self, count=2, fail_on_call=None, name="fake-yolo"):
        self.count_value, self.fail_on_call, self.calls, self.name = count, fail_on_call, 0, name

    def count(self, frames):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("counter broke")
        return [self.count_value] * len(frames)

    def boxes(self, frames):
        """Plain boxes (0-1 of the frame), one per counted person, side by side."""
        return [[(0.05 + 0.3 * i, 0.2, 0.3 + 0.3 * i, 0.95) for i in range(n)] for n in self.count(frames)]


class FakeExpressionScorer:
    """No faces on a uniform frame; otherwise `faces_per_frame` in-memory crops with fixed per-label scores."""

    def __init__(self, labels, faces_per_frame=1, scores=None, fail_on_call=None, name="fake-expression"):
        self.labels, self.faces_per_frame, self.name = list(labels), faces_per_frame, name
        self.scores = np.asarray(scores if scores is not None else [0.5, 0.2, 0.1, 0.1][:len(self.labels)], np.float32)
        self.fail_on_call, self.calls = fail_on_call, 0

    def faces(self, frame):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("scorer broke")
        if frame.std() == 0:
            return []
        return [frame[:8, :8].copy() for _ in range(self.faces_per_frame)]

    def score(self, crops):
        return np.tile(self.scores, (len(crops), 1))
