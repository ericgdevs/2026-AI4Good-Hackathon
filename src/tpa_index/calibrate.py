"""Propose thresholds from official dev-split data only (never eval). Values are data-derived: they go to the
gitignored config/thresholds.local.yaml, never into git."""
from collections import defaultdict

import numpy as np
import yaml

from tpa_index import config


def _rows(t):
    return t.to_pylist() if hasattr(t, "to_pylist") else list(t)


def _dev_videos(videos) -> set[str]:
    return {v["video_id"] for v in _rows(videos) if v["split"] == "dev"}


def dev_scores(predictions, videos, model: str) -> dict[str, np.ndarray]:
    """label -> dev-split scores of `model` (rules rows and other models excluded)."""
    dev = _dev_videos(videos)
    out = defaultdict(list)
    for p in _rows(predictions):
        if p["model"] == model and p["segment_id"].rsplit("_", 1)[0] in dev:
            out[p["label"]].append(p["score"])
    return {label: np.asarray(v, np.float64) for label, v in out.items()}


def propose_thresholds(predictions, videos, model: str, top_pct: float, min_segments: int,
                       min_threshold: float) -> dict[str, tuple[float | None, bool]]:
    """label -> (threshold, floored). threshold = max(percentile, min_threshold); None if too few dev segments.
    The floor keeps a label that never occurs in the dev videos from passing top_pct% of segments anyway."""
    out = {}
    for label, scores in sorted(dev_scores(predictions, videos, model).items()):
        if len(scores) < min_segments:
            out[label] = (None, False)
            continue
        value = float(np.percentile(scores, 100 - top_pct))
        out[label] = (max(value, min_threshold), value < min_threshold)
    return out


def idle_candidates(segments, videos, change_threshold: float) -> np.ndarray:
    """idle_score of dev-split segments whose picture barely changes."""
    dev = _dev_videos(videos)
    return np.asarray([s["idle_score"] for s in _rows(segments)
                       if s["video_id"] in dev and s.get("idle_score") is not None
                       and s.get("visual_change") is not None and s["visual_change"] < change_threshold], np.float64)


def propose_idle_threshold(segments, videos, top_pct: float, change_threshold: float,
                           min_segments: int) -> float | None:
    """(100 - top_pct) percentile of idle_score over dev low-change segments; change_threshold is never proposed."""
    scores = idle_candidates(segments, videos, change_threshold)
    return float(np.percentile(scores, 100 - top_pct)) if len(scores) >= min_segments else None


def write_local_thresholds(model: str, thresholds: dict[str, float | None], idle_score_threshold=None,
                           path=None) -> None:
    """Replace this model's blocks in the local thresholds file; other models are kept."""
    path = path or config.THRESHOLDS_LOCAL
    data = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    labels = data.setdefault("label_thresholds", {})
    labels[model] = {k: float(v) for k, v in thresholds.items() if v is not None}
    if idle_score_threshold is not None:
        data.setdefault("idle_thresholds", {})[model] = float(idle_score_threshold)
    path.write_text("# written by tpa-index calibrate; gitignored, data-derived\n" + yaml.safe_dump(data, sort_keys=True))
