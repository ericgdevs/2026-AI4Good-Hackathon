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


def null_floor(predictions, videos, model: str, null_labels: set[str], null_percentile: float,
               min_segments: int) -> tuple[float | None, int]:
    """(floor, n): the null_percentile of each dev segment's highest null-prompt score. None if too few segments.
    Unrelated prompts show how high pure noise scores; a real label must beat that."""
    dev = _dev_videos(videos)
    per_segment: dict[str, float] = {}
    for p in _rows(predictions):
        if p["model"] == model and p["label"] in null_labels and p["segment_id"].rsplit("_", 1)[0] in dev:
            per_segment[p["segment_id"]] = max(per_segment.get(p["segment_id"], 0.0), p["score"])
    if len(per_segment) < min_segments:
        return None, len(per_segment)
    return float(np.percentile(list(per_segment.values()), null_percentile)), len(per_segment)


def propose_thresholds(predictions, videos, model: str, top_pct: float, min_segments: int, min_threshold: float,
                       null_labels: set[str] = frozenset(), floor: float | None = None
                       ) -> dict[str, tuple[float | None, str | None]]:
    """label -> (threshold, floored_by). threshold = max(percentile, null floor, min_threshold);
    floored_by is "null" or "min" when a floor won, None otherwise. Null labels themselves are never proposed."""
    out = {}
    for label, scores in sorted(dev_scores(predictions, videos, model).items()):
        if label in null_labels:
            continue
        if len(scores) < min_segments:
            out[label] = (None, None)
            continue
        value = float(np.percentile(scores, 100 - top_pct))
        best, by = value, None
        if floor is not None and floor > best:
            best, by = floor, "null"
        if min_threshold > best:
            best, by = min_threshold, "min"
        out[label] = (best, by)
    return out


def idle_candidates(segments, videos, change_threshold: float) -> np.ndarray:
    """idle_score of dev-split segments whose picture barely changes."""
    dev = _dev_videos(videos)
    return np.asarray([s["idle_score"] for s in _rows(segments)
                       if s["video_id"] in dev and s.get("idle_score") is not None
                       and s.get("visual_change") is not None and s["visual_change"] < change_threshold], np.float64)


def propose_idle_threshold(segments, videos, top_pct: float, change_threshold: float, min_segments: int,
                           min_threshold: float = 0.0) -> tuple[float | None, bool]:
    """(threshold, floored): max((100 - top_pct) percentile of idle_score over dev low-change segments, min_threshold).
    None if too few segments. The floor stops a ~0 threshold from marking every still segment idle.
    change_threshold is never proposed."""
    scores = idle_candidates(segments, videos, change_threshold)
    if len(scores) < min_segments:
        return None, False
    value = float(np.percentile(scores, 100 - top_pct))
    return (min_threshold, True) if min_threshold > value else (value, False)


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
