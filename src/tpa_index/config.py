"""Load the YAML configs (labels, pipeline, videos) and fingerprint them."""
import hashlib
import json
import re
from pathlib import Path

import yaml

from tpa_index.paths import repo_root

CONFIG_DIR = repo_root() / "config"
LOCAL_VIDEOS = CONFIG_DIR / "videos.local.yaml"
EXAMPLE_VIDEOS = CONFIG_DIR / "videos.example.yaml"


class ConfigError(Exception):
    pass


def _load(path: Path) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot load {Path(path).name}: {type(e).__name__}") from None


LABEL_GROUPS = ("activity", "category", "actor", "audio", "expression", "indicator", "status")
EXPRESSIONS = {"confident", "upset", "positive", "neutral", "negative"}
OVERRIDABLE = ("window_s", "overlap_s", "frames_per_window")


def _check_labels(d: dict) -> dict:
    if not isinstance(d.get("guideline_version"), str):
        raise ConfigError("labels: guideline_version must be a string")
    labels = d.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ConfigError("labels: 'labels' must be a non-empty list")
    seen = set()
    for lab in labels:
        name = lab.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError("label ?: missing name")
        if name in seen:
            raise ConfigError(f"label {name}: duplicate name")
        seen.add(name)
        if lab.get("group") not in LABEL_GROUPS:
            raise ConfigError(f"label {name}: unknown group {lab.get('group')!r}")
        prompts = lab.get("prompts")
        if not isinstance(prompts, list) or not 3 <= len(prompts) <= 5 or not all(isinstance(x, str) and x for x in prompts):
            raise ConfigError(f"label {name}: needs 3-5 prompt phrasings")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(lab.get("color", ""))):
            raise ConfigError(f"label {name}: color must be #RRGGBB")
    groups = {lab["group"] for lab in labels}
    for group in LABEL_GROUPS:
        if group not in groups:
            raise ConfigError(f"labels: group {group} has no labels")
    expressions = {lab["name"] for lab in labels if lab["group"] == "expression"}
    if expressions != EXPRESSIONS:
        raise ConfigError(f"labels: expression labels must be exactly {sorted(EXPRESSIONS)}")
    return d


def _number(d, key, where="pipeline"):
    if not isinstance(d.get(key), (int, float)) or isinstance(d.get(key), bool):
        raise ConfigError(f"{where}: {key} must be a number")
    return d[key]


def _check_windows(where: str, window: float, overlap: float, frames) -> None:
    if not 0 <= overlap < window:
        raise ConfigError(f"{where}: need 0 <= overlap_s < window_s")
    if not isinstance(frames, int) or frames < 1:
        raise ConfigError(f"{where}: frames_per_window must be an integer >= 1")


def _check_pipeline(d: dict) -> dict:
    for key in ("window_s", "overlap_s", "frames_per_window", "min_face_px", "label_threshold", "audio_sample_rate"):
        _number(d, key)
    _check_windows("pipeline", d["window_s"], d["overlap_s"], d["frames_per_window"])
    if not 0 <= d["label_threshold"] <= 1:
        raise ConfigError("pipeline: label_threshold must be within 0-1")
    if not isinstance(d.get("video_phases"), list) or not d["video_phases"]:
        raise ConfigError("pipeline: video_phases must be a non-empty list")
    m = d.get("moments")
    if not isinstance(m, dict):
        raise ConfigError("pipeline: moments section missing")
    for key in ("max_gap_windows", "min_windows", "needs_review_below", "conflict_margin"):
        _number(m, key, "pipeline.moments")
    for group in [m.get("primary_group"), *(m.get("secondary_groups") or [])]:
        if group not in LABEL_GROUPS:
            raise ConfigError(f"pipeline.moments: unknown group {group!r}")
    overrides = d.get("overrides") or {}
    for section in ("by_kind", "by_video"):
        for target, values in (overrides.get(section) or {}).items():
            where = f"pipeline.overrides.{section}.{target}"
            bad = set(values or {}) - set(OVERRIDABLE)
            if bad:
                raise ConfigError(f"{where}: only {', '.join(OVERRIDABLE)} may be overridden (got {sorted(bad)[0]})")
            eff = {k: d[k] for k in OVERRIDABLE}
            if section == "by_video":  # a video may also inherit its kind's override; check the worst case
                for kind_values in (overrides.get("by_kind") or {}).values():
                    combo = {**eff, **kind_values, **values}
                    _check_windows(where, combo["window_s"], combo["overlap_s"], combo["frames_per_window"])
            eff.update(values or {})
            _check_windows(where, eff["window_s"], eff["overlap_s"], eff["frames_per_window"])
    return d


def _check_videos(videos: list, phases: list) -> list:
    seen = set()
    enums = {"orientation": {"landscape", "portrait"}, "kind": {"live", "edited", "clip", "promo"},
             "split": {"dev", "eval"}, "phase": set(phases)}
    for v in videos:
        vid = v.get("video_id")
        if not isinstance(vid, str) or not re.fullmatch(r"[a-z0-9]+", vid):
            raise ConfigError("video ?: video_id must be lowercase letters/digits only")
        if vid in seen:
            raise ConfigError(f"video {vid}: duplicate video_id")
        seen.add(vid)
        filename = v.get("filename")
        if not isinstance(filename, str) or Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ConfigError(f"video {vid}: filename must be relative to DATA_DIR without '..'")
        for key, allowed in enums.items():
            if v.get(key) not in allowed:
                raise ConfigError(f"video {vid}: {key} must be one of {sorted(allowed)}")
    return videos


def load_labels(path=None) -> dict:
    return _check_labels(_load(path or CONFIG_DIR / "labels.yaml"))


def load_pipeline(path=None) -> dict:
    return _check_pipeline(_load(path or CONFIG_DIR / "pipeline.yaml"))


def load_videos(path=None) -> list[dict]:
    if path is None:
        path = LOCAL_VIDEOS if LOCAL_VIDEOS.exists() else EXAMPLE_VIDEOS
    return _check_videos(_load(path).get("videos") or [], load_pipeline()["video_phases"])


def config_hash(labels: dict, pipeline: dict) -> str:
    blob = json.dumps({"labels": labels, "pipeline": pipeline}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def effective_settings(pipeline: dict, video: dict) -> dict:
    """Window settings for one video: defaults < overrides.by_kind < overrides.by_video."""
    overrides = pipeline.get("overrides") or {}
    out = {k: pipeline[k] for k in OVERRIDABLE}
    out.update((overrides.get("by_kind") or {}).get(video.get("kind"), {}))
    out.update((overrides.get("by_video") or {}).get(video.get("video_id"), {}))
    return out
