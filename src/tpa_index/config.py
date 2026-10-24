"""Load the YAML configs (labels, pipeline, videos) and fingerprint them."""
import hashlib
import json
import os
import re
from pathlib import Path

import yaml

from tpa_index.paths import repo_root

CONFIG_DIR = repo_root() / "config"
# env overrides exist so tests and checkpoints never touch the real local files
LOCAL_VIDEOS = Path(os.environ.get("TPA_LOCAL_VIDEOS", CONFIG_DIR / "videos.local.yaml"))
THRESHOLDS_LOCAL = Path(os.environ.get("TPA_THRESHOLDS_FILE", CONFIG_DIR / "thresholds.local.yaml"))
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
        source = lab.get("source", "model")
        if source not in ("model", "rules"):
            raise ConfigError(f"label {name}: source must be 'model' or 'rules'")
        prompts = lab.get("prompts")
        if source == "rules":
            if prompts:
                raise ConfigError(f"label {name}: rules labels must not have prompts")
        elif not isinstance(prompts, list) or not 3 <= len(prompts) <= 5 or not all(isinstance(x, str) and x for x in prompts):
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


def rules_labels(labels: dict) -> list[dict]:
    return [lab for lab in labels["labels"] if lab.get("source", "model") == "rules"]


def model_labels(labels: dict, groups=None) -> list[dict]:
    """Labels scored by a model, optionally limited to the given groups."""
    return [lab for lab in labels["labels"] if lab.get("source", "model") == "model"
            and (groups is None or lab["group"] in groups)]


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
    for key in ("max_gap_windows", "min_windows", "needs_review_margin"):
        _number(m, key, "pipeline.moments")
    if not 0 <= m["needs_review_margin"] <= 1:
        raise ConfigError("pipeline.moments: needs_review_margin must be within 0-1")
    _number(d, "uncertain_margin")
    if not 0 <= d["uncertain_margin"] < 1:
        raise ConfigError("pipeline: uncertain_margin must be within [0, 1)")
    if not isinstance(d.get("visual_groups"), list) or not d["visual_groups"] \
            or any(g not in LABEL_GROUPS for g in d["visual_groups"]):
        raise ConfigError(f"pipeline: visual_groups must be a non-empty list of {', '.join(LABEL_GROUPS)}")
    lt = d.get("label_thresholds")
    if lt is None:
        d["label_thresholds"] = lt = {}
    if not isinstance(lt, dict) or not all(isinstance(v, dict) for v in lt.values()):
        raise ConfigError("pipeline: label_thresholds must map model -> {label: threshold}")
    for model, per_label in lt.items():
        for label, t in per_label.items():
            if not isinstance(t, (int, float)) or isinstance(t, bool) or not 0 <= t <= 1:
                raise ConfigError(f"pipeline.label_thresholds.{model}.{label}: must be within 0-1")
    models = d.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("main"), dict) \
            or not all(isinstance(models["main"].get(k), str) for k in ("hpc", "mac")):
        raise ConfigError("pipeline.models: main.hpc and main.mac model ids are required")
    motion = models.get("motion") or {}
    if motion.get("enabled") and (not motion.get("id") or not motion.get("labels")):
        raise ConfigError("pipeline.models.motion: id and labels are required when enabled")
    if motion and (not isinstance(motion.get("frames_per_window"), int) or motion["frames_per_window"] < 1):
        raise ConfigError("pipeline.models.motion: frames_per_window must be an integer >= 1")
    batch = models.get("batch_windows") or {}
    for dev in ("cuda", "mps", "cpu"):
        if not isinstance(batch.get(dev), int) or batch[dev] < 1:
            raise ConfigError(f"pipeline.models.batch_windows.{dev}: must be an integer >= 1")
    idle = d.get("idle")
    if not isinstance(idle, dict) or not isinstance(idle.get("prompts"), list) or not 1 <= len(idle["prompts"]) <= 5:
        raise ConfigError("pipeline.idle: prompts must be a list of 1-5 phrasings")
    for key in ("score_threshold", "change_threshold"):
        _number(idle, key, "pipeline.idle")
        if not 0 <= idle[key] <= 1:
            raise ConfigError(f"pipeline.idle: {key} must be within 0-1")
    cal = d.get("calibrate")
    if not isinstance(cal, dict):
        raise ConfigError("pipeline: calibrate section missing")
    for where, block, key in (("pipeline.idle", idle, "calibrate_top_pct"), ("pipeline.calibrate", cal, "top_pct")):
        _number(block, key, where)
        if not 0 < block[key] < 100:
            raise ConfigError(f"{where}: {key} must be within (0, 100)")
    _number(cal, "min_threshold", "pipeline.calibrate")
    if not 0 <= cal["min_threshold"] <= 1:
        raise ConfigError("pipeline.calibrate: min_threshold must be within 0-1")
    if not isinstance(cal.get("min_segments"), int) or cal["min_segments"] < 1:
        raise ConfigError("pipeline.calibrate: min_segments must be an integer >= 1")
    it = d.setdefault("idle_thresholds", {}) or {}
    d["idle_thresholds"] = it
    if not isinstance(it, dict) or not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in it.values()):
        raise ConfigError("pipeline: idle_thresholds must map model -> threshold within 0-1")
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


def _check_pipeline_labels(d: dict, labels: dict) -> dict:
    """Checks that need the label vocabulary."""
    groups = set(d["visual_groups"])
    for g in groups:
        if not model_labels(labels, {g}):
            raise ConfigError(f"pipeline: visual group {g} has no model-scored labels")
    scored = {lab["name"] for lab in model_labels(labels)}
    for model, per_label in d["label_thresholds"].items():
        for label in per_label:
            if label not in scored:
                raise ConfigError(f"pipeline.label_thresholds.{model}.{label}: not a model-scored label")
    activity = {lab["name"] for lab in model_labels(labels, {"activity"})}
    for label in (d["models"].get("motion") or {}).get("labels") or []:
        if label not in activity:
            raise ConfigError(f"pipeline.models.motion: {label} is not an activity label")
    return d


def load_pipeline(path=None, labels=None) -> dict:
    d = _load(path or CONFIG_DIR / "pipeline.yaml")
    local = THRESHOLDS_LOCAL
    if local.exists():  # gitignored, data-derived (tpa-index calibrate); wins per model and label
        extra = _load(local)
        merged = {m: dict(v) for m, v in (d.get("label_thresholds") or {}).items()}
        for model, per_label in (extra.get("label_thresholds") or {}).items():
            merged.setdefault(model, {}).update(per_label or {})
        d["label_thresholds"] = merged
        d["idle_thresholds"] = {**(d.get("idle_thresholds") or {}), **(extra.get("idle_thresholds") or {})}
    _check_pipeline(d)
    return _check_pipeline_labels(d, labels or load_labels())


def idle_threshold(pipeline: dict, model: str | None) -> float:
    """Calibrated idle-score threshold for this model, else idle.score_threshold."""
    return (pipeline.get("idle_thresholds") or {}).get(model, pipeline["idle"]["score_threshold"])


def threshold(pipeline: dict, model: str | None, label: str) -> float:
    """Per-label threshold for this model, else the global label_threshold."""
    return (pipeline.get("label_thresholds") or {}).get(model, {}).get(label, pipeline["label_threshold"])


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
