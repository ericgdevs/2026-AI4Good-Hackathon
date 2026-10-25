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


LABEL_GROUPS = ("activity", "category", "actor", "audio", "expression", "indicator", "status", "null")
SOURCES = ("model", "rules", "staff")  # rules: computed flags; staff: set by staff only. Neither has prompts
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
        if source not in SOURCES:
            raise ConfigError(f"label {name}: source must be one of {', '.join(SOURCES)}")
        prompts = lab.get("prompts")
        if source != "model":
            if prompts:
                raise ConfigError(f"label {name}: {source} labels must not have prompts")
        elif not isinstance(prompts, list) or not 3 <= len(prompts) <= 5 or not all(isinstance(x, str) and x for x in prompts):
            raise ConfigError(f"label {name}: needs 3-5 prompt phrasings")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(lab.get("color", ""))):
            raise ConfigError(f"label {name}: color must be #RRGGBB")
        if "display" in lab and not (isinstance(lab["display"], str) and lab["display"].strip()):
            raise ConfigError(f"label {name}: display must be a non-empty string")
    groups = {lab["group"] for lab in labels}
    for group in LABEL_GROUPS:
        if group not in groups:
            raise ConfigError(f"labels: group {group} has no labels")
    expressions = {lab["name"] for lab in labels if lab["group"] == "expression"}
    if expressions != EXPRESSIONS:
        raise ConfigError(f"labels: expression labels must be exactly {sorted(EXPRESSIONS)}")
    obs = d.get("observation_types") or []
    names = [o.get("name") if isinstance(o, dict) else None for o in obs]
    for o, name in zip(obs, names):
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ConfigError(f"observation type {name!r}: name must be lowercase letters, digits and _")
        if names.count(name) > 1:
            raise ConfigError(f"observation type {name}: duplicate name")
        if name in seen:
            raise ConfigError(f"observation type {name}: clashes with a label name")
        if not isinstance(o.get("description"), str) or not o["description"].strip():
            raise ConfigError(f"observation type {name}: description required")
    return d


def observation_types(labels: dict) -> list[str]:
    return [o["name"] for o in labels.get("observation_types") or []]


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
    if not 0 <= _number(m, "close_ratio", "pipeline.moments") <= 1:
        raise ConfigError("pipeline.moments: close_ratio must be within 0-1")
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
    _number(cal, "null_percentile", "pipeline.calibrate")
    if not 0 < cal["null_percentile"] <= 100:
        raise ConfigError("pipeline.calibrate: null_percentile must be within (0, 100]")
    if not isinstance(cal.get("null_group"), str):
        raise ConfigError("pipeline.calibrate: null_group must be a quoted group name (bare null is not a name)")
    if cal["null_group"] in d["visual_groups"] or cal["null_group"] in [m.get("primary_group"), *(m.get("secondary_groups") or [])]:
        raise ConfigError("pipeline.calibrate: null_group must not be a visual, primary or secondary group")
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
    groups = set(d["visual_groups"]) | {d["calibrate"]["null_group"]}
    for g in groups:
        if not model_labels(labels, {g}):
            raise ConfigError(f"pipeline: group {g} (visual or null) has no model-scored labels")
    scored = {lab["name"] for lab in model_labels(labels)}
    for model, per_label in d["label_thresholds"].items():
        for label in per_label:
            if label not in scored:
                raise ConfigError(f"pipeline.label_thresholds.{model}.{label}: not a model-scored label")
    activity = {lab["name"] for lab in model_labels(labels, {"activity"})}
    for label in (d["models"].get("motion") or {}).get("labels") or []:
        if label not in activity:
            raise ConfigError(f"pipeline.models.motion: {label} is not an activity label")
    _check_signals(d, labels)
    _check_insights(d, labels)
    if not isinstance(d.get("app"), dict):
        raise ConfigError("pipeline: app section missing")
    _check_app(d["app"], d, labels)
    return d


def _check_weight_entry(entry, where: str, keys=("file", "sha256")) -> None:
    if not isinstance(entry, dict) or not all(k in entry for k in keys):
        raise ConfigError(f"{where}: needs {', '.join(keys)}")
    f = entry["file"]
    if not isinstance(f, str) or not f or "/" in f or "\\" in f or f in (".", ".."):
        raise ConfigError(f"{where}.file: must be a bare filename")
    if "sha256" in keys and entry["sha256"] is not None \
            and not (isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
        raise ConfigError(f"{where}.sha256: must be null or 64 lowercase hex characters")


def _check_mapping(mapping, where: str, allowed: set[str]) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise ConfigError(f"{where}: must map label -> [model class, ...]")
    seen: set[str] = set()
    for label, classes in mapping.items():
        if label not in allowed:
            raise ConfigError(f"{where}.{label}: not an allowed label ({', '.join(sorted(allowed))})")
        if not isinstance(classes, list) or not classes or not all(isinstance(c, str) and c for c in classes):
            raise ConfigError(f"{where}.{label}: must be a non-empty list of class names")
        for c in classes:
            if c in seen:
                raise ConfigError(f"{where}.{label}: class {c!r} is mapped to more than one label")
            seen.add(c)


def _check_signals(d: dict, labels: dict) -> None:
    """004 signal stages: models and signals sections (load time; model class names are checked at startup)."""
    models, sig = d["models"], d.get("signals")
    t = models.get("transcript")
    if not isinstance(t, dict) or not all(isinstance(t.get(k), str) for k in ("hpc", "mac")):
        raise ConfigError("pipeline.models.transcript: hpc and mac model sizes are required")
    if t.get("hpc_device") not in ("cuda", "cpu"):
        raise ConfigError("pipeline.models.transcript.hpc_device: must be cuda or cpu")
    if not isinstance(t.get("compute_type"), dict) or not all(isinstance(t["compute_type"].get(k), str)
                                                              for k in ("cuda", "cpu")):
        raise ConfigError("pipeline.models.transcript.compute_type: needs cuda and cpu")
    a = models.get("audio")
    if not isinstance(a, dict) or not all(isinstance(a.get(k), str) for k in ("hpc", "mac")):
        raise ConfigError("pipeline.models.audio: hpc and mac model ids are required")
    p = models.get("people")
    if not isinstance(p, dict):
        raise ConfigError("pipeline.models.people: hpc and mac weight entries are required")
    for k in ("hpc", "mac"):
        _check_weight_entry(p.get(k), f"pipeline.models.people.{k}")
    e = models.get("expression")
    if not isinstance(e, dict):
        raise ConfigError("pipeline.models.expression: face_detector and model are required")
    _check_weight_entry(e.get("face_detector"), "pipeline.models.expression.face_detector",
                        ("repo", "file", "revision", "sha256"))
    if not re.fullmatch(r"[0-9a-f]{40}", str(e["face_detector"]["revision"])):
        raise ConfigError("pipeline.models.expression.face_detector.revision: must be a 40-hex commit")
    _check_weight_entry(e.get("model"), "pipeline.models.expression.model", ("name", "file", "sha256"))
    if not isinstance(sig, dict):
        raise ConfigError("pipeline: signals section missing")
    for where, block, key in (("pipeline.signals.people", sig.get("people"), "person_conf"),
                              ("pipeline.signals.expression", sig.get("expression"), "face_score")):
        if not isinstance(block, dict):
            raise ConfigError(f"{where}: section missing")
        if not 0 < _number(block, key, where) < 1:
            raise ConfigError(f"{where}: {key} must be within (0, 1)")
    if not 0 <= _number(sig["people"], "min_box_height_frac", "pipeline.signals.people") < 1:
        raise ConfigError("pipeline.signals.people: min_box_height_frac must be within [0, 1)")
    side = sig["expression"].get("detect_max_side")
    if not isinstance(side, int) or isinstance(side, bool) or side < 320:
        raise ConfigError("pipeline.signals.expression.detect_max_side: must be an integer >= 320")
    audio_labels = {lab["name"] for lab in model_labels(labels, {"audio"})}
    amap = (sig.get("audio") or {}).get("mapping")
    _check_mapping(amap, "pipeline.signals.audio.mapping", audio_labels)
    missing = audio_labels - set(amap)
    if missing:
        raise ConfigError(f"pipeline.signals.audio.mapping: audio label {sorted(missing)[0]} has no classes")
    # confident is derived later (005), never model-scored here
    expr_labels = {lab["name"] for lab in model_labels(labels, {"expression"})} - {"confident"}
    _check_mapping(sig["expression"].get("mapping"), "pipeline.signals.expression.mapping", expr_labels)


_CONDITIONS = ("audio", "has_speech", "clears", "primary_in", "person_count", "indicator", "all", "any")


def _check_condition(c, where: str, names: dict, defined: list[str]) -> None:
    """One rule condition (contracts/config.md grammar); recursive for all/any."""
    if not isinstance(c, dict) or len(c) != 1:
        raise ConfigError(f"{where}: a condition is a mapping with exactly one key")
    (key, value), = c.items()
    if key not in _CONDITIONS:
        raise ConfigError(f"{where}: unknown condition {key!r} (allowed: {', '.join(_CONDITIONS)})")
    if key in ("all", "any"):
        if not isinstance(value, list) or not value:
            raise ConfigError(f"{where}.{key}: must be a non-empty list")
        for i, sub in enumerate(value):
            _check_condition(sub, f"{where}.{key}[{i}]", names, defined)
    elif key == "audio" and value not in names["audio"]:
        raise ConfigError(f"{where}.audio: {value!r} is not an audio label")
    elif key == "has_speech" and value is not True:
        raise ConfigError(f"{where}.has_speech: must be true")
    elif key == "clears" and value not in names["visual"]:
        raise ConfigError(f"{where}.clears: {value!r} is not a model-scored visual label")
    elif key == "primary_in" and value != "primary_like" and not (
            isinstance(value, list) and value and all(v in names["activity"] for v in value)):
        raise ConfigError(f"{where}.primary_in: must be primary_like or a list of activity labels")
    elif key == "person_count" and not (isinstance(value, dict) and len(value) == 1 and next(iter(value)) in
                                        ("eq", "gte", "lte") and isinstance(next(iter(value.values())), int)
                                        and not isinstance(next(iter(value.values())), bool)):
        raise ConfigError(f"{where}.person_count: must be {{eq|gte|lte: integer}}")
    elif key == "indicator" and value not in defined:
        raise ConfigError(f"{where}.indicator: {value!r} is not an earlier indicator rule")


def _check_insights(d: dict, labels: dict) -> None:
    """005 insights: block (load time)."""
    ins, where = d.get("insights"), "pipeline.insights"
    if not isinstance(ins, dict):
        raise ConfigError(f"{where}: section missing")
    for key, block in (("min_support", ins), ("failure_gallery", ins), ("resamples", ins.get("bootstrap") or {}),
                       ("block", ins.get("bootstrap") or {}), ("seed", ins.get("bootstrap") or {})):
        v = block.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v < (0 if key == "seed" else 1):
            raise ConfigError(f"{where}: {key} must be an integer >= {0 if key == 'seed' else 1}")
    if _number(ins, "applause_after_s", where) < 0:
        raise ConfigError(f"{where}: applause_after_s must be >= 0")
    if not 0 < _number(ins, "audio_threshold", where) < 1:
        raise ConfigError(f"{where}: audio_threshold must be within (0, 1)")
    lc = ins.get("learned")
    if not isinstance(lc, dict) or not isinstance(lc.get("folds"), int) or lc["folds"] < 2 \
            or not isinstance(lc.get("max_iter"), int) or lc["max_iter"] < 1 or not isinstance(lc.get("seed"), int) \
            or not isinstance(lc.get("C"), (int, float)) or lc["C"] <= 0:
        raise ConfigError(f"{where}.learned: needs folds (int >= 2), C (> 0), max_iter (int >= 1) and seed (int)")
    caps = (ins.get("expression_index") or {}).get("per_hour_cap")
    if not isinstance(caps, dict) or set(caps) != {"solo_moments_minutes_per_hour", "youth_voice_minutes_per_hour"} \
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in caps.values()):
        raise ConfigError(f"{where}.expression_index.per_hour_cap: needs solo_moments_minutes_per_hour and "
                          "youth_voice_minutes_per_hour, each a number > 0")
    if ins.get("labels_source") not in ("learned", "zero-shot"):
        raise ConfigError(f"{where}.labels_source: must be learned or zero-shot")
    names = {"audio": {lab["name"] for lab in model_labels(labels, {"audio"})},
             "activity": {lab["name"] for lab in model_labels(labels, {"activity"})},
             "visual": {lab["name"] for lab in model_labels(labels, set(d["visual_groups"]))}}
    pl = ins.get("primary_like")
    if not isinstance(pl, list) or not pl or not all(v in names["activity"] for v in pl):
        raise ConfigError(f"{where}.primary_like: must be a non-empty list of activity labels")
    fusion = ins.get("fusion")
    if not isinstance(fusion, list):
        raise ConfigError(f"{where}.fusion: must be a list")
    for i, rule in enumerate(fusion):
        w = f"{where}.fusion[{i}]"
        if not isinstance(rule, dict) or set(rule) != {"label", "require_any"}:
            raise ConfigError(f"{w}: needs exactly label and require_any")
        if rule["label"] not in names["activity"]:
            raise ConfigError(f"{w}.label: {rule['label']!r} is not a model-scored activity label")
        _check_condition({"any": rule["require_any"]}, f"{w}.require_any", names, [])
    rules = ins.get("indicator_rules")
    indicators = {lab["name"] for lab in labels["labels"] if lab["group"] == "indicator"}
    if not isinstance(rules, dict):
        raise ConfigError(f"{where}.indicator_rules: must map indicator -> condition")
    defined: list[str] = []
    for name, cond in rules.items():
        if name not in indicators:
            raise ConfigError(f"{where}.indicator_rules.{name}: not an indicator label")
        _check_condition(cond, f"{where}.indicator_rules.{name}", names, defined)
        defined.append(name)


def required_threshold(pipeline: dict, model: str, label: str, stage_hint: str = "predict") -> float:
    """Calibrated per-model threshold; never the global label_threshold (005 must not silently fall back)."""
    value = (pipeline.get("label_thresholds") or {}).get(model, {}).get(label)
    if value is None:
        raise ConfigError(f"no calibrated threshold for {label} (model {model}): run tpa-index calibrate "
                          f"--stage {stage_hint} --split dev on HPC, then sync")
    return value


SIGNAL_STAGES = ("transcript", "audio", "people", "expression")


def weights_dir() -> Path:
    """GitHub-release weights (not on Hugging Face): TPA_WEIGHTS_DIR, else $HF_HOME/tpa-weights."""
    if os.environ.get("TPA_WEIGHTS_DIR"):
        return Path(os.environ["TPA_WEIGHTS_DIR"]).expanduser()
    hf_home = os.environ.get("HF_HOME") or "~/.cache/huggingface"
    return Path(hf_home).expanduser() / "tpa-weights"


def pinned_weight(entry: dict) -> Path:
    """Path of a pinned weight file under weights_dir(), checked against its sha256. Never downloads."""
    path = weights_dir() / entry["file"]
    if not path.is_file():
        raise ConfigError(f"weights not found: {entry['file']}; run scripts/download_weights.sh on the login node")
    return check_sha256(path, entry)


def check_sha256(path: Path, entry: dict) -> Path:
    """`path` if its sha256 matches entry["sha256"]; ConfigError if unpinned or mismatched."""
    if not entry.get("sha256"):
        raise ConfigError(f"weights not pinned: set sha256 for {entry['file']}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != entry["sha256"]:
        raise ConfigError(f"weights checksum mismatch: {entry['file']}")
    return path


def stage_config_hash(stage: str, labels: dict, pipeline: dict, video: dict, hardware: str = "mac") -> str:
    """Resume key for one signal stage and video: config_hash over only the keys that stage reads on this
    hardware ("hpc" or "mac"). Prompts, other stages and the other machine's model entry are left out, so
    editing them reruns nothing."""
    if stage not in SIGNAL_STAGES:
        raise ConfigError(f"unknown signal stage {stage!r}")
    if hardware not in ("hpc", "mac"):
        raise ConfigError(f"hardware must be hpc or mac, not {hardware!r}")
    eff = effective_settings(pipeline, video)
    m = pipeline["models"][stage]
    model = {k: v for k, v in m.items() if k not in ("hpc", "mac")}
    if "hpc" in m:
        model["entry"] = m[hardware]
    part: dict = {"stage": stage, "window_s": eff["window_s"], "overlap_s": eff["overlap_s"], "model": model}
    if stage in ("transcript", "audio"):
        part["audio_sample_rate"] = pipeline["audio_sample_rate"]
    if stage in ("people", "expression"):
        part["frames_per_window"] = eff["frames_per_window"]
    if stage in pipeline["signals"]:
        part["signals"] = pipeline["signals"][stage]
    if stage == "expression":
        part["min_face_px"] = pipeline["min_face_px"]
    names = sorted(lab["name"] for lab in model_labels(labels, {stage}) if stage in ("audio", "expression"))
    return config_hash({"labels": names}, part)


def _check_app(app: dict, pipeline: dict, labels: dict) -> None:
    where = "pipeline.app"
    for key in ("port", "seed", "sample_size", "overlap_size", "rare_min_positives", "topup_per_label",
                "clip_height"):
        if not isinstance(app.get(key), int) or isinstance(app.get(key), bool) or app[key] < 1:
            raise ConfigError(f"{where}: {key} must be an integer >= 1")
    if app["overlap_size"] > app["sample_size"]:
        raise ConfigError(f"{where}: overlap_size must not exceed sample_size")
    _number(app, "padding_s", where)
    if not 0 <= app["padding_s"] <= 30:
        raise ConfigError(f"{where}: padding_s must be within 0-30")
    if not 240 <= app["clip_height"] <= 1080:
        raise ConfigError(f"{where}: clip_height must be within 240-1080")
    if not 1024 <= app["port"] <= 65535:
        raise ConfigError(f"{where}: port must be within 1024-65535")
    groups = {lab["group"] for lab in labels["labels"]}
    selectable = {lab["group"] for lab in labels["labels"] if lab.get("source", "model") != "rules"}
    for key in ("blind_groups", "indicator_groups"):
        if not isinstance(app.get(key), list) or not app[key]:
            raise ConfigError(f"{where}: {key} must be a non-empty list")
        for g in app[key]:
            if g not in groups:
                raise ConfigError(f"{where}: {key} has unknown group {g!r}")
    primary = pipeline["moments"]["primary_group"]
    if primary not in app["blind_groups"]:
        raise ConfigError(f"{where}: blind_groups must include the primary group {primary}")
    for g in app["blind_groups"]:
        if g == pipeline["calibrate"]["null_group"]:
            raise ConfigError(f"{where}: blind_groups must not include the null group")
        if g not in selectable:
            raise ConfigError(f"{where}: blind_groups group {g} has no model or staff labels")
    if app.get("uncertain_label") not in {lab["name"] for lab in rules_labels(labels)}:
        raise ConfigError(f"{where}: uncertain_label must be a rules label")
    if app.get("skip_observation") not in observation_types(labels):
        raise ConfigError(f"{where}: skip_observation must be one of the observation types")


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


# ---- 006: app theme (config/theme.yaml) --------------------------------------------------------------------------
THEME = CONFIG_DIR / "theme.yaml"
THEME_COLOURS = ("background", "surface", "text", "text_secondary", "accent", "accent_deep", "muted", "border",
                 "border_strong", "track", "accent_soft", "header_text", "header_text_muted", "attention")
AA = 4.5  # WCAG AA, normal text


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two #RRGGBB colours."""
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def load_theme(path=None) -> dict:
    d = _load(Path(path or THEME))
    hexes = [d.get(k) for k in THEME_COLOURS] + list(d.get("chart_series") or [])
    if not all(isinstance(h, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", h) for h in hexes):
        raise ConfigError("theme: every colour must be #RRGGBB")
    if len(d.get("chart_series") or []) < 3:
        raise ConfigError("theme: chart_series needs at least 3 colours")
    if not isinstance(d.get("credit"), str) or not d["credit"].strip():
        raise ConfigError("theme: credit text required")
    pairs = [(fg, bg) for fg in ("text", "text_secondary", "accent", "accent_deep") for bg in ("background", "surface")]
    extra = [("surface", "accent"), ("surface", "accent_deep"), ("header_text", "accent_deep"),  # top bar and buttons
             ("header_text_muted", "accent_deep"), ("attention", "surface"), ("text", "accent_soft"),
             ("accent_deep", "accent_soft")]
    for fg, bg in [*pairs, *extra]:
        ratio = contrast_ratio(d[fg], d[bg])
        if ratio < AA:
            raise ConfigError(f"theme: {fg} on {bg} has contrast {ratio:.2f}, below WCAG AA {AA}")
    return d


# ---- 006: guideline history (refuse a vocabulary change without a guideline bump) ----------------------------------
GUIDELINE_HISTORY = Path(os.environ.get("TPA_GUIDELINE_HISTORY", CONFIG_DIR / "guideline_history.yaml"))


def guideline_fingerprint(labels: dict) -> str:
    """Vocabulary only: label (name, group, source) and observation-type names. Prompts, colours, descriptions and
    display names are presentation, not guideline content."""
    vocab = {"labels": sorted([lab["name"], lab["group"], lab.get("source", "model")] for lab in labels["labels"]),
             "observations": sorted(observation_types(labels))}
    return hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()[:12]


def check_guideline(labels: dict, path=None) -> str:
    """Status line for `check`; raises if the current version is recorded with a different vocabulary."""
    path = Path(path or GUIDELINE_HISTORY)
    history = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    version, fp = labels["guideline_version"], guideline_fingerprint(labels)
    if version not in history:
        return f'guideline: {version} not recorded; add to config/guideline_history.yaml:  "{version}": "{fp}"'
    if history[version] != fp:
        raise ConfigError(f"labels: vocabulary changed without bumping guideline_version ({version})")
    return f"guideline: ok ({version})"


HELP = CONFIG_DIR / "help.yaml"
HELP_TABS = {"scorecard", "search", "review", "process", "blind", "export", "help"}  # optional per-entry `tabs:`


def load_help(path=None) -> list[dict]:
    """The Help tab's questions: each entry a topic, a question and an answer, in plain text with no links."""
    items = _load(Path(path or HELP)).get("help")
    if not isinstance(items, list) or not items:
        raise ConfigError("help: 'help' must be a non-empty list")
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) and item[k].strip()
                                                 for k in ("topic", "question", "answer")):
            raise ConfigError(f"help entry {i}: needs a topic, a question and an answer")
        if re.search(r"https?://|www\.", " ".join(item[k] for k in ("topic", "question", "answer")), re.IGNORECASE):
            raise ConfigError(f"help entry {i}: links are not allowed (the app works offline)")
        tabs = item.get("tabs", [])
        if not isinstance(tabs, list) or not set(tabs) <= HELP_TABS:
            raise ConfigError(f"help entry {i}: tabs must be a list of {', '.join(sorted(HELP_TABS))}")
    return items
