"""Load the YAML configs (labels, pipeline, videos) and fingerprint them."""
import hashlib
import json
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


def load_labels(path=None) -> dict:
    return _load(path or CONFIG_DIR / "labels.yaml")


def load_pipeline(path=None) -> dict:
    return _load(path or CONFIG_DIR / "pipeline.yaml")


def load_videos(path=None) -> list[dict]:
    if path is None:
        path = LOCAL_VIDEOS if LOCAL_VIDEOS.exists() else EXAMPLE_VIDEOS
    return _load(path).get("videos") or []


def config_hash(labels: dict, pipeline: dict) -> str:
    blob = json.dumps({"labels": labels, "pipeline": pipeline}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
