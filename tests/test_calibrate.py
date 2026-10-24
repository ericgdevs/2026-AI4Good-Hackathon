import numpy as np
import pytest
import yaml

from tpa_index import config
from tpa_index.calibrate import propose_idle_threshold, propose_thresholds, write_local_thresholds
from tpa_index.config import idle_threshold, load_pipeline

VIDEOS = [{"video_id": "mock01", "split": "dev"}, {"video_id": "mock02", "split": "dev"},
          {"video_id": "mock04", "split": "eval"}]


def preds(label, scores_by_video, model="m1"):
    return [{"segment_id": f"{vid}_{i * 6000}", "label": label, "score": s, "model": model}
            for vid, scores in scores_by_video.items() for i, s in enumerate(scores)]


def test_percentile_on_known_scores():
    rows = preds("singing", {"mock01": list(np.linspace(0, 1, 51)), "mock02": list(np.linspace(0, 1, 50))})
    ((value, floored),) = propose_thresholds(rows, VIDEOS, "m1", 10, 20, 0.05).values()
    assert value == pytest.approx(np.percentile(list(np.linspace(0, 1, 51)) + list(np.linspace(0, 1, 50)), 90))
    assert not floored


def test_eval_split_never_changes_thresholds():
    dev = {"mock01": list(np.linspace(0, 1, 40))}
    a = propose_thresholds(preds("singing", {**dev, "mock04": [0.0] * 40}), VIDEOS, "m1", 10, 20, 0.05)
    b = propose_thresholds(preds("singing", {**dev, "mock04": [1.0] * 40}), VIDEOS, "m1", 10, 20, 0.05)
    assert a == b


def test_absent_label_hits_the_floor():
    rows = preds("acting", {"mock01": [0.001] * 30})
    assert propose_thresholds(rows, VIDEOS, "m1", 10, 20, 0.05) == {"acting": (0.05, True)}


def test_too_few_segments_and_other_models_ignored():
    rows = preds("singing", {"mock01": [0.5] * 5}) + preds("singing", {"mock01": [0.9] * 40}, model="m2") \
        + preds("uncertain_mixed", {"mock01": [1.0] * 40}, model="rules")
    assert propose_thresholds(rows, VIDEOS, "m1", 10, 20, 0.05) == {"singing": (None, False)}


def test_idle_proposal_uses_dev_low_change_segments_only():
    segs = [{"video_id": "mock01", "idle_score": s, "visual_change": 0.001} for s in np.linspace(0.1, 0.5, 30)]
    segs += [{"video_id": "mock01", "idle_score": 0.99, "visual_change": 0.5}] * 30  # moving: ignored
    segs += [{"video_id": "mock04", "idle_score": 0.99, "visual_change": 0.001}] * 30  # eval: ignored
    segs += [{"video_id": "mock01", "idle_score": None, "visual_change": None}] * 5  # old rows: ignored
    got = propose_idle_threshold(segs, VIDEOS, 20, 0.02, 20)
    assert got == pytest.approx(np.percentile(np.linspace(0.1, 0.5, 30), 80))
    assert propose_idle_threshold(segs[:10], VIDEOS, 20, 0.02, 20) is None


def test_write_keeps_other_models(tmp_path):
    path = tmp_path / "t.local.yaml"
    path.write_text(yaml.safe_dump({"label_thresholds": {"mac": {"singing": 0.2}}, "idle_thresholds": {"mac": 0.3}}))
    write_local_thresholds("hpc", {"singing": 0.12, "acting": None}, 0.04, path)
    data = yaml.safe_load(path.read_text())
    assert data["label_thresholds"] == {"mac": {"singing": 0.2}, "hpc": {"singing": 0.12}}
    assert data["idle_thresholds"] == {"mac": 0.3, "hpc": 0.04}
    assert path.read_text().startswith("# written by tpa-index calibrate")


def test_idle_threshold_fallback_and_override(tmp_path, monkeypatch):
    p = load_pipeline()
    assert idle_threshold(p, "hpc") == p["idle"]["score_threshold"]
    local = tmp_path / "t.local.yaml"
    local.write_text("idle_thresholds: {hpc: 0.04}\n")
    monkeypatch.setattr(config, "THRESHOLDS_LOCAL", local)
    p = load_pipeline()
    assert idle_threshold(p, "hpc") == 0.04 and idle_threshold(p, "mac") == p["idle"]["score_threshold"]
