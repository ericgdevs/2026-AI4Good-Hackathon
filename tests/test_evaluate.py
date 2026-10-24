"""005 Part 1: evaluation metrics, bootstrap, ablation, signal-based indicators (mock world only)."""
import numpy as np
import pandas as pd
import pytest

from tests.insights_fixtures import build_world
from tpa_index.evaluate import block_bootstrap, eval_frame, run_evaluate
from tpa_index.review import blind_rows
from tpa_index.schema import append_reviews, read_official
from tpa_index.view import human_labels, segment_view


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


def _values(payload):
    return {(r["system"], r["scope"], r["group"], r["label"], r["metric"]): (r["value"], r["ci_low"], r["ci_high"])
            for r in payload["metrics"]}


def test_dev_labels_change_nothing(world):
    """SC-001: blind labels on dev videos never change a metric."""
    before = _values(run_evaluate(world["pipeline"], world["labels"]))
    segs = {s["segment_id"]: s for s in read_official("segments").to_pylist()}
    for sid in world["ids"]["d01"][:3] + world["ids"]["x01"][:5]:  # relabel / add dev labels
        append_reviews(blind_rows(segs[sid], "dance", ["performance"], False, "mock-staff-a", world["labels"],
                                  world["pipeline"]))
    after = run_evaluate(world["pipeline"], world["labels"])
    assert _values(after) == before
    assert after["eval_set"]["dev_labels_ignored"] == 3  # only main-queue dev segments are counted as ignored


def test_bootstrap_same_seed_same_numbers(world):
    """SC-006: the same seed gives identical intervals; another seed differs."""
    a = run_evaluate(world["pipeline"], world["labels"])
    b = run_evaluate(world["pipeline"], world["labels"])
    assert _values(a) == _values(b)
    df = pd.DataFrame({"video_id": ["v"] * 40, "start_s": np.arange(40.0), "x": np.arange(40.0)})
    stat = lambda idx: float(df["x"].to_numpy()[idx].mean())  # noqa: E731
    assert block_bootstrap(df, stat, 10, 200, 1) == block_bootstrap(df, stat, 10, 200, 1)
    assert block_bootstrap(df, stat, 10, 200, 1) != block_bootstrap(df, stat, 10, 200, 2)


def test_block_bootstrap_whole_blocks_per_video():
    """SC-011: every resample is made of whole blocks of 10 consecutive rows from one video (last one trimmed)."""
    df = pd.DataFrame({"video_id": ["a"] * 25 + ["b"] * 17, "start_s": list(range(25)) + list(range(17))})
    blocks = [tuple(range(0, 10)), tuple(range(10, 20)), tuple(range(20, 25)),
              tuple(range(25, 35)), tuple(range(35, 42))]
    seen = []
    block_bootstrap(df, lambda idx: seen.append(list(idx)) or 0.0, 10, 50, 7)
    assert len(seen) == 50
    for sample in seen:
        assert len(sample) == len(df)
        pos = 0
        while pos < len(sample):
            block = next((b for b in blocks if tuple(sample[pos:pos + len(b)]) == b), None)
            if block is None:  # only the final, trimmed block may be a prefix of a real block
                rest = tuple(sample[pos:])
                assert any(b[:len(rest)] == rest for b in blocks), sample
                break
            assert len({df["video_id"][i] for i in block}) == 1
            pos += len(block)


def test_ablation_rows_same_segments(world):
    """SC-009: both systems are scored on the identical segment set; fusion changes only its own row."""
    out = run_evaluate(world["pipeline"], world["labels"])
    a, b = out["ablation"]
    assert a["system"] == "zero-shot" and b["system"] == "zero-shot + signal rules"
    assert a["n_segments"] == b["n_segments"] == out["eval_set"]["labeled_eval_segments"]
    view = segment_view(world["pipeline"], world["labels"])
    df, _ = eval_frame(view, human_labels(world["labels"], world["pipeline"]))
    assert world["fusion_segment"] in set(df["segment_id"])
    assert "singing:1" in b["rules_fired"] and a["rules_fired"] == ""
    assert a["primary_accuracy"] != b["primary_accuracy"]  # the planted singing segment flips in the fusion row only


def test_metric_rules(world):
    out = run_evaluate(world["pipeline"], world["labels"])
    rows = [r for r in out["metrics"] if r["system"] == "zero-shot" and r["scope"] == "all"]
    by = {(r["group"], r["label"], r["metric"]): r for r in rows}
    # per-activity F1 uses primary + secondary human activities; uncertain_mixed is not a per-activity label
    assert ("activity", "uncertain_mixed", "f1") not in by
    assert by[("activity", "singing", "f1")]["support"] >= 5 and by[("activity", "singing", "f1")]["status"] == "ok"
    # too few to score below support 5, left out of macro-F1
    spoken = by[("activity", "lyricism_spoken_word", "f1")]
    assert spoken["status"] == "too few to score" and spoken["value"] is None
    assert "lyricism_spoken_word" in out["headline"][0]["excluded_activity"]
    # uncertain_mixed is its own class in the confusion matrix
    assert any(c["human_primary"] == "uncertain_mixed" for c in out["confusion"])
    # category kappa skips the segment with two ticked categories
    assert out["eval_set"]["category_kappa_excluded"] == 1
    # top-up labels are reported separately, never in the main metrics
    assert out["eval_set"]["labeled_eval_segments"] == 29 and out["topup"]
    # failure gallery: most confident wrong call first, IDs and labels only
    g = out["failure_gallery"]
    assert g[0]["segment_id"] == world["wrong_primary"] and g[0]["human_primary"] == "dance"
    assert set(g[0]) == {"rank", "segment_id", "video_id", "model_primary", "human_primary", "model_strength"}
    assert [r["model_strength"] for r in g] == sorted((r["model_strength"] for r in g), reverse=True)
    # per-video rows exist so the imbalance is visible
    assert {r["scope"] for r in out["metrics"]} >= {"all", "e01", "e02"}


def test_missing_people_run_stops_evaluate(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch, overrides={"skip_stages": [("people", "e02")]})
    from tpa_index.config import ConfigError

    with pytest.raises(ConfigError, match="official people runs missing for: e02"):
        run_evaluate(w["pipeline"], w["labels"])


def test_signal_indicator_table(world):
    out = run_evaluate(world["pipeline"], world["labels"])
    table = {r["indicator"]: r for r in out["signal_indicators"]}
    assert set(table) == {"solo_on_stage", "group_performance", "voice_projection"}
    assert table["group_performance"]["status"] == "ok" and table["group_performance"]["f1"] > 0.5
    assert table["solo_on_stage"]["status"] == "too few to score"
    assert all(r["tag"] == "model estimate (unvalidated)" for r in table.values())
    assert all(r["not_available"] == 0 for r in table.values())


def test_kappa_sentence_is_plain():
    from tpa_index.evaluate import kappa_sentence

    assert "moderate agreement" in kappa_sentence(0.5) and "undefined" in kappa_sentence(None)
