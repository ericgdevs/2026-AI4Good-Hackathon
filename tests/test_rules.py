"""005 rule grammar: conditions, fusion, signal-based indicator estimates (mock world)."""
import copy

import pytest

from tests.insights_fixtures import build_world
from tpa_index.rules import apply_fusion, context, evaluate_condition, indicator_estimates
from tpa_index.view import segment_view

SEG = {"audio_ok": {"singing_voice": True, "music": False, "speech": None}, "has_speech": True,
       "model_cleared": {"activity": {"singing"}, "category": {"performance"}, "actor": set()},
       "model_primary": "singing", "person_count": 1}
CTX = {"primary_like": ["singing", "dance"]}


@pytest.mark.parametrize("cond,expected", [
    ({"audio": "singing_voice"}, True), ({"audio": "music"}, False), ({"audio": "speech"}, None),
    ({"has_speech": True}, True), ({"clears": "performance"}, True), ({"clears": "dance"}, False),
    ({"primary_in": "primary_like"}, True), ({"primary_in": ["dance"]}, False),
    ({"person_count": {"eq": 1}}, True), ({"person_count": {"gte": 2}}, False), ({"person_count": {"lte": 1}}, True),
])
def test_each_condition(cond, expected):
    assert evaluate_condition(cond, SEG, CTX) is expected


def test_missing_signals_are_none():
    seg = {**SEG, "person_count": None, "has_speech": float("nan")}
    assert evaluate_condition({"person_count": {"eq": 1}}, seg, CTX) is None
    assert evaluate_condition({"has_speech": True}, seg, CTX) is None


@pytest.mark.parametrize("op,parts,expected", [
    ("all", [True, True], True), ("all", [True, None], None), ("all", [False, None], False),
    ("any", [False, False], False), ("any", [False, None], None), ("any", [True, None], True),
])
def test_all_any_truth_table(op, parts, expected):
    conds = {True: {"audio": "singing_voice"}, False: {"audio": "music"}, None: {"audio": "speech"}}
    assert evaluate_condition({op: [conds[p] for p in parts]}, SEG, CTX) is expected


def test_indicator_condition_sees_earlier_rules():
    rules = {"solo_on_stage": {"person_count": {"eq": 1}}, "voice_projection": {"indicator": "solo_on_stage"}}
    import pandas as pd

    out = indicator_estimates(pd.DataFrame([{**SEG, "segment_id": "s1"}]), rules, CTX)
    assert out.set_index("indicator")["fired"].to_dict() == {"solo_on_stage": True, "voice_projection": True}


@pytest.fixture
def world_view(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch)
    return w, segment_view(w["pipeline"], w["labels"])


def test_fusion_drops_unsupported_singing_and_repicks(world_view):
    w, view = world_view
    fused, counts = apply_fusion(view, w["pipeline"]["insights"]["fusion"], context(w["pipeline"]))
    before = view.set_index("segment_id").loc[w["fusion_segment"]]
    after = fused.set_index("segment_id").loc[w["fusion_segment"]]
    assert before["model_primary"] == "singing" and after["model_primary"] == "uncertain_mixed"
    assert "singing" not in after["model_cleared"]["activity"]
    assert counts["singing"]["fired"] >= 1
    supported = fused.set_index("segment_id").loc["e02_0"]
    assert supported["model_primary"] == "singing"  # singing_voice 0.8 keeps it


def test_fusion_not_applicable_without_audio(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch, overrides={"skip_stages": [("audio", "e02")]})
    view = segment_view(w["pipeline"], w["labels"])
    fused, counts = apply_fusion(view, w["pipeline"]["insights"]["fusion"], context(w["pipeline"]))
    assert fused.set_index("segment_id").loc["e02_0", "model_primary"] == "singing"
    assert counts["singing"]["not_applicable"] >= 4


def test_person_count_condition_from_config(world_view):
    """SC-012: the indicator rules follow config; a changed person_count changes which segments fire."""
    w, view = world_view
    rules = w["pipeline"]["insights"]["indicator_rules"]
    ctx = context(w["pipeline"])

    def solo(r):
        est = indicator_estimates(view, r, ctx)
        return set(est[(est["indicator"] == "solo_on_stage") & (est["fired"] == True)]["segment_id"])  # noqa: E712

    base = solo(rules)
    changed = copy.deepcopy(rules)
    changed["solo_on_stage"]["all"][0] = {"person_count": {"eq": 3}}
    assert "e02_0" in base and "e02_0" not in solo(changed) and "e02_12000" in solo(changed)
