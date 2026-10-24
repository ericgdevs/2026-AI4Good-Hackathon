"""005 Part 2: youth-only indicators, time classes, tags, headline (mock world only)."""
import re

import pytest

from tests.insights_fixtures import build_world
from tpa_index.indicators import STAFF_VERIFIED, run_insights, tag
from tpa_index.review import blind_rows
from tpa_index.schema import append_reviews, read_official

TAG = re.compile(r"^(staff-verified|model estimate \(unvalidated\)|model estimate \(learned\)|"
                 r"mixed \((\d{1,2})% staff-verified\))$")


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


def _ind(out, indicator, scope):
    return next(r for r in out["indicators"] if r["indicator"] == indicator and r["scope"] == scope)


def _tb(out, scope):
    return next(b for b in out["time_breakdown"] if b["scope"] == scope)


def _zero_shot(world):
    p = dict(world["pipeline"])
    p["insights"] = {**p["insights"], "labels_source": "zero-shot"}
    return p


def test_staff_led_stretch_not_in_youth_numbers(world):
    """SC-002: the model-only staff stretch (e02) and a verified staff segment (e01) add nothing to youth numbers."""
    pipeline = _zero_shot(world)
    base = run_insights(pipeline, world["labels"])
    e02 = _tb(base, "e02")
    assert e02["staff_led_model_s"] >= 5 * 6.0  # the five planted staff segments
    youth_e02 = _ind(base, "performance_share", "e02")["denominator"]
    assert youth_e02 == e02["youth_s"] and youth_e02 + e02["staff_led_model_s"] + e02["staff_led_verified_s"] \
        + e02["idle_s"] == e02["total_s"]
    # verified variant: the labeler marks an e01 performance segment as staff-led
    seg = {s["segment_id"]: s for s in read_official("segments").to_pylist()}["e01_96000"]
    before_youth, before_staff = _tb(base, "e01")["youth_s"], _tb(base, "e01")["staff_led_verified_s"]
    from tpa_index.review import previous_blind

    append_reviews(blind_rows(seg, None, ["instruction", "actor_staff_facilitation"], True, "mock-staff-a",
                              world["labels"], pipeline, previous=previous_blind(seg["segment_id"], "mock-staff-a")))
    after = run_insights(pipeline, world["labels"])
    assert _tb(after, "e01")["staff_led_verified_s"] == before_staff + 6.0
    assert _tb(after, "e01")["youth_s"] == before_youth - 6.0
    assert _ind(after, "performance_share", "e01")["denominator"] == before_youth - 6.0


def test_time_totals_match_duration(world):
    """SC-003 (part 2): youth + staff-led + idle = total = duration per video; no share above 1."""
    out = run_insights(world["pipeline"], world["labels"])
    for b in out["time_breakdown"]:
        parts = b["youth_s"] + b["staff_led_verified_s"] + b["staff_led_model_s"] + b["idle_s"]
        assert abs(parts - b["total_s"]) < 1e-6
        if b["scope_type"] == "video":
            assert abs(b["total_s"] - 120.0) < 1e-6
    assert all(r["value"] is None or r["value"] <= 1 + 1e-9 for r in out["indicators"] if r["unit"] == "share")


def test_every_indicator_row_has_tag(world):
    """SC-004: every indicator and headline row carries an allowed tag and a verified share in [0, 1]."""
    out = run_insights(world["pipeline"], world["labels"])
    for r in out["indicators"]:
        m = TAG.match(r["tag"])
        assert m, r
        assert 0.0 <= r["verified_share"] <= 1.0
        if m.group(2):
            assert 1 <= int(m.group(2)) <= 99
    for r in out["headline"]:
        assert r["status"] != "ok" or TAG.match(r["tag"])
    assert tag(1.0) == STAFF_VERIFIED and tag(0.0) == "model estimate (unvalidated)" and tag(0.5) == \
        "mixed (50% staff-verified)" and tag(0.0, "model estimate (learned)") == "model estimate (learned)"


def test_learned_source_tags_model_numbers_learned(world):
    out = run_insights(world["pipeline"], world["labels"])
    assert out["labels_source"] == "learned" and out["model_tag"] == "model estimate (learned)"
    unreviewed = _ind(out, "performance_share", "d01")  # dev video: labeled only 3 segments
    assert unreviewed["tag"].startswith("mixed") or unreviewed["tag"] == "model estimate (learned)"
    x01 = _ind(out, "performance_share", "x01")  # never labeled
    assert x01["tag"] == "model estimate (learned)"
    assert _ind(out, "positive_expression_rate", "x01")["tag"] == "model estimate (unvalidated)"


def test_indicator_rules(world):
    out = run_insights(_zero_shot(world), world["labels"])
    # positive expression: faces only, argmax; e02 faces are positive, e01 faces neutral
    assert _ind(out, "positive_expression_rate", "e02")["value"] == 1.0
    assert _ind(out, "positive_expression_rate", "e01")["value"] == 0.0
    assert _ind(out, "positive_expression_rate", "e01")["tag"] == "model estimate (unvalidated)"
    # original work: staff only
    ow = _ind(out, "original_work_minutes", "all")
    assert ow["status"] == "not tagged yet" and ow["value"] is None
    # unknown phase: excluded from the headline and listed
    assert out["excluded_videos"] == ["x01"] and {r["phase"] for r in out["headline"]} == {"early", "showcase"}
    # youth voice counts spoken word with speech/has_speech, never staff-led time
    assert _ind(out, "youth_voice_minutes", "e01")["value"] > 0
    # solo moments: contiguous runs (e01 has solo segments 0-1 and spoken word 16-18 with one person)
    assert _ind(out, "solo_moments_count", "e01")["value"] >= 1


def test_applause_window_counts_each_second_once(world):
    out = run_insights(_zero_shot(world), world["labels"])
    r = _ind(out, "audience_response_per_perf_min", "e01")
    # the applause segment right after spoken word (6 s slice) is inside the 10 s window, counted once
    assert r["numerator"] == pytest.approx(6.0)


def test_headline_phases_and_change(world):
    out = run_insights(world["pipeline"], world["labels"])
    perf = [r for r in out["headline"] if r["indicator"] == "performance_share"]
    early, show = (next(r for r in perf if r["phase"] == p) for p in ("early", "showcase"))
    assert early["change_vs_early"] is None
    assert show["change_vs_early"] == pytest.approx(show["value"] - early["value"])
