"""006 US2: Timeline tab on 005's mock world (tests/insights_fixtures.py). Segment classes come from 005's chain."""
import json

import pytest
import yaml

from tests.insights_fixtures import SECRET, SENTINEL, T0, build_world
from tpa_index import demo, review
from tpa_index.config import CONFIG_DIR, ConfigError, load_labels
from tpa_index.schema import append_blind_queue, read_official


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


def _tl(world):
    return demo.DemoController(labels=world["labels"], pipeline=world["pipeline"])


def _lock_eval(world):
    """An overlap segment with only one reviewer: blind labeling is unfinished, so the eval lock holds again."""
    seg = next(s for s in read_official("segments").to_pylist() if s["segment_id"] == world["ids"]["d01"][0])
    append_blind_queue([{"queue": "overlap", "position": 0, "segment_id": seg["segment_id"],
                         "video_id": "d01", "start_s": seg["start_s"], "end_s": seg["end_s"], "target_label": None,
                         "seed": 2026, "created_at": T0}])


def test_timeline_signals(tmp_path, monkeypatch):
    world = build_world(tmp_path, monkeypatch, overrides={"skip_stages": [("people", "x01"), ("audio", "x01"),
                                                                          ("expression", "x01"),
                                                                          ("transcript", "x01")]})
    tl = _tl(world)
    rows, hidden = tl.segments_table("d01", show_staff=True)
    assert hidden == 0 and len(rows) == 20
    assert rows[0] == ["0:00–0:08", "1", "No", "Singing voice", "AI suggestion, unvalidated: Neutral", "Youth"]
    assert rows[2][1] == "3"
    assert rows[10] == ["1:00–1:08", "0", "No", "Silence", "no faces", "Idle / setup"]
    assert rows[16][2] == "Yes" and rows[16][3] == "Speech"
    assert all(r[4].startswith(demo.EXPRESSION_PREFIX) or r[4] in ("no faces", "none", demo.NA) for r in rows)
    missing, _ = tl.segments_table("x01", show_staff=True)
    assert {(r[1], r[2], r[3], r[4]) for r in missing} == {(demo.NA,) * 4}

    table, ids = tl.moments_table("d01")
    assert len(table) == len(ids) > 0
    assert "Lyricism / spoken word" in {r[1] for r in table}
    assert not any("_" in r[1] or "_" in r[2] for r in table)  # readable names, no raw label IDs
    one, _ = tl.segments_table("d01", moment_id=ids[0], show_staff=True)
    assert 0 < len(one) < 20


def test_staff_led_excluded_by_default(world):
    """SC-005: staff-led rows are hidden by default and counted; classes are exactly 005's segment_table."""
    from tpa_index.indicators import segment_table
    from tpa_index.rules import context, indicator_estimates
    from tpa_index.view import segment_view, verified_labels

    tl = _tl(world)
    rows, hidden = tl.segments_table("e02")
    assert hidden == 6 and len(rows) == 14  # planted staff stretch e02 10-14 (model) + top-up label at 15 (verified)
    assert not any(r[5].startswith("Staff-led") for r in rows)
    shown, _ = tl.segments_table("e02", show_staff=True)
    assert [r[5] for r in shown[10:16]] == [f"Staff-led · {demo.UNVALIDATED}"] * 5 + ["Staff-led · staff-verified"]
    e01, hidden = tl.segments_table("e01", show_staff=True)
    assert [r[5] for r in e01[13:16]] == ["Staff-led · staff-verified"] * 3  # blind labels decide
    d01, hidden = tl.segments_table("d01", show_staff=True)
    assert [r[5] for r in d01[13:16]] == [f"Staff-led · {demo.UNVALIDATED}"] * 3  # no blind label: model rule

    p, labels = world["pipeline"], world["labels"]
    view = segment_view(p, labels)
    est = indicator_estimates(view, p["insights"]["indicator_rules"], context(p))
    expected = segment_table(view, verified_labels(view, labels, p), est, labels)
    assert [s["cls"] for s in tl.segments] == expected["cls"].tolist()

    bad = load_labels()
    next(lab for lab in bad["labels"] if lab["name"] == "instruction")["name"] = "teaching"
    with pytest.raises(ConfigError, match="instruction"):
        demo.check_rule_labels(bad)


def test_timeline_never_shows_transcript(world):
    """SC-004 (Timeline): the planted transcript text, its character count and the planted filename never appear."""
    from tpa_index.app import build_app

    tl = _tl(world)
    assert all("transcript_chars" not in s and "visual_scores" not in s for s in tl.segments)
    out = [tl.video_choices()]
    for vid in ("d01", "e01", "e02"):
        out += [tl.moments_table(vid), tl.segments_table(vid), tl.segments_table(vid, show_staff=True)]
    text = json.dumps(out)
    assert ".mp4" not in text and "mock/" not in text
    text += json.dumps(build_app(labels=world["labels"], pipeline=world["pipeline"]).config, default=str)
    assert SENTINEL not in text and SECRET not in text and "mock/" not in text  # Gradio's own sample asset is an .mp4
    assert not any("transcript" in h.lower() for h in demo.SEGMENT_HEADERS + demo.MOMENT_HEADERS)


def test_names_and_eval_lock(world):
    """The Timeline uses the Review tab's eval lock: eval videos are not listed and their rows stay empty."""
    from tpa_index.app import ReviewController

    tl = _tl(world)
    names = dict((v, n) for n, v in tl.video_choices()[0])
    assert names["d01"] == "Video 1 · early · live (d01)"  # e01 is also "Video 1 · early", same kind: the ID breaks the tie
    assert names["x01"] == "Video 1 · phase not set"

    _lock_eval(world)
    choices, reason = tl.video_choices()
    rv = ReviewController(labels=world["labels"], pipeline=world["pipeline"])
    assert reason and reason == rv.video_choices()[1]
    assert [v for _, v in choices] == [v["video_id"] for v in review.review_videos()] == ["d01", "x01"]
    assert [n for n, _ in choices] == ["Video 1 · early", "Video 1 · phase not set"]
    for vid in ("e01", "e02"):
        assert tl.moments_table(vid) == ([], []) and tl.segments_table(vid, show_staff=True) == ([], 0)
        assert tl.clip(vid, tl.moments_table("d01")[1][0])[0] is None

    shown = [{"video_id": "v04", "phase": "showcase", "kind": "live"}, {"video_id": "v14", "phase": "showcase",
                                                                         "kind": "promo"}]
    assert demo.video_name(shown[0], shown[:1]) == "Video 4 · showcase"
    assert demo.video_name({"video_id": "w04", "phase": "showcase", "kind": "clip"},
                           [shown[0], {"video_id": "w04", "phase": "showcase", "kind": "clip"}]) == "Video 4 · showcase · clip"


def test_display_keys_are_presentation_only(labels, tmp_path):
    from tpa_index.config import check_guideline, guideline_fingerprint

    assert demo.display_label("lyricism_spoken_word", labels) == "Lyricism / spoken word"
    assert demo.display_label("group_performance", labels) == "Group performance"
    assert check_guideline(labels) == f"guideline: ok ({labels['guideline_version']})"
    raw = yaml.safe_load((CONFIG_DIR / "labels.yaml").read_text())
    for lab in raw["labels"]:
        lab.pop("display", None)
    plain = tmp_path / "labels.yaml"
    plain.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert guideline_fingerprint(load_labels(plain)) == guideline_fingerprint(labels)
    raw["labels"][0]["display"] = " "
    plain.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ConfigError, match="display"):
        load_labels(plain)
