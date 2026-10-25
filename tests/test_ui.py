"""006 UI redesign (Draft 3): tab order, Scorecard compare and download, Search filter, Review decisions and
observations, offline fonts. Mock world only (tests/insights_fixtures.py plus a real 005 insights run)."""
import json
import re

import docx
import pytest

from tests.insights_fixtures import build_world
from tests.test_demo import _lock_eval
from tests.test_insights import _evaluate_folder
from tpa_index import cli, demo, report, ui_html
from tpa_index.schema import read_official, read_reviews

TABS = ["Scorecard", "Search moments", "Review queue", "Blind labels", "Export"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch)
    assert cli.main(["insights"]) == 0
    return w


def _app(world):
    from tpa_index.app import build_app

    return build_app(labels=world["labels"], pipeline=world["pipeline"])


def _by_id(app):
    return {c["props"].get("elem_id"): c for c in app.config["components"]}


def test_tab_order_and_scorecard_opens_first(world):
    app = _app(world)
    tabs = [c["props"]["label"] for c in app.config["components"] if c["type"] == "tabitem"]
    assert tabs == TABS


def test_compare_dropdowns(world):
    ins = demo.InsightsController()
    assert ins.default_pair() == ("early", "showcase")
    by = _by_id(_app(world))
    assert by["scorecard-from"]["props"]["value"] == "early" and by["scorecard-to"]["props"]["value"] == "showcase"
    assert [v for _, v in by["scorecard-from"]["props"]["choices"]] == ins.phases()

    cmp = ins.compare("early", "showcase")
    head = {(r["indicator"], r["phase"]): r for r in ins.state["headline"]}
    for r in cmp["rows"]:
        assert r["a"] == pytest.approx(float(head[(r["indicator"], "early")]["value"]))
        assert r["tag"] == head[(r["indicator"], "showcase")]["tag"]
    drives = ui_html.drives(cmp)
    assert all(re.escape(r["tag"]) and r["tag"] in drives for r in cmp["rows"])
    assert ui_html.WEIGHTS_NOTE in drives and "Youth voice time" in drives
    idx = {r["scope"]: float(r["value"]) for r in ins.state["index"]}
    gauge = ui_html.gauge(cmp)
    assert f">{idx['showcase']:.0f}<" in gauge and f"{round(idx['showcase']) - round(idx['early']):d} points from early" in gauge
    assert "never a young person" in gauge and cmp["index_tag"] in gauge
    swapped = ins.compare("showcase", "early")
    assert swapped["index_b"] == pytest.approx(idx["early"])
    assert "Not available" in ui_html.gauge(ins.compare("early", "mid"))  # a phase with no videos


def test_scorecard_facts(world):
    ins = demo.InsightsController()
    facts = {f["cap"]: f for f in ins.facts()}
    assert facts["AI agreed with staff"]["value"] == "n/a"  # no evaluate result yet
    _evaluate_folder(report.latest_result("insights").parent)
    ins.load()
    facts = {f["cap"]: f for f in ins.facts()}
    assert facts["AI agreed with staff"]["value"] == "70/100" and "range 50–85" in facts["AI agreed with staff"]["note"]
    assert facts["Checked on"]["value"] == "30"
    assert "staff-led left out" in facts["Youth time counted"]["note"]
    assert ins.subtitle() == "4 videos · youth time only"  # the mock videos have no cohort


def test_download_writes_word_report(world, tmp_path, monkeypatch):
    from tpa_index.app import report_download

    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    path, msg = report_download()
    assert msg == "" and path.endswith(".docx") and path.startswith(str(tmp_path))
    text = "\n".join(p.text for p in docx.Document(path).paragraphs)
    assert "Program insights" in text and "SAMPLE" not in text


def test_download_without_insights(tmp_path, monkeypatch):
    from tpa_index.app import build_app, report_download

    build_world(tmp_path, monkeypatch)
    assert report_download() == (None, demo.EMPTY)
    by = _by_id(build_app())
    assert by["scorecard-download"]["props"]["visible"] is False and by["scorecard-empty"]["props"]["value"] == demo.EMPTY


def test_search_filter(world):
    tl = demo.DemoController(labels=world["labels"], pipeline=world["pipeline"])
    every = tl.search("")["results"]
    assert every and all(0 <= m <= 1 for m in tl.search("")["marks"])
    moments = {m["moment_id"]: m for m in read_official("moments").to_pylist()}

    def labs(r):
        return [lab for lab in [moments[r["moment_id"]]["primary_label"], *moments[r["moment_id"]]["secondary_labels"],
                                *moments[r["moment_id"]]["indicators"]]]

    solo = tl.search("solo")["results"]
    assert solo and all("solo_on_stage" in labs(r) or r["detail"].startswith("1 person") for r in solo)
    assert all("singing" in labs(r) for r in tl.search("solo singing")["results"])  # none in this world: 3 people
    spoken = tl.search("spoken word")["results"]
    assert spoken and all("lyricism_spoken_word" in labs(r) for r in spoken)
    group = tl.search("group performance")["results"]
    assert group and all("group_performance" in labs(r) or not r["detail"].startswith(("1 person", "people n/a"))
                         for r in group)
    review_needed = tl.search("needs review")["results"]
    assert review_needed and all(moments[r["moment_id"]]["needs_review"] for r in review_needed)
    assert len(tl.search("needs review", video_id="d01")["results"]) <= len(review_needed)
    assert tl.search("no such label at all")["results"] == []
    shown = tl.search("", show_staff=True)
    assert len(shown["results"]) == len(every) + tl.search("")["hidden"]
    assert all(r["video"].startswith("Video ") and "×" in r["detail"] for r in every)  # plain names, ratio only
    assert not any("%" in r["detail"] for r in every)

    _lock_eval(world)
    locked = tl.search("")["results"]
    assert locked and {r["video_id"] for r in locked} <= {"d01", "x01"}


def test_review_decision_observation_and_note_save(world):
    from tpa_index.app import ReviewController
    from tpa_index.export import build_records

    rv = ReviewController(labels=world["labels"], pipeline=world["pipeline"])
    tl = demo.DemoController(labels=world["labels"], pipeline=world["pipeline"])
    rows = rv.timeline("d01")[1]
    current = rv.open(rows, 0)[0]
    choices, ticked = rv.activities(current)
    assert choices[0][1] == current["primary_label"] and choices[0][0].endswith("(main)")
    assert "above its cut-off" in rv.suggestion(current) and "%" not in rv.suggestion(current)
    others = [lab for lab in current["secondary_labels"]]
    extra = next(v for _, v in choices if v not in ticked)
    keep = [lab for lab in ticked if lab not in others[:1]] + [extra]
    out = rv.decide(current, "Staff-A", "accept", keep, None, "my own words")
    assert out[3].startswith("Saved: accept")
    mine = [r for r in read_reviews().to_pylist() if r["mode"] == "review" and r["target_id"] == current["moment_id"]]
    by = {r["label"]: r for r in mine}
    assert by[current["primary_label"]]["action"] == "accept" and by[current["primary_label"]]["note"] == "my own words"
    assert by[extra]["action"] == "add" and by[extra]["source"] == "staff"
    if others:
        assert by[others[0]]["action"] == "reject"
    obs = rv.add_observation(current, "staff-a", "expression_positive", "smiling at the end")
    assert obs[3].startswith("Saved")

    rec = next(r for r in build_records(read_official("moments").to_pylist(), read_reviews().to_pylist(),
                                        world["labels"]) if r["start_s"] == current["start_s"] and r["video_id"] == "d01")
    assert rec["source"] == "staff" and rec["review_decision"] == "accepted"
    assert rec["contextual_observation"] == "expression_positive: smiling at the end"
    assert extra in rec["secondary_labels"] + rec["indicators"]
    assert demo.queue_status(current, rv._state()) == "Accepted"
    assert "AI noticed" in ui_html.noticed(tl.summary(current["moment_id"]))
    assert "(unvalidated)" in ui_html.noticed(tl.summary(current["moment_id"])) or \
        "No faces" in ui_html.noticed(tl.summary(current["moment_id"]))

    refused = rv.decide(current, "", "accept", keep, None, "x")
    assert refused[0] is None and "Enter a reviewer" in refused[3]


def test_queue_status_mapping():
    m = {"moment_id": "m", "primary_label": "dance", "secondary_labels": ["performance"], "indicators": []}
    row = lambda lab, action: {("m", lab): {"action": action}}  # noqa: E731
    assert demo.queue_status(m, {}) == "Not reviewed"
    assert demo.queue_status(m, row("dance", "defer")) == "Second look"
    assert demo.queue_status(m, row("dance", "accept")) == "Partly reviewed"
    assert demo.queue_status(m, {**row("dance", "accept"), **row("performance", "reject")}) == "Accepted"
    assert demo.queue_status(m, {**row("dance", "edit"), **row("performance", "accept")}) == "Edited"
    assert demo.strength_text(1.42) == "1.4× above its cut-off"


def test_no_external_font_or_cdn(world):
    from tpa_index.app import PAGE_JS, page_css, theme

    t = theme()
    fonts = [f for f in (*t._font, *t._font_mono) if hasattr(f, "stylesheet")]
    assert {f.name for f in fonts} == {"IBM Plex Sans", "IBM Plex Mono"}
    assert all(f.stylesheet()["url"] is None for f in fonts)  # served from Gradio's own static folder
    assert not t._stylesheets
    page = json.dumps(_app(world).config, default=str) + page_css() + PAGE_JS
    for host in ("googleapis", "gstatic", "cdnjs", "jsdelivr", "unpkg", "cdn."):
        assert host not in page, host
    assert "http" not in page_css() and "http" not in PAGE_JS


def test_review_shortcuts_in_page_js():
    from tpa_index.app import PAGE_JS

    for key, target in (("a", "#review-accept"), ("e", "#review-edit"), ("r", "#review-reject"),
                        ("d", "#review-defer")):
        assert f"{key}: '{target}'" in PAGE_JS
    assert "ArrowDown" in PAGE_JS and "#review-next" in PAGE_JS
    guard = PAGE_JS.index("t.tagName === 'TEXTAREA'")
    assert guard < PAGE_JS.index("#review-decide")  # text fields are checked before any shortcut
