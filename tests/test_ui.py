"""006 UI redesign (Draft 3): tab order, Scorecard compare and download, Search filter, Review decisions and
observations, offline fonts. Mock world only (tests/insights_fixtures.py plus a real 005 insights run)."""
import json

import docx
import pytest

from tests.insights_fixtures import build_world
from tests.test_demo import _lock_eval
from tests.test_insights import _evaluate_folder
from tpa_index import cli, demo, report, ui_html
from tpa_index.schema import read_official, read_reviews

TABS = ["Scorecard", "Search moments", "Review queue", "Process a video", "Blind labels", "Export", "Help"]


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
    assert "Youth indicators by phase" in drives and "Youth voice time" in drives
    assert "What drives it" not in drives and "25%" not in drives and "Expression Index" not in drives
    for r in cmp["rows"]:  # short label shown, exact tag on hover and in the legend
        label, _ = ui_html.tag_kind(r["tag"])
        assert f'title="{r["tag"]}"' in drives and label in drives and f"{label} = {r['tag']}" in drives
    assert {ui_html.tag_kind(t)[0] for t in ("staff-verified", "mixed (40% staff-verified)", "model estimate (learned)",
                                             "model estimate (unvalidated)")} == {
        "Staff-verified", "Partly verified", "AI estimate (trained)", "AI estimate (unchecked)"}

    voice = {ph: float(head[("youth_voice_minutes_per_hour", ph)]["value"]) for ph in ("early", "showcase")}
    hero = ui_html.hero(cmp)
    assert f">{voice['showcase']:.1f}<" in hero and f">{voice['early']:.1f}<" in hero
    assert f"×{voice['showcase'] / voice['early']:.1f} since early sessions" in hero
    assert head[("youth_voice_minutes_per_hour", "showcase")]["tag"] in hero and ui_html.HERO_TEXT in hero
    swapped = ui_html.hero(ins.compare("showcase", "early"))
    assert f">{voice['early']:.1f}<" in swapped and "since showcase" in swapped
    assert "Not available" in ui_html.hero(ins.compare("early", "mid"))  # a phase with no videos
    page = json.dumps(_app(world).config, default=str)
    assert "Expression Index" not in page and "scorecard-limitations" not in page and "current vocabulary" not in page


def test_scorecard_facts(world):
    ins = demo.InsightsController()
    facts = {f["cap"]: f for f in ins.facts()}
    assert facts["AI agreed with human labels"]["value"] == "n/a"  # no evaluate result yet
    _evaluate_folder(report.latest_result("insights").parent)
    ins.load()
    facts = {f["cap"]: f for f in ins.facts()}
    agreed = facts["AI agreed with human labels"]
    assert agreed["value"] == "70%"
    assert agreed["note"] == "21 of 30 blind-labelled clips matched the blind human label · range 50–85%"  # 0.7 x 30
    assert facts["Checked on"]["value"] == "30"
    assert "staff-led left out" in facts["Youth time counted"]["note"]
    clips = read_official("segments").num_rows  # every official clip in the mock world
    assert ins.subtitle() == f"4 videos · {clips} clips · youth time only"  # the mock videos have no cohort


def test_download_writes_word_report(world, tmp_path, monkeypatch):
    from tpa_index.app import report_download

    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    path, msg = report_download()
    assert msg == "" and path.endswith(".docx") and path.startswith(str(tmp_path))
    text = "\n".join(p.text for p in docx.Document(path).paragraphs)
    from tpa_index.word_report import TITLE

    assert TITLE in text and "SAMPLE" not in text


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


def test_header_is_one_bar(world):
    """The tab bar is the only top bar: no separate header block; app name before the tabs and the note pushed right
    are drawn on that one element; the credit is in the footer (the header credit ID stays, hidden)."""
    from tpa_index.app import page_css

    by = _by_id(_app(world))
    assert "app-header" not in by and "app-top" not in by
    css = page_css()
    bar = "#app-tabs > .tab-wrapper"
    assert f'{bar}::before {{ content: "{ui_html.APP_NAME} {ui_html.APP_TAG}"' in css
    assert f'{bar}::after {{ content: "Local only · no faces stored"' in css
    assert "flex-wrap: wrap" in css.split(f"{bar} {{", 1)[1].split("}", 1)[0]
    assert by["app-credit-footer"]["props"]["visible"] is not False
    assert by["app-credit-header"]["props"]["visible"] is False
    assert "logo" not in css.lower() and not [c for c in _app(world).config["components"] if c["type"] == "image"]


def _local_phases(tmp_path, monkeypatch, phases):
    """A live video config (as staff edit it) that sets phases differently from the stored runs."""
    import yaml

    from tpa_index import config

    videos = [{"video_id": v, "filename": f"mock/{v}.mp4", "orientation": "landscape", "kind": "live",
               "phase": ph, "cohort": None, "split": "eval" if v.startswith("e") else "dev"} for v, ph in phases.items()]
    path = tmp_path / "videos.live.yaml"
    path.write_text(yaml.safe_dump({"videos": videos}))
    monkeypatch.setattr(config, "LOCAL_VIDEOS", path)


def test_live_phase_in_every_dropdown(world, tmp_path, monkeypatch):
    """Item 1: a phase set in the video config shows in Search, Review and Process, as insights reads it."""
    from tpa_index import process
    from tpa_index.app import build_app

    _local_phases(tmp_path, monkeypatch, {"d01": "mid", "e01": "early", "e02": "showcase", "x01": "showcase"})
    tl = demo.DemoController(labels=world["labels"], pipeline=world["pipeline"])
    names = {v: n for n, v in tl.video_choices()[0]}
    assert names["x01"].startswith("Video 1 · showcase") and names["d01"].startswith("Video 1 · mid")
    assert not any("phase not set" in n for n in names.values())
    by = _by_id(build_app(labels=world["labels"], pipeline=world["pipeline"]))
    for dd in ("search-video", "review-video"):
        shown = {v: n for n, v in by[dd]["props"]["choices"]}
        assert shown["x01"].startswith("Video 1 · showcase") and shown["d01"].startswith("Video 1 · mid"), dd
    proc = {v: n for n, v in process.video_choices()[0]}
    assert proc["x01"].startswith("Video 1 · showcase") and proc["d01"].startswith("Video 1 · mid")
    spans = [ph for ph, _, _ in tl.search("")["spans"]]
    assert spans == ["early", "mid", "showcase"]  # lanes in phase order


def test_staff_led_toggle_and_line(world):
    """Item 2: the toggle really filters, and a line always says what happened."""
    tl = demo.DemoController(labels=world["labels"], pipeline=world["pipeline"])
    none = tl.search("")
    assert none["staff"] == 0 and ui_html.staff_line(none) == "No staff-led moments found by the AI in these videos."
    one = tl.search("", video_id="d01")
    assert ui_html.staff_line(one) == "No staff-led moments found by the AI in this video."
    target = one["results"][0]["moment_id"]
    for s in tl.segments:  # plant a staff-led moment: every clip in it classed staff-led
        if s["moment_id"] == target:
            s["cls"] = "staff_model"
    hidden, shown = tl.search("", video_id="d01"), tl.search("", video_id="d01", show_staff=True)
    assert target not in [r["moment_id"] for r in hidden["results"]] and hidden["hidden"] == 1
    assert target in [r["moment_id"] for r in shown["results"]] and shown["hidden"] == 0
    assert ui_html.staff_line(hidden) == "1 staff-led moment hidden."
    assert ui_html.staff_line(shown) == "1 staff-led moment shown."
    assert "1 staff-led moment hidden." in ui_html.strip(hidden)


def test_review_queue_details(world, monkeypatch):
    """Items 4, 7-11: compact queue items, More collapsed, readable sorted pickers, nothing to decide until a moment
    is picked, and a plain message for a video without moments."""
    from tpa_index.app import EMPTY_QUEUE, PICK_MOMENT, ReviewController, build_app
    from tpa_index.demo import display_label

    rv = ReviewController(labels=world["labels"], pipeline=world["pipeline"])
    rows = rv.timeline("d01")[1]
    first = rv.queue_choices(rows)[0][0]
    assert first.split("\n") == [f"0:00–0:32 · {display_label(rows[0]['primary_label'], world['labels'])}", "Not reviewed"]
    head = rv.queue_head("Video 1 · early", rows)
    assert "To review · Video 1 · early" in head and "0 / 3 done" in head
    assert rv.suggestion(None) == PICK_MOMENT

    by = _by_id(build_app(labels=world["labels"], pipeline=world["pipeline"]))
    assert by["review-suggest"]["props"]["value"] == PICK_MOMENT
    for b in ("accept", "edit", "reject", "defer"):
        assert by[f"review-{b}"]["props"]["interactive"] is False
    assert by["review-more"]["props"]["open"] is False
    for dd in ("review-add-label", "review-add-obs"):
        names = [n for n, _ in by[dd]["props"]["choices"]]
        assert names == sorted(names, key=str.lower) and not any("_" in n for n in names), dd
    assert "Performance" in [n for n, _ in by["review-add-label"]["props"]["choices"]]
    assert "Staging issue" in [n for n, _ in by["review-add-obs"]["props"]["choices"]]

    assert rv.queue_empty("d01", rows) == ""
    rv.moments = [m for m in rv.moments if m["video_id"] != "x01"]  # a video where the AI found no moment
    assert rv.queue_empty("x01", rv.timeline("x01")[1]) == EMPTY_QUEUE


def test_eval_lock_holds_without_showing_its_message(world):
    """The lock is unchanged (eval videos stay out of Search and Review); only its message is no longer shown."""
    from tpa_index.app import ReviewController, build_app
    from tpa_index.review import eval_unlocked

    _lock_eval(world)
    assert not eval_unlocked()[0]
    app = build_app(labels=world["labels"], pipeline=world["pipeline"])
    page = json.dumps([c for c in app.config["components"] if c["props"].get("elem_id") != "help-list"], default=str)
    assert "Eval videos are hidden" not in page and "without a second reviewer" not in page  # (the Help list explains it)
    by = _by_id(app)
    for dd in ("search-video", "review-video"):
        assert not {"e01", "e02"} & {v for _, v in by[dd]["props"]["choices"]}, dd
    rv = ReviewController(labels=world["labels"], pipeline=world["pipeline"])
    assert rv.timeline("e01") == ([], [], "Pick a video.")
