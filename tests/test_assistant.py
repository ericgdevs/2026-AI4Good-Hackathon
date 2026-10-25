"""006 Help assistant: grounded on help.yaml, Ollama on 127.0.0.1:11434 only. A fake opener; never the real one."""
import io
import json
import urllib.error

import pytest

from tpa_index import assistant, ui_html
from tpa_index.config import load_help

ITEMS = load_help()


class FakeOllama:
    """Stands in for the HTTP opener: records each request and replies like /api/chat (or fails)."""

    def __init__(self, reply="A clip is an 8-second slice of a video.", error=None):
        self.reply, self.error, self.requests = reply, error, []

    def open(self, req, timeout=None):
        self.requests.append((req.full_url, json.loads(req.data), timeout))
        if self.error:
            raise self.error
        return io.BytesIO(json.dumps({"message": {"role": "assistant", "content": self.reply}}).encode())


@pytest.fixture
def fake(monkeypatch):
    f = FakeOllama()
    monkeypatch.setattr(assistant, "_OPENER", f)
    return f


def test_retrieval_picks_the_right_entries():
    qs = [it["question"] for it in assistant.top_entries(ITEMS, "What is a clip?")]
    assert qs[0] == "What is a clip? What is a moment?" and len(qs) <= 3
    assert assistant.top_entries(ITEMS, "How do I export to JSON or CSV?")[0]["question"] == "What does Export do?"
    assert assistant.top_entries(ITEMS, "Does anything leave this computer?")[0]["question"] == \
        "Does anything leave this computer?"
    assert assistant.top_entries(ITEMS, "zebra xylophone") == []


def test_request_goes_only_to_local_ollama(fake):
    answer, based = assistant.ask("What is a clip?", ITEMS, "qwen2.5:1.5b")
    assert answer == fake.reply and based[0] == "What is a clip? What is a moment?"
    [(url, body, timeout)] = fake.requests
    assert url == "http://127.0.0.1:11434/api/chat" and timeout <= 30
    assert body["model"] == "qwen2.5:1.5b" and body["stream"] is False
    assert body["options"] == {"temperature": 0.2, "num_predict": 120}
    for bad in ("http://example.org:11434/api/chat", "http://localhost.evil:11434/api/chat",
                "https://127.0.0.1:11434/api/chat", "http://127.0.0.1:8080/api/chat"):
        with pytest.raises(ValueError):
            assistant._post(bad, {})
    assert len(fake.requests) == 1


def test_prompt_has_only_help_text_and_the_question(fake):
    q = "How do I export? " * 30  # longer than the cap
    assistant.ask(q, ITEMS)
    msgs = fake.requests[0][1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == q.strip()[:300]
    system = msgs[0]["content"]
    entries = assistant.top_entries(ITEMS, q.strip()[:300])
    expected = "\n\n".join(f"Q: {it['question']}\nA: {it['answer']}" for it in entries)
    assert system == f"{assistant.SYSTEM}\n\nHelp entries:\n{expected}"
    assert assistant.DONT_KNOW in system and "ONLY" in system


def test_no_app_data_in_the_prompt(fake, tmp_path, monkeypatch):
    # app data lives under OUTPUT_DIR/DATA_DIR; none of it (or any path / video name) may reach the model
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "secret_outputs"))
    assistant.ask("What is the Scorecard?", ITEMS)
    sent = json.dumps(fake.requests[0][1])
    for leak in ("secret_outputs", str(tmp_path), ".mp4", ".parquet", ".duckdb", "segment_id", "video_id"):
        assert leak not in sent
    allowed = {assistant.SYSTEM, "What is the Scorecard?", *(it["question"] for it in ITEMS),
               *(it["answer"] for it in ITEMS)}
    content = "\n".join(m["content"] for m in fake.requests[0][1]["messages"])
    for chunk in content.replace("Help entries:", "").replace("Q: ", "\n").replace("A: ", "\n").split("\n"):
        assert not chunk.strip() or chunk.strip() in allowed


@pytest.mark.parametrize("error", [urllib.error.URLError(ConnectionRefusedError()),  # Ollama not running
                                   urllib.error.HTTPError(assistant.OLLAMA_URL, 404, "model not found", {}, None),
                                   TimeoutError()])
def test_not_available_message(monkeypatch, error):
    monkeypatch.setattr(assistant, "_OPENER", FakeOllama(error=error))
    assert assistant.ask("What is a clip?", ITEMS) == (assistant.UNAVAILABLE, [])
    assert "The questions below still work." in assistant.UNAVAILABLE


def test_no_match_or_empty_needs_no_model(fake):
    assert assistant.ask("zebra xylophone", ITEMS) == (assistant.DONT_KNOW, [])
    assert assistant.ask("   ", ITEMS) == ("", [])
    assert fake.requests == []


def test_answer_html_and_app_wiring():
    from tpa_index.app import build_app

    html = ui_html.help_answer_html("<b>x</b>", ["Q1?", "Q2?"])
    assert "&lt;b&gt;" in html and "Based on: Q1? · Q2?" in html
    assert ui_html.help_answer_html("", []) == ""
    by = {c["props"].get("elem_id"): c["props"] for c in build_app().config["components"]}
    assert {"help-ask", "help-ask-btn", "help-answer", "help-search", "help-list"} <= set(by)
    assert by["help-ask"]["max_length"] == 300


# ---- the panel on Search moments, Review queue and Blind labels -------------------------------------------------
def _components(demo):
    return {c["props"].get("elem_id"): c for c in demo.config["components"]}


def test_panel_on_every_tab_collapsed_and_wired_to_the_question_only():
    from tpa_index.app import build_app

    demo = build_app()
    by = _components(demo)
    for tab in ("search", "review", "blind"):
        assert by[f"{tab}-assistant"]["props"]["open"] is False  # collapsed by default
        assert by[f"{tab}-assistant"]["props"]["label"] == "Ask for help · runs on this computer"
        assert "tpa-assistant" in by[f"{tab}-assistant"]["props"]["elem_classes"]
        assert by[f"{tab}-ask"]["props"]["max_length"] == assistant.MAX_QUESTION
    for tab in ("search", "review", "blind", "help"):
        assert {f"{tab}-ask", f"{tab}-ask-btn", f"{tab}-answer"} <= set(by)
        ids = {b._id: getattr(b, "elem_id", None) for b in demo.blocks.values()}
        btn = next(i for i, e in ids.items() if e == f"{tab}-ask-btn")
        fns = [f for f in demo.fns.values() if any(t[0] == btn for t in f.targets)]
        assert fns and all([ids[i._id] for i in f.inputs] == [f"{tab}-ask"] for f in fns)  # no app state passed in
    # the review panel sits at the bottom of the review column, after every control
    order = [c["props"].get("elem_id") for c in demo.config["components"]]
    assert order.index("review-assistant") > max(order.index(f"review-{b}") for b in ("accept", "defer", "more", "next"))
    assert order.index("blind-assistant") > order.index("blind-save")


def test_tab_name_boosts_that_tabs_entries():
    q = "why is it hidden?"
    assert assistant.top_entries(ITEMS, q, tab="search")[0]["question"] == 'What does "staff-led moments hidden" mean?'
    assert assistant.top_entries(ITEMS, q, tab="blind")[0]["question"] == "Why are the AI's answers hidden in Blind labels?"
    assert assistant.top_entries(ITEMS, "what do the statuses mean", tab="review")[0]["question"] == \
        "What do the queue statuses mean?"
    assert assistant.top_entries(ITEMS, "zebra xylophone", tab="review") == []  # a boost never adds unrelated entries


def _allowed_lines(tab):
    return {assistant.SYSTEM, f"Current tab: {assistant.TAB_NAMES[tab]}", *(it["question"] for it in ITEMS),
            *(it["answer"] for it in ITEMS)}


def _only_help_text(body, question, tab):
    system, user = body["messages"]
    assert user == {"role": "user", "content": question}
    allowed = _allowed_lines(tab)
    for chunk in system["content"].replace("Help entries:", "").replace("Q: ", "\n").replace("A: ", "\n").split("\n"):
        assert not chunk.strip() or chunk.strip() in allowed, chunk


def test_blind_panel_never_sends_ai_suggestions_or_app_data(fake, tmp_path, monkeypatch):
    """With a full mock world loaded (moments, AI scores, blind queue, reviews), the Blind tab's panel sends only
    help.yaml text, the question and the tab name."""
    from tests.insights_fixtures import build_world
    from tpa_index.app import build_app
    from tpa_index.schema import read_official, read_reviews

    world = build_world(tmp_path, monkeypatch)
    demo = build_app(labels=world["labels"], pipeline=world["pipeline"])
    rv = demo.review
    moment = rv.open(rv.timeline("d01")[1], 0)[0]
    secrets = {rv.suggestion(moment), moment["moment_id"], moment["primary_label"], "mock-staff-a",
               *(m["moment_id"] for m in read_official("moments").to_pylist()),
               *(r["target_id"] for r in read_reviews().to_pylist())}
    secrets |= {f"{s:.3f}" for s in rv.scores.get(world["ids"]["d01"][0], {}).values()}

    ids = {b._id: getattr(b, "elem_id", None) for b in demo.blocks.values()}
    for tab in ("blind", "review", "search"):
        btn = next(i for i, e in ids.items() if e == f"{tab}-ask-btn")
        fn = next(f for f in demo.fns.values() if any(t[0] == btn for t in f.targets))
        question = "What does the AI suggest for this clip and what is the main label?"
        outputs = list(fn.fn(question))
        assert outputs[0] == ui_html.help_answer_html("Thinking…", [])
        body = fake.requests[-1][1]
        sent = json.dumps(body)
        for secret in secrets:
            if secret and len(secret) > 3 and secret not in " ".join(f"{it['question']} {it['answer']}" for it in ITEMS):
                assert secret not in sent, (tab, secret)
        _only_help_text(body, question, tab)
        assert f"Current tab: {assistant.TAB_NAMES[tab]}" in body["messages"][0]["content"]


def test_typing_in_the_panel_never_triggers_shortcuts():
    from tpa_index.app import PAGE_JS

    text_guard = PAGE_JS.index("t.tagName === 'TEXTAREA'")
    panel_guard = PAGE_JS.index("t.closest('.tpa-assistant')")
    first_key = min(PAGE_JS.index("#review-decide"), PAGE_JS.index("#blind-checklist"))
    assert text_guard < first_key and panel_guard < first_key  # both checks run before any shortcut
