"""006 Help assistant: grounded on help.yaml, Ollama on 127.0.0.1:11434 only. A fake opener; never the real one."""
import io
import json
import urllib.error

import pytest
from html import escape

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


def test_answer_html():
    html = ui_html.help_answer_html("<b>x</b>", ["Q1?", "Q2?"])
    assert "&lt;b&gt;" in html and "Based on: Q1? · Q2?" in html
    assert ui_html.help_answer_html("", []) == ""
    chat = ui_html.help_chat_html([("What is a <clip>?", "A slice.", ["What is a clip? What is a moment?"]),
                                   ("And Export?", assistant.UNAVAILABLE, [])])
    assert chat.index("What is a &lt;clip&gt;?") < chat.index("A slice.") < chat.index("And Export?")
    assert "Based on: What is a clip? What is a moment?" in chat and escape(assistant.UNAVAILABLE) in chat
    assert ui_html.HELP_EMPTY in ui_html.help_chat_html([])


# ---- the one global drawer -----------------------------------------------------------------------------------------
def _components(demo):
    return {c["props"].get("elem_id"): c for c in demo.config["components"]}


def _drawer_fn(demo):
    """The handlers on the Ask button and on Enter in the question box."""
    ids = {b._id: getattr(b, "elem_id", None) for b in demo.blocks.values()}
    btn = next(i for i, e in ids.items() if e == "help-ask-btn")
    box = next(i for i, e in ids.items() if e == "help-ask")
    fns = [f for f in demo.fns.values() if any((t[0] == btn and t[1] == "click") or (t[0] == box and t[1] == "submit")
                                               for t in f.targets)]
    return fns, ids


def test_one_global_drawer_and_no_in_tab_panels():
    from tpa_index.app import build_app

    demo = build_app()
    order = [c["props"].get("elem_id") for c in demo.config["components"]]
    for elem in ("help-fab", "help-drawer", "help-chat", "help-ask", "help-ask-btn", "help-drawer-head"):
        assert order.count(elem) == 1, elem
    assert not [e for e in order if e and e.endswith(("-assistant", "-answer"))]  # the in-tab panels are gone
    assert order.index("help-drawer") > order.index("app-footer")  # outside every tab: one drawer for the whole app
    by = _components(demo)
    assert "help-list" in by and "help-search" in by  # the Help tab keeps its searchable questions
    assert 'id="help-fab-btn"' in by["help-fab"]["props"]["value"] and "<svg" in by["help-fab"]["props"]["value"]
    assert 'id="help-close"' in by["help-drawer-head"]["props"]["value"]
    assert "Help assistant" in by["help-drawer-head"]["props"]["value"]
    assert by["help-ask"]["props"]["max_length"] == assistant.MAX_QUESTION
    fns, ids = _drawer_fn(demo)
    assert len(fns) == 2 and len({f.fn for f in fns}) == 1  # Ask and Enter, one handler
    for f in fns:
        assert [ids[i._id] for i in f.inputs][0] == "help-ask" and len(f.inputs) == 3  # question, tab name, turns


def test_drawer_opens_closes_and_escape_in_page_js():
    from tpa_index.app import PAGE_JS, page_css

    assert "#help-fab-btn" in PAGE_JS and "#help-close" in PAGE_JS and "tpa-help-open" in PAGE_JS
    assert "setHelp(!helpOpen())" in PAGE_JS and "#help-drawer.tpa-help-open" in PAGE_JS  # the edge tab toggles
    escape = PAGE_JS.index("e.key === 'Escape'")
    assert escape < PAGE_JS.index("t.tagName === 'TEXTAREA'")  # Escape closes even while typing in the drawer
    css = page_css()
    assert "position: fixed" in css.split("#help-drawer {", 1)[1].split("}", 1)[0]  # overlays, never reflows
    assert "#help-drawer.tpa-help-open { transform: none" in css and "translateX(100%)" in css
    assert "max-width: 640px" in css and "width: 100vw" in css  # full width on narrow windows


def test_tab_name_is_tracked_and_passed(fake):
    from tpa_index.app import build_app

    demo = build_app()
    ids = {b._id: getattr(b, "elem_id", None) for b in demo.blocks.values()}
    tabs = next(i for i, e in ids.items() if e == "app-tabs")
    on_tab = next(f for f in demo.fns.values() if any(t[0] == tabs and t[1] == "select" for t in f.targets))

    class Evt:
        def __init__(self, value):
            self.value = value

    names = {"Scorecard": "scorecard", "Search moments": "search", "Review queue": "review",
             "Process a video": "process", "Blind labels": "blind", "Export": "export", "Help": "help"}
    for label, name in names.items():
        assert on_tab.fn(Evt(label)) == name
    fn = _drawer_fn(demo)[0][0].fn
    for name in names.values():
        list(fn("What does Export do?", name, []))
        assert f"Current tab: {assistant.TAB_NAMES[name]}" in fake.requests[-1][1]["messages"][0]["content"]
    list(fn("What does Export do?", "not-a-tab", []))  # anything else is ignored, never sent
    assert "Current tab" not in fake.requests[-1][1]["messages"][0]["content"]


def test_each_question_answered_on_its_own(fake):
    from tpa_index.app import build_app

    fn = _drawer_fn(build_app())[0][0].fn
    first = list(fn("What is a clip?", "help", []))
    assert "Thinking…" in first[0][0] and first[0][2] == ""  # Thinking… shown, the box cleared
    chat, turns, _ = first[-1]
    assert len(turns) == 1 and fake.reply in chat
    list(fn("What does Export do?", "help", turns))
    msgs = fake.requests[-1][1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[1]["content"] == "What does Export do?"
    assert "What is a clip?" not in msgs[1]["content"] and fake.reply not in json.dumps(msgs)  # no history sent


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


def test_drawer_never_sends_ai_suggestions_or_app_data(fake, tmp_path, monkeypatch):
    """With a full mock world loaded (moments, AI scores, blind queue, reviews), the drawer sends only help.yaml text,
    the question and the tab name, on the Blind tab and everywhere else."""
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
    help_text = " ".join(f"{it['question']} {it['answer']}" for it in ITEMS)
    fn = _drawer_fn(demo)[0][0].fn
    turns = []
    for tab in ("blind", "review", "search", "scorecard"):
        question = "What does the AI suggest for this clip and what is the main label?"
        outputs = list(fn(question, tab, turns))
        turns = outputs[-1][1]
        body = fake.requests[-1][1]
        sent = json.dumps(body)
        for secret in secrets:
            if secret and len(secret) > 3 and secret not in help_text:
                assert secret not in sent, (tab, secret)
        _only_help_text(body, question, tab)


def test_keys_in_the_drawer_never_trigger_shortcuts():
    from tpa_index.app import PAGE_JS

    drawer_guard = PAGE_JS.index("t.closest('#help-drawer, #help-fab')")
    text_guard = PAGE_JS.index("t.tagName === 'TEXTAREA'")
    first_key = min(PAGE_JS.index("#review-decide"), PAGE_JS.index("#blind-checklist"))
    assert drawer_guard < first_key and text_guard < first_key  # A/E/R/D, down arrow and Blind keys come later
