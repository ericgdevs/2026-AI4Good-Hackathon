"""006 Help tab: static questions from config/help.yaml (no model, no network)."""
import json
import re

import pytest
import yaml

from tpa_index import ui_html
from tpa_index.config import ConfigError, load_help

TOPICS = ["Getting started", "The tabs", "Reviewing", "Understanding the numbers"]


def test_help_yaml_loads_grouped_by_topic():
    items = load_help()
    assert len(items) == 20 and list(dict.fromkeys(it["topic"] for it in items)) == TOPICS
    html = ui_html.help_html(items)
    assert html.count("<details") == 20 and all(f">{t}<" in html for t in TOPICS)
    assert "What is this app?" in html and "Everything runs on this computer." in html


def test_search_filters_question_and_answer_text():
    items = load_help()
    assert ui_html.filter_help(items, "") == items
    hits = ui_html.filter_help(items, "CUT-OFF")  # case-insensitive, in the question or the answer
    assert {it["question"] for it in hits} == {'What does "Needs review" mean?', 'What does "× above its cut-off" mean?'}
    assert [it["question"] for it in ui_html.filter_help(items, "json csv")] == ["What does Export do?"]
    assert "No questions match" in ui_html.help_html(items, "zzz-nothing")
    html = ui_html.help_html(items, "export")
    assert "What does Export do?" in html and "What is a clip?" not in html


def test_no_external_urls_and_bad_entries_refused(tmp_path):
    from tpa_index.app import build_app

    text = json.dumps(load_help())
    assert not re.search(r"https?://|www\.", text)
    for bad in ([{"topic": "T", "question": "Q", "answer": "see https://example.org"}],
                [{"topic": "T", "question": "Q"}], []):
        path = tmp_path / "help.yaml"
        path.write_text(yaml.safe_dump({"help": bad}))
        with pytest.raises(ConfigError):
            load_help(path)
    by = {c["props"].get("elem_id"): c["props"] for c in build_app().config["components"]}
    assert by["help-list"]["value"] == ui_html.help_html(load_help())
    assert not re.search(r"https?://|www\.", by["help-list"]["value"])
