"""006: plain-English Word report of the insights results, on 005's mock world plus a real 005 insights run."""
import re

import docx
import pytest

from tests.insights_fixtures import SECRET, SENTINEL, build_world
from tests.test_insights import _evaluate_folder, _rows
from tpa_index import cli, report, word_report
from tpa_index.config import ConfigError
from tpa_index.indicators import LIMITATIONS

SECTIONS = ["What this report is", "Key findings at a glance",
            "What did young people do, from early sessions to the showcase?",
            "TPA Expression Index (optional example)", "How much of the footage was youth-led?",
            "How accurate is the AI?", "What the labels mean", "Limitations"]
JARGON = ["F1", "kappa", "macro", "zero-shot", "ablation", "threshold"]


def document_text(path) -> str:
    """Body paragraphs and table cells in document order, then the footer and file properties."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(path)
    parts = []
    for block in d.iter_inner_content():
        if isinstance(block, Paragraph):
            parts.append(block.text)
        elif isinstance(block, Table):
            parts += [p.text for row in block.rows for c in row.cells for p in c.paragraphs]
    parts += [p.text for s in d.sections for p in s.footer.paragraphs]
    props = d.core_properties
    parts += [props.author, props.last_modified_by, props.comments, props.title]
    return "\n".join(parts)


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch)
    assert cli.main(["insights"]) == 0
    return {**w, "folder": report.latest_result("insights")}


def test_report_sections_tags_and_findings(world, tmp_path):
    path = word_report.write_word_report(tmp_path / "r.docx")
    text = document_text(path)
    positions = [text.index(s) for s in SECTIONS]
    assert positions == sorted(positions)
    head = _rows(world["folder"] / "headline.csv")
    for r in head:
        assert f"{report.show(float(r['value']), r['unit'])} ({r['tag']})" in text
    assert report.INDEX_CAVEAT in text
    for line in LIMITATIONS:
        assert line in text
    assert f"Results: {world['folder'].name}" in text and "Guideline version 0." in text
    assert "python-docx" not in text

    findings = word_report.key_findings(head)
    assert 3 <= len(findings) <= 4 and all(f in text for f in findings)
    by = {(r["indicator"], r["phase"]): r for r in head}
    voice = (by[("youth_voice_minutes_per_hour", "early")], by[("youth_voice_minutes_per_hour", "showcase")])
    a, b = (float(r["value"]) for r in voice)
    expected = (f"Youth voice time {'rose' if b > a else 'fell' if b < a else 'stayed the same'} from {a:.1f} to "
                f"{b:.1f} minutes per hour of youth footage")
    assert any(f.startswith(expected) for f in findings)
    assert "Not available: no comparison" in text  # no evaluate folder yet


def test_report_with_evaluate_and_plain_words(world, tmp_path):
    _evaluate_folder(world["folder"].parent)
    text = document_text(word_report.write_word_report(tmp_path / "r.docx"))
    assert "The AI on its own: the main activity label matched the staff member's label about 70 out of 100 times " \
           "(range 50–85)." in text
    assert "AI suggestions are a starting point. Staff review confirms them" in text
    plain = text
    for line in LIMITATIONS:  # 005's limitations are quoted verbatim and may use technical words
        plain = plain.replace(line, "")
    for word in JARGON:
        assert not re.search(re.escape(word), plain, re.IGNORECASE), word
    assert SENTINEL not in text and SECRET not in text and "transcript" not in text.lower()


def test_report_needs_insights(tmp_path, monkeypatch):
    build_world(tmp_path, monkeypatch)
    with pytest.raises(ConfigError, match="not available"):
        word_report.write_word_report(tmp_path / "r.docx")


def test_key_findings_handle_missing_phases():
    rows = [{"indicator": "performance_share", "phase": "early", "value": "0.4", "unit": "share", "tag": "staff-verified",
             "status": "ok"},
            {"indicator": "performance_share", "phase": "showcase", "value": "0.65", "unit": "share",
             "tag": "mixed (40% staff-verified)", "status": "ok"},
            {"indicator": "solo_moments_minutes_per_hour", "phase": "showcase", "value": "", "unit": "",
             "tag": "", "status": "no videos in this phase"}]
    assert word_report.key_findings(rows) == [
        "Time spent performing rose from 40% to 65% of youth time, from early sessions to the showcase "
        "(early: staff-verified; showcase: mixed (40% staff-verified))."]


def test_sample_flag_adds_banner(world, tmp_path):
    from docx import Document

    normal = document_text(word_report.write_word_report(tmp_path / "r.docx"))
    assert word_report.SAMPLE_BANNER not in normal and "SAMPLE" not in normal
    path = word_report.write_word_report(tmp_path / "s.docx", sample=True)
    d = Document(path)
    assert d.paragraphs[1].text == word_report.SAMPLE_BANNER  # right under the title
    assert word_report.SAMPLE_BANNER in [p.text for p in d.sections[0].footer.paragraphs]
