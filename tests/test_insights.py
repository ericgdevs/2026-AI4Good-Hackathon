"""006 US3: Insights tab reads 005 result folders by file only.

005 contract (specs/005-insights-eval/contracts/results.md), as merged:
- headline.csv columns: indicator, phase, value, unit, tag, verified_share, change_vs_early, status (status last).
- headline indicator IDs (indicators.HEADLINE): performance_share, solo_moments_minutes_per_hour,
  youth_voice_minutes_per_hour, positive_expression_rate.
- tags: staff-verified | model estimate (unvalidated) | model estimate (learned) | mixed (NN% staff-verified) | metric.
- schema_version 1; a folder is complete when manifest.json exists.
Fixture folders are made by 005's own `tpa-index insights` on its mock world, then copied and edited for variants."""
import csv
import json
import shutil
import time

import pytest

from tests.insights_fixtures import SECRET, SENTINEL, build_world
from tpa_index import cli, demo, report
from tpa_index.config import load_theme
from tpa_index.indicators import HEADLINE

TAGS = ["staff-verified", "model estimate (unvalidated)", "model estimate (learned)", "mixed (40% staff-verified)"]


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _evaluate_folder(results):
    """A complete evaluate folder with only what the Insights tab reads (ablation.csv in 005's columns)."""
    out = results / "evaluate-20991231T000000-bbbb"
    out.mkdir()
    cols = report.COLUMNS["ablation.csv"]
    vals = {"system": "zero-shot", "n_segments": "30", "primary_accuracy": "0.7", "primary_accuracy_ci_low": "0.5",
            "primary_accuracy_ci_high": "0.85", "macro_f1_activity": "0.6", "macro_f1_activity_ci_low": "0.4",
            "macro_f1_activity_ci_high": "0.75", "kappa_primary": "0.55"}
    _write(out / "ablation.csv", [{c: vals.get(c, "") for c in cols}])
    (out / "manifest.json").write_text(json.dumps({"schema_version": 1, "kind": "evaluate",
                                                   "created_at": "2099-12-31T00:00:00Z", "limitations": []}))
    return out


@pytest.fixture
def results(tmp_path, monkeypatch):
    world = build_world(tmp_path, monkeypatch)
    assert cli.main(["insights"]) == 0
    folder = report.latest_result("insights")
    return {"world": world, "folder": folder, "root": folder.parent}


def test_reads_real_005_output_on_mock(results):
    """T062: the real 005 output on mock data is read as written: every headline row with its tag verbatim."""
    ins = demo.InsightsController()
    assert ins.available and ins.latest_folder() == results["folder"]
    head = _rows(results["folder"] / "headline.csv")
    table = ins.table()
    assert len(table) == len(head)
    for row, h in zip(table, head):
        assert row[0] == demo.indicator_name(h["indicator"]) and row[1] == h["phase"] and row[3] == h["tag"]
    assert [r["indicator"] for r in head[::len(head) // len(HEADLINE)]] == list(HEADLINE)
    tb = next(r for r in _rows(results["folder"] / "time_breakdown.csv") if r["scope_type"] == "all")
    staff = (float(tb["staff_led_verified_s"]) + float(tb["staff_led_model_s"])) / 60
    assert f"staff-led time {staff:.1f} min" in ins.context()
    assert ins.index_table() and all(r[2] for r in ins.index_table())  # label column as 005 wrote it
    assert ins.ablation_table() == []  # no evaluate result yet


def test_insights_reads_newest_complete(results, tmp_path):
    root, folder = results["root"], results["folder"]
    newer = root / "insights-20991231T235959-aaaa"
    shutil.copytree(folder, newer)
    (newer / "manifest.json").unlink()  # incomplete: ignored
    _evaluate_folder(root)
    ins = demo.InsightsController()
    assert ins.latest_folder() == folder
    assert ins.ablation_table() == [["zero-shot", "30", "70% (95% CI 50–85%)", "60% (95% CI 40–75%)", "0.55",
                                     "metric"]]
    assert all(r[1] != "mid" for r in ins.table())

    # with_mid: mid rows in file order, every tag form, and a status row drawn as text
    mid = root / "insights-20991231T000001-cccc"
    shutil.copytree(folder, mid)
    head = _rows(mid / "headline.csv")
    out = []
    for i, ind in enumerate(HEADLINE):
        early, showcase = [dict(r) for r in head if r["indicator"] == ind]
        early["tag"], showcase["tag"] = TAGS[i], TAGS[(i + 1) % 4]
        m = {**early, "phase": "mid", "tag": TAGS[(i + 2) % 4]}
        if i == 3:
            m.update(value="", tag="", change_vs_early="", status="no videos in this phase")
        out += [early, m, showcase]
    _write(mid / "headline.csv", out)
    ins.load()
    assert ins.latest_folder() == mid
    table = ins.table()
    assert [r[3] for r in table] == [r["tag"] for r in out]
    assert {r[3] for r in table} >= set(TAGS)
    assert [r[1] for r in table[:3]] == ["early", "mid", "showcase"]
    assert table[10][2] == "no videos in this phase" and table[10][3] == ""
    figs = ins.figures()
    assert len(figs) == 4
    theme = load_theme()
    to_hex = lambda c: "#{:02X}{:02X}{:02X}".format(*(round(x * 255) for x in c[:3]))  # noqa: E731
    for i, fig in enumerate(figs):
        assert to_hex(fig.get_facecolor()) == theme["surface"]
        bars = fig.axes[0].patches
        assert {to_hex(b.get_facecolor()) for b in bars} <= {c.upper() for c in theme["chart_series"]}
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert len(bars) == (2 if i == 3 else 3)
        assert all(" · " in t for t in texts if t != "no videos in this phase")
    assert "no videos in this phase" in [t.get_text() for t in figs[3].axes[0].texts]

    m = json.loads((mid / "manifest.json").read_text())
    (mid / "manifest.json").write_text(json.dumps({**m, "schema_version": 2}))
    ins.load()
    assert ins.message() == f"{demo.EMPTY} (unsupported results format)" and ins.table() == [] and ins.figures() == []


def test_insights_empty_state(tmp_path, monkeypatch):
    from tpa_index.app import build_app

    build_world(tmp_path, monkeypatch)
    ins = demo.InsightsController()
    assert ins.message() == demo.EMPTY and ins.source_line() == "" and ins.table() == [] and ins.context() == ""
    by_id = {c["props"].get("elem_id"): c["props"] for c in build_app().config["components"]}
    assert by_id["insights-empty"]["value"] == demo.EMPTY


def test_insights_never_shows_transcript(results):
    """SC-004 (Insights)."""
    from tpa_index.app import build_app

    _evaluate_folder(results["root"])
    ins = demo.InsightsController()
    texts = [t.get_text() for f in ins.figures() for ax in f.axes for t in [*ax.texts, ax.title]]
    page = json.dumps([ins.source_line(), ins.table(), ins.context(), ins.index_table(), ins.ablation_table(), texts])
    page += json.dumps(build_app().config, default=str)
    assert SENTINEL not in page and SECRET not in page and "transcript_chars" not in page
    assert str(results["root"]) not in page  # the source line uses created_at, never the folder path


def test_judge_flow_is_fast(results):
    """SC-008: Timeline signals in under 3 s, Insights in under 2 s; first screens use plain names, no segment IDs."""
    t = time.perf_counter()
    tl = demo.DemoController()
    rows, _ = tl.segments_table("d01")
    assert time.perf_counter() - t < 3 and rows
    t = time.perf_counter()
    ins = demo.InsightsController()
    figs = ins.figures()
    assert time.perf_counter() - t < 2 and figs
    assert all(name.startswith("Video ") for name, _ in tl.video_choices()[0])
    ids = [sid for v in results["world"]["ids"].values() for sid in v]
    shown = json.dumps([rows, tl.moments_table("d01")[0], ins.table(), ins.index_table()])
    assert not any(sid in shown for sid in ids)
