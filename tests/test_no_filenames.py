"""006 SC-009: no video filename, .mp4 name or absolute data/output path in any 006 output, including the Word report
(mock data only)."""
import json
import os
import re

from tests.insights_fixtures import build_world
from tests.test_benchmark import VIDEO, fake_clock, fake_runner
from tests.test_word_report import document_text
from tpa_index import benchmark, cli, demo, export, word_report
from tpa_index.config import EXAMPLE_VIDEOS, load_videos


def test_no_filenames_anywhere(tmp_path, monkeypatch):
    from tpa_index.app import build_app
    from tpa_index.schema import read_table

    build_world(tmp_path, monkeypatch)
    assert cli.main(["insights"]) == 0
    _, (json_path, csv_path) = export.export_official()
    texts = {p.name: p.read_text() for p in json_path.parent.iterdir()}  # timestamped pair + latest pair

    results = benchmark.run_benchmark(VIDEO, runner=fake_runner(), clock=fake_clock())
    for p in benchmark.write_summary(benchmark.summarize(results)):
        texts[p.name] = p.read_text()

    tl, ins = demo.DemoController(), demo.InsightsController()
    out = [tl.video_choices(), tl.note]
    for _, vid in tl.video_choices()[0]:
        out += [tl.moments_table(vid), tl.segments_table(vid, show_staff=True)]
    out += [ins.source_line(), ins.message(), ins.table(), ins.context(), ins.index_table(), ins.ablation_table(),
            [t.get_text() for f in ins.figures() for ax in f.axes for t in [*ax.texts, ax.title]]]
    texts["controllers"] = json.dumps(out)
    texts["word report"] = document_text(word_report.write_word_report(tmp_path / "report.docx"))
    app = json.dumps(build_app().config, default=str).replace("world.mp4", "")  # Gradio's own sample video asset

    stored = {f for f in read_table("videos").column("filename").to_pylist() if f}  # mock world + benchmark runs
    names = stored | {v["filename"] for v in load_videos(EXAMPLE_VIDEOS)}
    roots = [os.environ["DATA_DIR"], os.environ["OUTPUT_DIR"]]
    assert len(texts) == 8 and stored
    for where, text in [*texts.items(), ("app", app)]:
        for needle in [*names, *(n.rsplit("/", 1)[-1] for n in names), *roots]:
            assert needle not in text, (where, needle)
        assert ".mp4" not in text, (where, re.findall(r"[\w./-]*\.mp4", text))
