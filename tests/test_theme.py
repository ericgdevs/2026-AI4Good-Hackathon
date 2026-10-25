"""006 US6: TPA palette in one config place, WCAG AA contrast, light theme and credit line."""
import re
import subprocess

import pytest
import yaml

from tpa_index import config
from tpa_index.config import ConfigError, contrast_ratio, load_theme
from tpa_index.paths import repo_root


def test_palette_meets_wcag_aa(tmp_path):
    """SC-010: every text/background pair is at least 4.5:1; teal and coral are never text; one config place."""
    p = load_theme()
    measured = {("text", "background"): 15.67, ("text_secondary", "background"): 4.55,
                ("accent", "background"): 5.35, ("accent_deep", "background"): 10.75,
                ("text", "surface"): 16.74, ("text_secondary", "surface"): 4.86,
                ("accent", "surface"): 5.71, ("accent_deep", "surface"): 11.49, ("surface", "accent"): 5.71}
    for (fg, bg), want in measured.items():
        assert contrast_ratio(p[fg], p[bg]) == pytest.approx(want, abs=0.005), (fg, bg)

    bad = {**p, "text_secondary": "#9A9CA3"}
    path = tmp_path / "theme.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError, match="text_secondary on background"):
        load_theme(path)

    from tpa_index.app import theme

    t = theme()
    text_slots = ("body_text_color", "body_text_color_subdued", "table_text_color", "block_title_text_color",
                  "block_label_text_color", "link_text_color", "button_primary_text_color")
    for slot in text_slots:
        assert getattr(t, slot).upper() not in {c.upper() for c in p["chart_series"][1:]}, slot

    tracked = subprocess.run(["git", "ls-files", "src", "config"], cwd=repo_root(), capture_output=True,
                             text=True).stdout.split()
    tracked.append("src/tpa_index/app.py")  # also while uncommitted
    palette = {p[k].upper() for k in config.THEME_COLOURS} | {c.upper() for c in p["chart_series"]}
    for f in set(tracked):
        if f == "config/theme.yaml" or not (repo_root() / f).exists():
            continue
        text = (repo_root() / f).read_text(errors="ignore").upper()
        hits = [h for h in palette if h in text]
        assert not hits, (f, hits)
    assert p["muted"].upper() == "#C9C4CF"  # neutral chart fill, kept at 005's value
    report = (repo_root() / "src/tpa_index/report.py").read_text()
    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|[\"']white[\"']", report)  # every chart colour from the theme


def test_light_theme_and_credit(tmp_path, monkeypatch, pipeline, labels):
    from tpa_index import schema
    from tpa_index.app import PAGE_JS, build_app, launch_kwargs, page_css, theme
    from tpa_index.mock import generate

    p = load_theme()
    t = theme()
    assert t.body_background_fill == p["background"] and t.body_text_color == p["text"]
    assert t.button_primary_background_fill == p["accent"] and t.block_title_text_color == p["accent_deep"]
    assert p["accent_deep"] in page_css()

    run_id = generate(seed=0, n_videos=2)
    path = tmp_path / "official.yaml"
    path.write_text(f"predict: [{run_id}]\n")
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)
    demo = build_app(pipeline=pipeline, labels=labels)
    comps = demo.config["components"]
    by_id = {c["props"].get("elem_id"): c["props"] for c in comps}
    assert by_id["app-credit-header"]["value"] == "Built for The Performance Academy"
    assert by_id["app-credit-footer"]["value"] == "Built for The Performance Academy"
    assert not [c for c in comps if c["type"] in ("image", "gallery")]
    kwargs = launch_kwargs(tmp_path, 7999)
    assert kwargs["js"] == PAGE_JS and "__theme" in PAGE_JS and "'light'" in PAGE_JS
    assert re.search(r"h1, h2, h3 \{ color: #[0-9A-Fa-f]{6}; \}", kwargs["css"])
