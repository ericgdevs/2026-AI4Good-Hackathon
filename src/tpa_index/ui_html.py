"""006 UI: small HTML/SVG pieces for the Scorecard, Search and Review tabs. Every colour comes from config/theme.yaml;
every number and tag is passed in (tags verbatim). No external URLs, scripts or fonts."""
import math
from html import escape

from tpa_index.config import load_theme
from tpa_index.demo import PHASE_NAMES as _PHASES, indicator_name

MEASURES = {"youth_voice_minutes_per_hour": ("Youth voice time", "min per hour"),
            "solo_moments_minutes_per_hour": ("Solo moments", "min per hour"),
            "performance_share": ("Time spent performing", "share of youth time"),
            "positive_expression_rate": ("Positive expressions", "clips with faces (unvalidated)")}
WEIGHTS_NOTE = "Each measure counts equally (25%). Shares are out of 100; minutes per hour are out of 60."
INDEX_NOTE = ("A summary of the four measures on the right. It describes the footage, never a young person, and "
              "rests partly on unvalidated AI suggestions.")
APP_NAME, APP_TAG = "This Is ME", "· TPA"


def _p():
    return load_theme()


def phase_name(phase: str) -> str:
    return _PHASES.get(phase, phase)


def tag_dot(tag: str) -> str:
    """Dot colour for a tag (chart colours: accent = learned, teal = mixed, coral = unvalidated)."""
    p = _p()
    s = p["chart_series"]
    colour = (p["accent_deep"] if tag == "staff-verified" else s[1] if tag.startswith("mixed") else
              s[0] if tag == "model estimate (learned)" else s[2])
    return (f'<span class="tpa-tag"><span class="tpa-dot" style="background:{colour}"></span>'
            f'{escape(tag)}</span>')


def header() -> str:
    p = _p()
    return (f'<div class="tpa-topbar"><span class="tpa-appname">{APP_NAME} '
            f'<span style="font-weight:400;color:{p["header_text_muted"]}">{APP_TAG}</span></span>'
            f'<span class="tpa-topnote">Local only · no faces stored</span></div>')


def _fmt(v, unit) -> str:
    if v is None:
        return "n/a"
    return f"{100 * v:.0f}%" if unit == "share" else f"{v:.1f}"


def gauge(cmp: dict) -> str:
    """Half-circle Expression Index gauge: phase b large, a tick for phase a."""
    p = _p()
    a, b, va, vb = cmp["a"], cmp["b"], cmp["index_a"], cmp["index_b"]
    if vb is None:
        return (f'<div class="tpa-box tpa-pad"><p class="tpa-cap">TPA Expression Index</p>'
                f'<p class="tpa-muted">Not available for {escape(phase_name(b or ""))} in these results.</p></div>')
    arc = math.pi * 120
    dash = f"{arc * vb / 100:.1f} {arc:.1f}"
    tick = ""
    if va is not None:
        th = math.pi * (1 - va / 100)
        c, s = math.cos(th), math.sin(th)
        tick = (f'<line x1="{150 + 104 * c:.1f}" y1="{150 - 104 * s:.1f}" x2="{150 + 136 * c:.1f}" '
                f'y2="{150 - 136 * s:.1f}" stroke="{p["text"]}" stroke-width="3"></line>')
    change = ""
    if va is not None:
        d = round(vb) - round(va)
        change = f"{'+' if d >= 0 else '−'}{abs(d)} points from {escape(phase_name(a).lower())}. "
    legend_a = (f'<span class="tpa-legend"><span style="width:14px;height:3px;background:{p["text"]}"></span>'
                f'{escape(phase_name(a))}: <strong class="tpa-mono">{va:.0f}</strong></span>' if va is not None else "")
    return f"""<section class="tpa-box tpa-pad tpa-col" aria-label="Expression Index">
<div class="tpa-row-between"><p class="tpa-cap">TPA Expression Index</p>{tag_dot(cmp["index_tag"])}</div>
<div style="display:flex;justify-content:center">
<svg width="300" height="176" viewBox="0 0 300 176" role="img"
 aria-label="{escape(phase_name(b))} {vb:.0f} out of 100{f'; {escape(phase_name(a))} {va:.0f}' if va is not None else ''}">
<path d="M 30 150 A 120 120 0 0 1 270 150" fill="none" stroke="{p["track"]}" stroke-width="22"></path>
<path d="M 30 150 A 120 120 0 0 1 270 150" fill="none" stroke="{p["accent"]}" stroke-width="22"
 stroke-dasharray="{dash}"></path>{tick}
<text x="150" y="128" text-anchor="middle" class="tpa-mono" font-size="46" font-weight="600"
 fill="{p["text"]}">{vb:.0f}</text>
<text x="150" y="152" text-anchor="middle" font-size="13" fill="{p["text_secondary"]}">{escape(phase_name(b).lower())} · out of 100</text>
<text x="22" y="172" class="tpa-mono" font-size="11" fill="{p["text_secondary"]}">0</text>
<text x="262" y="172" class="tpa-mono" font-size="11" fill="{p["text_secondary"]}">100</text>
</svg></div>
<div style="display:flex;justify-content:center;gap:24px;font-size:14px">{legend_a}
<span class="tpa-legend"><span style="width:14px;height:10px;border-radius:2px;background:{p["accent"]}"></span>
{escape(phase_name(b))}: <strong class="tpa-mono">{vb:.0f}</strong></span></div>
<p class="tpa-muted" style="text-align:center;margin:4px 0 0">{change}{INDEX_NOTE}
<br>Label: {escape(cmp["index_label"])}</p></section>"""


def drives(cmp: dict) -> str:
    """'What drives it': the headline measures for both phases, the ×change with a small bar, and the tag."""
    p = _p()
    a, b = phase_name(cmp["a"]), phase_name(cmp["b"])
    body = []
    for r in cmp["rows"]:
        title, unit = MEASURES.get(r["indicator"], (indicator_name(r["indicator"]), r["unit"]))
        if r["ratio"] is not None:
            change, width = f"×{r['ratio']:.1f}", max(0, min(100, round((r["ratio"] - 1) / 5.5 * 100)))
        else:
            change, width = ("new" if r["new"] else "n/a"), (100 if r["new"] else 0)
        ea = _fmt(r["a"], r["unit"]) if r["a"] is not None else escape(r["a_status"] or "n/a")
        eb = _fmt(r["b"], r["unit"]) if r["b"] is not None else escape(r["b_status"] or "n/a")
        body.append(f"""<tr><td><span style="font-weight:600">{escape(title)}</span><br>
<span class="tpa-small">{escape(unit)}</span></td><td class="tpa-mono" style="color:{p["text_secondary"]}">{ea}</td>
<td class="tpa-mono" style="font-weight:600">{eb}</td>
<td><div style="display:flex;align-items:center;gap:8px"><div class="tpa-bar"><div style="width:{width}%;background:{p["accent"]}"></div></div>
<span class="tpa-mono" style="font-size:13px">{change}</span></div></td><td>{tag_dot(r["tag"])}</td></tr>""")
    return f"""<section class="tpa-box" aria-label="What drives the index" style="padding:4px 0 0">
<div class="tpa-row-between" style="padding:20px 20px 8px"><p class="tpa-cap">What drives it</p>
<span class="tpa-small">{WEIGHTS_NOTE}</span></div><div style="overflow-x:auto">
<table class="tpa-table"><thead><tr><th>Measure</th><th>{escape(a)}</th><th>{escape(b)}</th><th>Change</th>
<th>Source</th></tr></thead><tbody>{"".join(body)}</tbody></table></div></section>"""


def facts(items: list[dict]) -> str:
    cells = "".join(f'<div class="tpa-fact"><p class="tpa-cap" style="margin-bottom:8px">{escape(f["cap"])}</p>'
                    f'<p class="tpa-mono tpa-big">{escape(str(f["value"]))}</p>'
                    f'<p class="tpa-muted" style="margin:4px 0 0">{escape(f["note"])}</p></div>' for f in items)
    return f'<section class="tpa-box tpa-facts" aria-label="Confidence in these numbers">{cells}</section>'


def strip(count: int, marks: list[float], hidden: int = 0) -> str:
    """'N moments match' and where they fall along the shown videos, early -> showcase."""
    p = _p()
    ticks = "".join(f'<div style="position:absolute;top:4px;bottom:4px;width:4px;border-radius:2px;'
                    f'background:{p["accent"]};left:calc({100 * m:.2f}% - 2px)"></div>' for m in marks)
    more = f" · {hidden} staff-led moment{'s' if hidden != 1 else ''} hidden" if hidden else ""
    return f"""<section class="tpa-box tpa-pad tpa-col" aria-label="Where the results are">
<div class="tpa-row-between" style="font-size:14px"><span><strong class="tpa-mono">{count}</strong> moment{'s' if count != 1 else ''} match{'' if count != 1 else 'es'}{more}</span>
<span class="tpa-muted">across the shown videos, early → showcase</span></div>
<div style="position:relative;height:28px;border-radius:4px;background:{p["track"]}" role="img"
 aria-label="Positions of matching moments">{ticks}</div>
<div class="tpa-row-between tpa-small"><span>Early sessions</span><span>Mid program</span><span>Showcase</span></div>
</section>"""


def noticed(info: dict | None) -> str:
    """'AI noticed · suggestions, unvalidated' for one moment (counts and cues only)."""
    if not info:
        return ('<div class="tpa-box tpa-pad"><p class="tpa-label">AI noticed <span style="font-weight:500">'
                '· suggestions, unvalidated</span></p><p class="tpa-muted">Pick a moment.</p></div>')
    n = info["people"]
    people = "People in view: not available" if n is None else f"<strong>{n}</strong> {'person' if n == 1 else 'people'} in view"
    sounds = ("Sounds: not available" if info["sounds"] is None else
              ", ".join(info["sounds"]) if info["sounds"] else "No clear sounds")
    speech = ("Speech: not available" if info["speech"] is None else
              "Someone speaking" if info["speech"] else "No one speaking")
    expr = ("Expression: not available" if info["expression"] is None else
            "No faces in view" if info["expression"] == "no faces" else
            f"Expression estimate: {escape(info['expression'])} (unvalidated)")
    return f"""<div class="tpa-box tpa-pad tpa-col"><p class="tpa-label">AI noticed <span style="font-weight:500">
· suggestions, unvalidated</span></p><div class="tpa-noticed"><span>{people}</span><span>{escape(sounds)}</span>
<span>{speech}</span><span>{expr}</span></div></div>"""


# ---- Process a video ----
STATUS_TEXT = {"waiting": "Waiting", "running": "Running", "done": "Done", "failed": "Did not finish",
               "skipped": "Skipped"}
SOUND_NAMES = {"singing_voice": "Singing", "music": "Music", "speech": "Speech", "applause": "Applause"}


def _mins(secs: float) -> str:
    secs = max(0, round(secs))
    return f"{secs // 60} min {secs % 60:02d} s" if secs >= 60 else f"{secs} s"


def process_status(snap: dict | None) -> str:
    """Per-step status, clip counts, elapsed time and the estimate, from process.Job.snapshot()."""
    from tpa_index.process import STAGES

    p = _p()
    if not snap:
        return ('<div class="tpa-box tpa-pad"><p class="tpa-cap">Progress</p><p class="tpa-muted">Pick a video and '
                'press Process. The steps run one after another on this laptop.</p></div>')
    colour = {"waiting": p["track"], "running": p["accent"], "done": p["accent_deep"],
              "failed": p["chart_series"][2], "skipped": p["muted"]}
    n = snap["n_segments"]
    rows = []
    for stage, title, model in STAGES:
        st = snap["status"][stage]
        detail = ""
        if st == "running" and n:
            done = snap["counts"]["predict" if stage == "predict" else "frames"] if stage in (
                "predict", "people", "expression") else None
            detail = f"{min(done, n)} of {n} clips" if done is not None else "whole video"
        rows.append(f"""<tr><td><span class="tpa-tag"><span class="tpa-dot" style="background:{colour[st]}"></span>
{escape(STATUS_TEXT[st])}</span></td><td><span style="font-weight:600">{escape(title)}</span>
<span class="tpa-small"> · {escape(model)}</span></td><td class="tpa-mono tpa-small">{escape(detail)}</td></tr>""")
    est = snap["estimate_s"]
    if snap["done"]:
        timing = f"Finished in {_mins(snap['elapsed_s'])}."
    elif est is None:
        timing = f"Elapsed {_mins(snap['elapsed_s'])} · no estimate yet (first run on this laptop)."
    else:
        timing = (f"Elapsed {_mins(snap['elapsed_s'])} · about {_mins(max(est - snap['elapsed_s'], 0))} left "
                  "(estimate from earlier runs on this laptop; model loading not included).")
    return f"""<section class="tpa-box" aria-label="Progress" style="padding:4px 0 0">
<div class="tpa-row-between" style="padding:20px 20px 8px"><p class="tpa-cap">Progress · {escape(snap["video_id"])}</p>
<span class="tpa-small tpa-mono">{escape(timing)}</span></div>
<div style="overflow-x:auto"><table class="tpa-table"><tbody>{"".join(rows)}</tbody></table></div></section>"""


def preview_panel(view: dict | None, labels: dict) -> str:
    """The live preview: the latest blurred frame (or why none is shown) and the shown clip's AI suggestions."""
    from tpa_index.demo import display_label

    p = _p()
    if not view or not view["shown"]:
        return ('<div class="tpa-box tpa-pad"><p class="tpa-cap">Live preview</p><p class="tpa-muted">Sampled frames '
                'appear here while the models run, about once a second. Faces are blurred before anything is shown, '
                'and nothing is saved.</p></div>')
    name = lambda n: escape(display_label(n, labels))  # noqa: E731
    seg = view["shown"]
    image = (f'<img class="tpa-preview-img" src="{view["image"]}" alt="Sampled frame, faces blurred">'
             if view["image"] else f'<p class="tpa-muted">{escape(view["hidden"] or "")}</p>')
    people = ('<p class="tpa-small">Boxes: person detection (counts only)</p>' if view["image"] and seg["people_boxes"]
              else "")
    act = view.get("activity")
    activity = ("<span>AI suggestion (uncalibrated): not yet</span>" if not act else
                f'<span>AI suggestion (uncalibrated): <strong>{name(act)}</strong></span>'
                + "".join(f'<span class="tpa-small">next: {name(lab)} <span class="tpa-mono">{r:.2f}×</span> the top'
                          '</span>' for lab, r in view.get("others") or []))
    sounds = view.get("sounds")
    meter = ('<span class="tpa-small">not measured yet</span>' if sounds is None else "".join(
        f'<div class="tpa-meter"><span>{escape(SOUND_NAMES[k])}</span><div class="tpa-bar"><div style="width:'
        f'{100 * max(0.0, min(1.0, v)):.0f}%;background:{p["accent"]}"></div></div></div>'
        for k, v in sounds.items() if k in SOUND_NAMES))
    speech = {None: "not measured yet", True: "yes", False: "no"}[view.get("speech")]
    expr = view.get("expression")
    expression = ("not measured yet" if expr is None else "no faces in view" if expr == "no faces" else name(expr))
    return f"""<section class="tpa-box tpa-pad tpa-col" aria-label="Live preview">
<div class="tpa-row-between"><p class="tpa-cap">Live preview</p>
<span class="tpa-small tpa-mono">clip {int(seg["start_s"] // 60)}:{int(seg["start_s"] % 60):02d}</span></div>
<div class="tpa-preview"><div>{image}{people}</div><div class="tpa-col" style="gap:10px;font-size:14px">
{activity}<div><p class="tpa-label">Sounds</p>{meter}</div><span>Speech: {speech}</span>
<span>Expression (unvalidated): {expression}</span></div></div>
<p class="tpa-small">In memory only: frames are blurred inside face boxes and never saved.</p></section>"""


def page_css() -> str:
    """Draft 3 look from the theme: deep-purple top bar and tabs, white boxes with a 1px border and 6px radius."""
    p = _p()
    return f"""
h1, h2, h3 {{ color: {p['accent_deep']}; }}
.gradio-container {{ background: {p['background']}; }}
.tpa-mono, .tpa-mono * {{ font-family: var(--font-mono); }}
.tpa-topbar {{ background: {p['accent_deep']}; display: flex; flex-wrap: wrap; align-items: center;
  justify-content: space-between; gap: 0 36px; padding: 0 32px; border-radius: 6px 6px 0 0; }}
.tpa-appname {{ font-size: 17px; font-weight: 700; color: {p['surface']}; padding: 16px 0; }}
.tpa-topnote {{ font-size: 13px; color: {p['header_text_muted']}; }}
#app-tabs > .tab-wrapper {{ background: {p['accent_deep']}; padding: 0 32px; height: auto; margin-top: -16px;
  border-radius: 0 0 6px 6px; }}
#app-tabs > .tab-wrapper .tab-container {{ height: auto; gap: 28px; }}
#app-tabs > .tab-wrapper .tab-container::after {{ display: none; }}
#app-tabs > .tab-wrapper button {{ color: {p['header_text']}; font-size: 15px; font-weight: 500;
  padding: 18px 4px 15px; height: auto; background: transparent; }}
#app-tabs > .tab-wrapper button:hover {{ color: {p['surface']}; }}
#app-tabs > .tab-wrapper button.selected {{ color: {p['surface']}; }}
#app-tabs > .tab-wrapper button.selected::after {{ background-color: {p['surface']}; height: 3px; }}
.tpa-box {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: 6px; box-sizing: border-box;
  min-width: 0; color: {p['text']}; }}
.tpa-pad {{ padding: 24px; }}
.tpa-col {{ display: flex; flex-direction: column; gap: 12px; }}
.tpa-row-between {{ display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }}
.tpa-cap {{ font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: {p['text_secondary']}; margin: 0; }}
.tpa-label {{ font-size: 13px; font-weight: 600; color: {p['text_secondary']}; margin: 0; }}
.tpa-muted {{ font-size: 13px; line-height: 1.6; color: {p['text_secondary']}; margin: 0; }}
.tpa-small {{ font-size: 12px; color: {p['text_secondary']}; }}
.tpa-big {{ margin: 0; font-size: 28px; font-weight: 600; }}
.tpa-tag {{ display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: {p['text_secondary']}; }}
.tpa-dot {{ width: 8px; height: 8px; border-radius: 50%; flex: none; }}
.tpa-legend {{ display: inline-flex; align-items: center; gap: 8px; }}
.tpa-table {{ width: 100%; border-collapse: collapse; min-width: 520px; }}
.tpa-table th {{ font-size: 12px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
  color: {p['text_secondary']}; text-align: left; padding: 12px 16px; border-bottom: 1px solid {p['border']};
  white-space: nowrap; }}
.tpa-table td {{ font-size: 15px; padding: 16px; border-bottom: 1px solid {p['track']}; vertical-align: middle; }}
.tpa-bar {{ width: 70px; height: 8px; border-radius: 2px; background: {p['track']}; overflow: hidden; }}
.tpa-bar > div {{ height: 100%; }}
.tpa-facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
.tpa-fact {{ padding: 20px 24px; border-right: 1px solid {p['track']}; }}
.tpa-noticed {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px;
  font-size: 14px; }}
#scorecard-title h1 {{ margin: 0; font-size: 26px; font-weight: 700; color: {p['text']}; }}
#scorecard-subtitle p, #review-queue-head p {{ margin: 0; font-size: 14px; color: {p['text_secondary']}; }}
#review-queue label {{ border-radius: 6px; }}
#review-queue label.selected {{ background: {p['accent_soft']}; color: {p['accent_deep']}; }}
.tpa-quick button {{ border: 1px solid {p['border_strong']}; }}
.tpa-hint {{ font-size: 12px; color: {p['text_secondary']}; }}
.tpa-hint kbd {{ font-family: var(--font-mono); font-size: 11px; padding: 1px 6px; border-radius: 4px;
  background: {p['track']}; border: 1px solid {p['border']}; color: {p['text']}; }}
.tpa-preview {{ display: grid; grid-template-columns: minmax(0, 3fr) minmax(220px, 2fr); gap: 24px; }}
.tpa-preview-img {{ width: 100%; height: auto; border-radius: 6px; border: 1px solid {p['border']}; display: block; }}
.tpa-meter {{ display: grid; grid-template-columns: 80px 1fr; align-items: center; gap: 8px; font-size: 13px;
  margin-top: 6px; }}
.tpa-meter .tpa-bar {{ width: 100%; }}
#process-title h1 {{ margin: 0; font-size: 26px; font-weight: 700; color: {p['text']}; }}
#process-intro p {{ margin: 0; font-size: 14px; color: {p['text_secondary']}; }}
"""
