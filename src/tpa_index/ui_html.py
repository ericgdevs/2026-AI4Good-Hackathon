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
HERO = "youth_voice_minutes_per_hour"
HERO_TEXT = "Minutes of young people speaking or singing, per hour of youth footage."
APP_NAME, APP_TAG = "This Is ME", "· TPA"


def _p():
    return load_theme()


def phase_name(phase: str) -> str:
    return _PHASES.get(phase, phase)


def tag_kind(tag: str) -> tuple[str, str]:
    """(short plain label, dot colour) for one of 005's tags (colours: deep accent = staff-verified, teal = mixed,
    accent = learned, coral = unvalidated)."""
    p = _p()
    s = p["chart_series"]
    if tag == "staff-verified":
        return "Staff-verified", p["accent_deep"]
    if tag.startswith("mixed"):
        return "Partly verified", s[1]
    if tag == "model estimate (learned)":
        return "AI estimate (trained)", s[0]
    return "AI estimate (unchecked)", s[2]


def tag_dot(tag: str) -> str:
    """A small dot and a short label; the exact 005 tag is the hover text."""
    label, colour = tag_kind(tag)
    return (f'<span class="tpa-tag" title="{escape(tag)}"><span class="tpa-dot" style="background:{colour}"></span>'
            f'{escape(label)}</span>')


def _fmt(v, unit) -> str:
    if v is None:
        return "n/a"
    return f"{100 * v:.0f}%" if unit == "share" else f"{v:.1f}"


def hero(cmp: dict) -> str:
    """Youth voice time: phase b large, phase a beside it, the change, and the exact tag."""
    p = _p()
    r = next((x for x in cmp["rows"] if x["indicator"] == HERO), None)
    title = MEASURES[HERO][0]
    if r is None or r["b"] is None:
        why = escape((r or {}).get("b_status") or "not in these results")
        return (f'<section class="tpa-box tpa-pad tpa-col"><p class="tpa-cap">{title}</p>'
                f'<p class="tpa-muted">Not available for {escape(phase_name(cmp["b"]))}: {why}.</p></section>')
    a, b = phase_name(cmp["a"]), phase_name(cmp["b"])
    change = (f"×{r['ratio']:.1f} since {escape(a.lower())}" if r["ratio"] is not None else
              f"new since {escape(a.lower())}" if r["new"] else "")
    before = (f'<span class="tpa-muted">{escape(a)}: <strong class="tpa-mono">{r["a"]:.1f}</strong> min per hour</span>'
              if r["a"] is not None else "")
    return f"""<section class="tpa-box tpa-pad tpa-col" aria-label="{title}">
<p class="tpa-cap">{title}</p>
<div class="tpa-hero"><span class="tpa-mono tpa-hero-num">{r["b"]:.1f}</span>
<span class="tpa-hero-unit">min per hour<br><span class="tpa-small">{escape(b)}</span></span></div>
<div class="tpa-row-between" style="align-items:center">{before}
<span class="tpa-mono" style="color:{p["accent_deep"]};font-weight:600">{change}</span></div>
<p class="tpa-muted" style="margin:0">{HERO_TEXT}</p>
<p class="tpa-small" style="margin:0">Source: {escape(r["tag"])}</p></section>"""


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
    return f"""<section class="tpa-box" aria-label="Youth indicators by phase" style="padding:4px 0 0">
<div class="tpa-row-between" style="padding:20px 20px 8px"><p class="tpa-cap">Youth indicators by phase</p></div>
<div style="overflow-x:auto">
<table class="tpa-table"><thead><tr><th>Measure</th><th>{escape(a)}</th><th>{escape(b)}</th><th>Change</th>
<th>Source</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>{legend(r["tag"] for r in cmp["rows"])}</section>"""


def legend(tags) -> str:
    """One line under the table: each dot and short label next to the exact tag it stands for."""
    parts = []
    for t in dict.fromkeys(t for t in tags if t):
        label, colour = tag_kind(t)
        parts.append(f'<span class="tpa-tag"><span class="tpa-dot" style="background:{colour}"></span>'
                     f'{escape(label)} = {escape(t)}</span>')
    return f'<div class="tpa-legend-line">{" ".join(parts)}</div>' if parts else ""


def facts(items: list[dict]) -> str:
    cells = "".join(f'<div class="tpa-fact"><p class="tpa-cap" style="margin-bottom:8px">{escape(f["cap"])}</p>'
                    f'<p class="tpa-mono tpa-big">{escape(str(f["value"]))}</p>'
                    f'<p class="tpa-muted" style="margin:4px 0 0">{escape(f["note"])}</p></div>' for f in items)
    return f'<section class="tpa-box tpa-facts" aria-label="Confidence in these numbers">{cells}</section>'


def staff_line(found: dict) -> str:
    """Always says what the staff-led toggle did."""
    n, hidden = found["staff"], found["hidden"]
    if not n:
        return f"No staff-led moments found by the AI in {'these videos' if found['all_videos'] else 'this video'}."
    plural = "s" if n != 1 else ""
    return f"{n} staff-led moment{plural} hidden." if hidden else f"{n} staff-led moment{plural} shown."


def strip(found: dict) -> str:
    """'N moments match', the staff-led line, and where matches fall along the videos in phase order."""
    p = _p()
    count = len(found["results"])
    ticks = "".join(f'<div style="position:absolute;top:4px;bottom:4px;width:4px;border-radius:2px;'
                    f'background:{p["accent"]};left:calc({100 * m:.2f}% - 2px)"></div>' for m in found["marks"])
    edges = "".join(f'<div style="position:absolute;top:0;bottom:0;width:1px;background:{p["border_strong"]};'
                    f'left:{100 * lo:.2f}%"></div>' for _, lo, _ in found["spans"][1:])
    labels = "".join(f'<span class="tpa-small" style="position:absolute;left:{100 * lo:.2f}%;'
                     f'width:{100 * (hi - lo):.2f}%;text-align:center;white-space:nowrap;overflow:hidden;'
                     f'text-overflow:ellipsis">{escape(phase_name(ph) if ph != "unknown" else "Phase not set")}</span>'
                     for ph, lo, hi in found["spans"])
    return f"""<section class="tpa-box tpa-pad tpa-col" aria-label="Where the results are">
<div class="tpa-row-between" style="font-size:14px"><span><strong class="tpa-mono">{count}</strong> moment{'s' if count != 1 else ''} match{'' if count != 1 else 'es'}</span>
<span class="tpa-muted">{escape(staff_line(found))}</span></div>
<div style="position:relative;height:28px;border-radius:4px;background:{p["track"]}" role="img"
 aria-label="Positions of matching moments, early sessions to showcase">{edges}{ticks}</div>
<div style="position:relative;height:18px">{labels}</div>
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


def filter_help(items: list[dict], query: str) -> list[dict]:
    """Entries whose question or answer contains every word of the query (case-insensitive)."""
    words = (query or "").lower().split()
    return [it for it in items if all(w in f"{it['question']} {it['answer']}".lower() for w in words)]


def help_html(items: list[dict], query: str = "") -> str:
    """The Help list grouped by topic (in file order), each question an expandable item. Plain text only."""
    found = filter_help(items, query)
    if not found:
        return '<div class="tpa-box tpa-pad"><p class="tpa-muted">No questions match that search.</p></div>'
    parts = []
    for topic in dict.fromkeys(it["topic"] for it in found):
        rows = "".join(f'<details class="tpa-qa"><summary>{escape(it["question"])}</summary>'
                       f'<p>{escape(it["answer"])}</p></details>' for it in found if it["topic"] == topic)
        parts.append(f'<section class="tpa-box tpa-pad tpa-col"><p class="tpa-cap">{escape(topic)}</p>{rows}'
                     '</section>')
    return f'<div class="tpa-col" style="gap:16px">{"".join(parts)}</div>'


HELP_FAB = ('<button id="help-fab-btn" type="button" aria-controls="help-drawer" aria-label="Open the help assistant">'
            '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/>'
            '<path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 2.5-3 4"/><path d="M12 17.5h.01"/></svg><span>Help</span></button>')
HELP_DRAWER_HEAD = ('<div class="tpa-drawer-head"><span><strong>Help assistant</strong> · runs on this computer</span>'
                    '<button id="help-close" type="button" aria-label="Close the help assistant">×</button></div>')
HELP_EMPTY = ("Ask anything about the app. Answers come only from the Help questions, and nothing from your videos or "
              "reviews is used.")


def help_chat_html(turns) -> str:
    """The drawer's conversation: each question, then its answer with the help.yaml questions it was based on."""
    if not turns:
        return f'<div class="tpa-chat"><p class="tpa-muted">{escape(HELP_EMPTY)}</p></div>'
    parts = []
    for question, answer, based in turns:
        parts.append(f'<div class="tpa-chat-q">{escape(question)}</div>')
        parts.append(help_answer_html(answer, list(based or [])))
    return f'<div class="tpa-chat" aria-live="polite">{"".join(parts)}</div>'


def help_answer_html(answer: str, based_on: list[str]) -> str:
    """The Help assistant's answer box, with the help.yaml questions it was grounded on."""
    if not answer:
        return ""
    based = (f'<p class="tpa-small">Based on: {escape(" · ".join(based_on))}</p>' if based_on else "")
    return f'<div class="tpa-box tpa-pad tpa-col" aria-live="polite"><p>{escape(answer)}</p>{based}</div>'


def page_css() -> str:
    """Draft 3 look from the theme: deep-purple top bar and tabs, white boxes with a 1px border and 6px radius."""
    p = _p()
    return f"""
h1, h2, h3 {{ color: {p['accent_deep']}; }}
.gradio-container {{ background: {p['background']}; }}
.tpa-mono, .tpa-mono * {{ font-family: var(--font-mono); }}
/* page: full width, content centred at 1200px like the mockups */
.gradio-container {{ max-width: none !important; padding: 0 !important; margin: 0 !important; }}
.gradio-container main, .gradio-container .main {{ padding: 0 !important; max-width: none !important; }}
.gradio-container .contain > div {{ gap: 0; }}
/* one full-width top bar: the tab bar, with the app name before the tabs and the note pushed right */
#app-tabs {{ gap: 0; }}
#app-tabs > .tab-wrapper {{ background: {p['accent_deep']}; height: auto; margin: 0; padding: 0 max(32px, calc((100% - 1136px) / 2));
  display: flex; flex-wrap: wrap; align-items: center; column-gap: 36px; border-radius: 0; }}
#app-tabs > .tab-wrapper::before {{ content: "{APP_NAME} {APP_TAG}"; color: {p['surface']}; font-size: 17px;
  font-weight: 700; padding: 16px 0; white-space: nowrap; }}
#app-tabs > .tab-wrapper::after {{ content: "Local only · no faces stored"; color: {p['header_text_muted']};
  font-size: 13px; margin-left: auto; padding: 8px 0; white-space: nowrap; }}
#app-tabs > .tab-wrapper .tab-container {{ width: auto; flex: 0 1 auto; height: auto; gap: 0 28px; flex-wrap: wrap;
  overflow: visible; }}
#app-tabs > .tab-wrapper .tab-container::after {{ display: none; }}
#app-tabs > .tab-wrapper button {{ color: {p['header_text']}; font-size: 15px; font-weight: 500;
  padding: 18px 4px 15px; height: auto; background: transparent; }}
#app-tabs > .tab-wrapper button:hover {{ color: {p['surface']}; }}
#app-tabs > .tab-wrapper button.selected {{ color: {p['surface']}; }}
#app-tabs > .tab-wrapper button.selected::after {{ background-color: {p['surface']}; height: 3px; }}
#app-tabs > .tabitem {{ max-width: 1200px; margin: 0 auto; padding: 32px 32px 24px; box-sizing: border-box;
  border: 0; background: transparent; }}
#app-footer {{ max-width: 1200px; margin: 0 auto; padding: 8px 32px 32px; box-sizing: border-box;
  align-items: center; }}
#app-footer p {{ font-size: 12px; color: {p['text_secondary']}; margin: 0; }}
/* our HTML pieces draw their own boxes: no extra block padding or border around them */
#scorecard-hero, #scorecard-drives, #scorecard-facts, #search-strip, #review-noticed, #search-try,
#review-hint, #process-status, #process-preview {{ padding: 0 !important; border: 0 !important;
  background: transparent !important; }}
#scorecard-hero > div, #scorecard-drives > div, #scorecard-facts > div, #search-strip > div,
#review-noticed > div, #process-status > div, #process-preview > div {{ padding: 0 !important; }}
#scorecard-hero, #scorecard-drives, #scorecard-hero > div, #scorecard-drives > div, #scorecard-hero .prose,
#scorecard-drives .prose, #scorecard-hero .tpa-box, #scorecard-drives .tpa-box {{ height: 100%; }}
.tpa-table th, .tpa-table td {{ border-left: 0 !important; border-right: 0 !important; border-top: 0 !important;
  background: transparent !important; }}
.tpa-table, .tpa-table tr, .tpa-table thead, .tpa-table tbody {{ border: 0 !important; margin: 0 !important;
  box-shadow: none !important; }}
#search-results {{ --font-mono: var(--font); }}
.tpa-quick {{ align-items: center; gap: 6px; flex-wrap: wrap; }}
#search-try {{ flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; margin-right: 6px; }}
.tpa-pills {{ gap: 6px; flex-wrap: wrap; }}
.tpa-pills button, .tpa-quick button {{ background: {p['surface']}; border: 1px solid {p['border_strong']};
  font-size: 14px; font-weight: 500; padding: 8px 12px; flex: 0 0 auto !important; width: auto !important;
  min-width: max-content !important; white-space: nowrap; min-height: 40px; }}
.tpa-quick, .tpa-pills {{ justify-content: flex-start; }}
#search-try p {{ margin: 0; }}
#review-next {{ align-self: center; max-height: 44px; }}
.tpa-decide {{ gap: 6px; flex-wrap: nowrap; }}
.tpa-decide button {{ flex: 1 1 0; min-height: 48px; }}
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
.tpa-table {{ width: 100%; border-collapse: collapse; }}
.tpa-table th {{ font-size: 12px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
  color: {p['text_secondary']}; text-align: left; padding: 12px; border-bottom: 1px solid {p['border']}; }}
.tpa-table td {{ font-size: 15px; padding: 14px 12px; border-bottom: 1px solid {p['track']}; vertical-align: middle; }}
.tpa-table td:last-child {{ width: 26%; }}
.tpa-table td:last-child .tpa-tag {{ align-items: center; }}
.tpa-bar {{ width: 70px; height: 8px; border-radius: 2px; background: {p['track']}; overflow: hidden; }}
.tpa-bar > div {{ height: 100%; }}
.tpa-hero {{ display: flex; align-items: baseline; gap: 12px; margin: 8px 0; }}
.tpa-hero-num {{ font-size: 64px; font-weight: 600; line-height: 1; color: {p['text']}; }}
.tpa-hero-unit {{ font-size: 15px; color: {p['text']}; }}
.tpa-legend-line {{ display: flex; flex-wrap: wrap; gap: 6px 18px; padding: 12px 16px 16px; }}
.tpa-facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
.tpa-fact {{ padding: 20px 24px; border-right: 1px solid {p['track']}; }}
.tpa-noticed {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px;
  font-size: 14px; }}
#scorecard-title h1 {{ margin: 0; font-size: 26px; font-weight: 700; color: {p['text']}; }}
#scorecard-subtitle p, #review-queue-head p {{ margin: 0; font-size: 14px; color: {p['text_secondary']}; }}
/* review queue: its own scrolling box, capped to the window; one line per item plus a smaller status line */
#review-queue-box {{ height: min(calc(100vh - 220px), 760px); overflow-y: auto; border: 1px solid {p['border']};
  border-radius: 6px; background: {p['surface']}; padding: 6px; flex: none; }}
#review-queue-box:not(:has(label)) {{ display: none; }}
#review-queue-box .form {{ background: transparent !important; border: 0 !important; box-shadow: none !important; }}
#review-queue {{ border: 0 !important; padding: 0 !important; background: transparent !important; }}
#review-queue .wrap {{ flex-direction: column; gap: 4px; }}
#review-queue label {{ width: 100%; box-sizing: border-box; border-radius: 6px; padding: 8px 10px; margin: 0;
  border: 0 !important; background: transparent !important; box-shadow: none !important; align-items: flex-start; }}
#review-queue label span {{ display: block; min-width: 0; white-space: pre; overflow: hidden; text-overflow: ellipsis;
  font-size: 12px; line-height: 1.45; color: {p['text_secondary']}; }}
#review-queue label span::first-line {{ font-size: 14px; font-weight: 600; color: {p['text']}; }}
#review-queue label.selected {{ background: {p['accent_soft']} !important; }}
#review-queue label.selected span::first-line {{ color: {p['accent_deep']}; }}
#review-queue-head, #review-queue-head > div {{ padding: 0 !important; border: 0 !important;
  background: transparent !important; }}
.tpa-qhead {{ display: flex; gap: 4px; font-size: 14px; white-space: nowrap; min-width: 0; color: {p['text']}; }}
.tpa-qname {{ overflow: hidden; text-overflow: ellipsis; min-width: 0; }}
.tpa-qcount {{ flex: none; color: {p['text_secondary']}; }}
#review-queue-empty p {{ font-size: 14px; color: {p['text_secondary']}; margin: 0; }}
#review-lock p, #search-lock p {{ font-size: 12px; color: {p['text_secondary']}; margin: 0; }}
#review-suggest p {{ font-size: 18px; margin: 0; }}
.tpa-hint {{ font-size: 12px; color: {p['text_secondary']}; }}
.tpa-hint kbd {{ font-family: var(--font-mono); font-size: 11px; padding: 1px 6px; border-radius: 4px;
  background: {p['track']}; border: 1px solid {p['border']}; color: {p['text']}; }}
.tpa-preview {{ display: grid; grid-template-columns: minmax(0, 3fr) minmax(220px, 2fr); gap: 24px; }}
.tpa-preview-img {{ width: 100%; height: auto; border-radius: 6px; border: 1px solid {p['border']}; display: block; }}
.tpa-meter {{ display: grid; grid-template-columns: 80px 1fr; align-items: center; gap: 8px; font-size: 13px;
  margin-top: 6px; }}
.tpa-meter .tpa-bar {{ width: 100%; }}
#process-title h1, #help-title h1 {{ margin: 0; font-size: 26px; font-weight: 700; color: {p['text']}; }}
.tpa-qa {{ border-top: 1px solid {p['track']}; padding: 10px 0; }}
.tpa-qa:first-of-type {{ border-top: 0; }}
.tpa-qa summary {{ cursor: pointer; font-size: 15px; font-weight: 600; color: {p['text']}; }}
.tpa-qa p {{ margin: 8px 0 0; font-size: 14px; line-height: 1.6; color: {p['text']}; }}
#help-list {{ padding: 0 !important; border: 0 !important; background: transparent !important; }}
/* help: an edge tab on every page and a slide-out drawer that overlays the page (layout underneath never moves) */
#help-fab {{ position: fixed !important; right: 0; top: 50%; transform: translateY(-50%); z-index: 1000;
  width: auto !important; min-width: 0 !important; padding: 0 !important; border: 0 !important;
  background: transparent !important; transition: right .2s ease; }}
#help-fab-btn {{ display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 12px 10px;
  background: {p['surface']}; color: {p['accent_deep']}; border: 1px solid {p['border']}; border-right: 0;
  border-radius: 8px 0 0 8px; box-shadow: -2px 2px 8px rgba(0, 0, 0, .08); cursor: pointer; font-size: 12px;
  font-weight: 600; }}
#help-fab-btn:hover {{ background: {p['accent_soft']}; }}
#help-drawer {{ position: fixed !important; top: 0; right: 0; height: 100vh; width: 400px; max-width: 100vw;
  z-index: 1001; margin: 0 !important; padding: 0 !important; gap: 0 !important; background: {p['surface']};
  border-left: 1px solid {p['border']}; box-shadow: -8px 0 24px rgba(0, 0, 0, .10); display: flex;
  flex-direction: column; transform: translateX(100%); visibility: hidden;
  transition: transform .2s ease, visibility .2s; }}
#help-drawer.tpa-help-open {{ transform: none; visibility: visible; }}
#help-fab.tpa-help-open {{ right: 400px; }}
@media (max-width: 640px) {{ #help-drawer {{ width: 100vw; }} #help-fab.tpa-help-open {{ display: none; }} }}
#help-drawer-head, #help-chat {{ padding: 0 !important; border: 0 !important; background: transparent !important; }}
.tpa-drawer-head {{ display: flex; justify-content: space-between; align-items: center; padding: 16px 18px;
  border-bottom: 1px solid {p['border']}; font-size: 15px; color: {p['text']}; }}
.tpa-drawer-head strong {{ color: {p['accent_deep']}; }}
#help-close {{ background: transparent; border: 0; font-size: 26px; line-height: 1; cursor: pointer;
  color: {p['text_secondary']}; padding: 0 4px; }}
#help-chat {{ flex: 1 1 auto; overflow-y: auto; min-height: 0; }}
.tpa-chat {{ display: flex; flex-direction: column; gap: 10px; padding: 16px 18px; }}
.tpa-chat-q {{ align-self: flex-end; max-width: 85%; background: {p['accent_soft']}; color: {p['text']};
  padding: 8px 12px; border-radius: 10px 10px 2px 10px; font-size: 14px; }}
.tpa-chat .tpa-box {{ font-size: 14px; }}
#help-drawer-input .block, #help-drawer-input .form {{ border: 0 !important; box-shadow: none !important;
  background: transparent !important; padding: 0 !important; }}
#help-drawer-input {{ flex: none; align-items: center; padding: 12px 18px 16px !important; border-top: 1px solid {p['border']};
  background: {p['surface']}; margin: 0 !important; }}
#process-msg p:last-child {{ font-size: 12px; color: {p['text_secondary']}; }}
#process-intro p {{ margin: 0; font-size: 14px; color: {p['text_secondary']}; }}
"""
