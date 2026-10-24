"""006: a plain-English Word report of the newest 005 insights result, for TPA staff and funders.
Reads result files through demo.InsightsController (no parsing of its own). Aggregates only: no video names, no
clip-level rows, no transcript text, no individual scores; expressions are always unvalidated."""
from datetime import datetime, timezone
from pathlib import Path

from tpa_index.config import ConfigError, load_labels, load_theme
from tpa_index.demo import InsightsController, indicator_name

PHASE_NAMES = {"early": "Early sessions", "mid": "Mid program", "showcase": "Showcase"}
# plain subject for key findings, unit words after a number, one-sentence explanation
PLAIN = {
    "performance_share": ("Time spent performing", "of youth time",
                          "The share of youth time where young people were performing, for example singing, dancing, "
                          "acting or spoken word."),
    "solo_moments_minutes_per_hour": ("Solo performance time", "minutes per hour of youth footage",
                                      "Minutes per hour of youth footage where one young person was performing alone "
                                      "on stage."),
    "youth_voice_minutes_per_hour": ("Youth voice time", "minutes per hour of youth footage",
                                     "Minutes per hour of youth footage where young people performed with their "
                                     "voice: singing, speaking or spoken word."),
    "positive_expression_rate": ("Positive expressions (unvalidated)", "of clips with visible faces",
                                 "Among 8-second clips where faces were visible, the share where the AI suggested a "
                                 "positive expression such as a smile. This is an unvalidated AI suggestion about what "
                                 "the camera shows, never a measure of feelings."),
}
SYSTEMS = {"zero-shot": "The AI on its own", "zero-shot + signal rules": "The AI with extra sound and people checks"}
TAG_MEANINGS = [
    ("staff-verified", "Every 8-second clip behind this number was checked and confirmed by TPA staff."),
    ("model estimate (learned)", "No clip behind this number was checked by staff; it comes from an AI model trained "
                                 "on staff labels."),
    ("mixed (NN% staff-verified)", "Part of this number was confirmed by staff; NN tells you what percentage."),
    ("model estimate (unvalidated)", "No clip behind this number was checked by staff; it is an AI suggestion only."),
]
SAMPLE_BANNER = "SAMPLE — generated from mock test data. These numbers are not real program results."
INTRO = ("This report summarises what an AI-assisted review of TPA program footage found. It covers youth time only: "
         "time led by staff and idle or setup time are left out. It describes the footage as a whole, never any "
         "individual young person. The AI makes suggestions, and staff review confirms them; each number says how "
         "much of it staff have confirmed.")


def _num(v):
    return float(v) if v not in (None, "") else None


def _ok(r) -> bool:
    return r.get("status", "ok") in ("", "ok") and _num(r["value"]) is not None


def _plain_value(v: float, unit: str) -> str:
    return f"{100 * v:.0f}%" if unit == "share" else f"{v:.1f}"


def key_findings(headline: list[dict]) -> list[str]:
    """One sentence per headline indicator with both an early and a showcase value (at most four)."""
    out = []
    for ind in dict.fromkeys(r["indicator"] for r in headline):
        by = {r["phase"]: r for r in headline if r["indicator"] == ind}
        early, show = by.get("early"), by.get("showcase")
        if not (early and show and _ok(early) and _ok(show)):
            continue
        a, b = _num(early["value"]), _num(show["value"])
        subject, unit_words, _ = PLAIN.get(ind, (indicator_name(ind), "", ""))
        verb = "rose" if b > a else "fell" if b < a else "stayed the same"
        pa, pb = _plain_value(a, early["unit"]), _plain_value(b, show["unit"])
        tags = early["tag"] if early["tag"] == show["tag"] else f"early: {early['tag']}; showcase: {show['tag']}"
        out.append(f"{subject} {verb} from {pa} to {pb} {unit_words}, from early sessions to the showcase "
                   f"({tags}).".replace("  ", " "))
    return out[:4]


def _rgb(hex_colour: str):
    from docx.shared import RGBColor

    return RGBColor.from_string(hex_colour.lstrip("#").upper())


def _box(doc, title: str, lines: list[str], fill: str):
    """A one-cell shaded table: python-docx has no text box, so this is the usual way to draw one."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    cell = doc.add_table(rows=1, cols=1).cell(0, 0)
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:fill"), fill.lstrip("#"))
    cell._tc.get_or_add_tcPr().append(shading)
    cell.paragraphs[0].add_run(title).bold = True
    for line in lines:
        cell.add_paragraph(line, style="List Bullet")
    doc.add_paragraph()


def _table(doc, header: list[str], rows: list[list[str]]):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = text
    for row in rows:
        for cell, text in zip(t.add_row().cells, row):
            cell.text = text
    return t


def write_word_report(dest, root=None, sample: bool = False) -> Path:
    """Write the report for the newest complete insights folder (and evaluate folder, if any) to dest.
    sample=True marks a report built from mock data: a banner under the title and in the footer."""
    import docx
    from docx.shared import Pt

    from tpa_index.report import INDEX_CAVEAT, show

    ins = InsightsController(root)
    if not ins.available:
        raise ConfigError(ins.message())
    st, theme = ins.state, load_theme()
    doc = docx.Document()
    props = doc.core_properties  # the template's defaults name the library; replace them
    props.author, props.last_modified_by, props.comments = "TPA video index", "TPA video index", ""
    props.title = "Program insights report"
    normal = doc.styles["Normal"].font
    normal.size, normal.color.rgb = Pt(10.5), _rgb(theme["text"])
    for level in (0, 1, 2):
        doc.styles["Title" if level == 0 else f"Heading {level}"].font.color.rgb = _rgb(theme["accent_deep"])

    doc.add_heading("Program insights: youth activity in TPA footage", level=0)
    created = datetime.strptime(st["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    if sample:
        banner = doc.add_paragraph().add_run(SAMPLE_BANNER)
        banner.bold, banner.font.size, banner.font.color.rgb = True, Pt(13), _rgb(theme["accent_deep"])
    doc.add_paragraph(f"Results from {created:%d %B %Y}.").runs[0].italic = True
    doc.add_heading("What this report is", level=1)
    doc.add_paragraph(INTRO)

    head = st["headline"]
    findings = key_findings(head) or ["No early-to-showcase comparison is available in these results."]
    _box(doc, "Key findings at a glance", findings, theme["background"])

    doc.add_heading("What did young people do, from early sessions to the showcase?", level=1)
    phases = [p for p in ("early", "mid", "showcase") if any(r["phase"] == p for r in head)]
    rows = []
    for ind in dict.fromkeys(r["indicator"] for r in head):
        by = {r["phase"]: r for r in head if r["indicator"] == ind}
        cells = [f"{show(_num(by[p]['value']), by[p]['unit'])} ({by[p]['tag']})" if p in by and _ok(by[p])
                 else (by[p]["status"] if p in by else "") for p in phases]
        rows.append([PLAIN.get(ind, (indicator_name(ind),))[0], *cells])
    _table(doc, ["What we measured", *[PHASE_NAMES[p] for p in phases]], rows)
    for ind in dict.fromkeys(r["indicator"] for r in head):
        if ind in PLAIN:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(PLAIN[ind][0] + ": ").bold = True
            p.add_run(PLAIN[ind][2])

    doc.add_heading("TPA Expression Index (optional example)", level=1)
    doc.add_paragraph(INDEX_CAVEAT)
    if st["index"]:
        _table(doc, ["Program phase", "Index (0-100)", "Label", "Label of its least-confirmed part"],
               [[PHASE_NAMES.get(r["scope"], r["scope"]),
                 show(_num(r["value"]), "index_0_100") if r["value"] != "" else r["status"], r["label"], r["tag"]]
                for r in st["index"]])  # 005's order, never ranked
    else:
        doc.add_paragraph("Not available in these results.")

    doc.add_heading("How much of the footage was youth-led?", level=1)
    t = st["time"]
    if t:
        parts = [("Youth", "youth_s"), ("Staff-led, confirmed by staff", "staff_led_verified_s"),
                 ("Staff-led, AI estimate", "staff_led_model_s"), ("Idle or setup", "idle_s")]
        total = _num(t["total_s"]) or 0.0
        _table(doc, ["Time", "Minutes", "Share of all footage"],
               [[name, f"{_num(t[k]) / 60:.1f}", f"{100 * _num(t[k]) / total:.0f}%" if total else "n/a"]
                for name, k in parts])
        doc.add_paragraph("Staff-led and idle time are left out of every youth number in this report.")
    else:
        doc.add_paragraph("Not available in these results.")

    doc.add_heading("How accurate is the AI?", level=1)
    if st["ablation"]:
        n = st["ablation"][0]["n_segments"]
        doc.add_paragraph(f"A staff member labeled {n} short clips (8 seconds each) without seeing the AI's suggestions. "
                          "We compared each version of the AI with those labels.")
        for i, r in enumerate(st["ablation"], 1):
            name = SYSTEMS.get(r["system"]) or ("The AI trained on staff labels (tested only on clips it had not "
                                                 "seen)" if r["system"].startswith("learned") else f"AI version {i}")
            v, lo, hi = (_num(r[k]) for k in ("primary_accuracy", "primary_accuracy_ci_low", "primary_accuracy_ci_high"))
            if v is None:
                text = "this comparison could not be made."
            else:
                rng = f" (range {100 * lo:.0f}–{100 * hi:.0f})" if lo is not None and hi is not None else ""
                text = f"the main activity label matched the staff member's label about {100 * v:.0f} out of 100 times{rng}."
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(name + ": ").bold = True
            p.add_run(text)
        doc.add_paragraph("AI suggestions are a starting point. Staff review confirms them before they are used.")
    else:
        doc.add_paragraph("Not available: no comparison with staff labels has been run yet.")

    _box(doc, "What the labels mean", [f"{tag}: {meaning}" for tag, meaning in TAG_MEANINGS], theme["background"])

    doc.add_heading("Limitations", level=1)
    for line in st["limitations"]:
        doc.add_paragraph(line, style="List Bullet")

    footer = doc.sections[0].footer.paragraphs[0]
    # 005 insights manifests leave guideline_versions empty: then name the vocabulary in use, marked as such
    versions = ", ".join(st["guideline_versions"]) or f"{load_labels()['guideline_version']} (current vocabulary)"
    footer.text = (f"Guideline version {versions} · "
                   f"Results: {st['folder']} · Generated {datetime.now(timezone.utc):%Y-%m-%d}")
    footer.runs[0].font.size, footer.runs[0].font.color.rgb = Pt(8), _rgb(theme["text_secondary"])
    if sample:
        footer.insert_paragraph_before().add_run(SAMPLE_BANNER).bold = True

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc.save(dest)
    return dest
