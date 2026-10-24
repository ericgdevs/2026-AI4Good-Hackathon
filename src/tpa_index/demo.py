"""006 demo tabs: Timeline (moments plus per-segment 004 signals) and Insights (005 result folders).
Read-only. Segment classes come from 005's own chain (view -> rules -> indicators), never a restated rule. Rows are
built from an allow-list: no transcript text or counts, no filenames, no segment IDs."""
import re

from tpa_index import review
from tpa_index.app import _cut, _fmt_conf, _fmt_time
from tpa_index.config import ConfigError, load_labels, load_pipeline
from tpa_index.paths import redact

# the label names 005's staff-led rule reads (indicators.segment_table); checked at startup
RULE_LABELS = ("actor_participant", "actor_staff_facilitation", "instruction")
UNVALIDATED = "model estimate (unvalidated)"
LED_BY = {"youth": "Youth", "staff_verified": "Staff-led · staff-verified",
          "staff_model": f"Staff-led · {UNVALIDATED}", "idle": "Idle / setup"}
STAFF = ("staff_verified", "staff_model")
NA = "not available"
EXPRESSION_PREFIX = "AI suggestion, unvalidated: "
MOMENT_HEADERS = ["time", "primary", "other labels", "confidence", "review status"]
SEGMENT_HEADERS = ["time", "people (person detection, counts only)", "speech", "sounds", "expression", "led by"]
NOTE = "Signals are observable cues from AI models, unvalidated. Staff-led time is not counted in youth indicators."


def display_label(name: str, labels: dict) -> str:
    lab = next((lab for lab in labels["labels"] if lab["name"] == name), None)
    if lab and lab.get("display"):
        return lab["display"]
    text = name.replace("_", " ")
    return text[:1].upper() + text[1:]


def video_name(video: dict, shown: list[dict]) -> str:
    """`Video 4 · showcase`; kind is added only when two shown videos would share a name, the opaque ID last."""
    def base(v):
        m = re.search(r"(\d+)$", v["video_id"])
        phase = "phase not set" if v["phase"] in (None, "unknown") else v["phase"]
        return f"Video {int(m.group(1)) if m else v['video_id']} · {phase}"
    name = base(video)
    if sum(base(v) == name for v in shown) > 1:
        name += f" · {video['kind']}"
        if sum(f"{base(v)} · {v['kind']}" == name for v in shown) > 1:
            name += f" ({video['video_id']})"
    return name


def check_rule_labels(labels: dict) -> None:
    names = {lab["name"] for lab in labels["labels"]}
    missing = [n for n in RULE_LABELS if n not in names]
    if missing:
        raise ConfigError(f"labels: the staff-led rule needs {', '.join(missing)} (renamed or removed in labels.yaml)")


class DemoController:
    """Timeline handlers (plain methods, testable without a browser)."""

    def __init__(self, cutter=None, root=None, labels=None, pipeline=None):
        self.cutter, self.root = cutter, root
        self.labels = labels or load_labels()
        self.pipeline = pipeline or load_pipeline(labels=self.labels)
        check_rule_labels(self.labels)
        self.reload()

    def reload(self):
        from tpa_index.indicators import segment_table
        from tpa_index.rules import _missing, context, indicator_estimates
        from tpa_index.schema import read_official, read_reviews
        from tpa_index.view import main_model, segment_view, verified_labels

        self.moments = sorted(read_official("moments", root=self.root).to_pylist(),
                              key=lambda m: (m["video_id"], m["start_s"]))
        self.state = review.latest_state(read_reviews(self.root), "review")
        self.segments, self.note = [], ""
        if main_model(read_official("predictions", root=self.root)) is None:
            self.note = "Segment signals not available: no official predictions."
            return
        try:
            # zero-shot view: the insights folders hold aggregates only, and no model is trained while the app runs
            view = segment_view(self.pipeline, self.labels, self.root)
            verified = verified_labels(view, self.labels, self.pipeline, self.root)
            est = indicator_estimates(view, self.pipeline["insights"]["indicator_rules"], context(self.pipeline))
            table = segment_table(view, verified, est, self.labels)
            cls = dict(zip(table["segment_id"], table["cls"]))
        except ConfigError as e:
            self.note = redact(f"Segment signals not available: {e}")
            return
        keep = ("video_id", "moment_id", "start_s", "end_s", "person_count", "has_speech", "faces_detected",
                "audio_ok", "expr_top")  # allow-list: transcript counts and raw scores are never kept
        self.segments = [{**{k: None if _missing(s[k]) else s[k] for k in keep}, "cls": cls[s["segment_id"]]}
                         for s in view.to_dict("records")]  # pandas turns a missing signal into NaN

    def _allowed(self) -> dict[str, dict]:
        return {v["video_id"]: v for v in review.review_videos(self.root)}

    def video_choices(self) -> tuple[list[tuple[str, str]], str]:
        """Same eval lock and message as the Review tab: eval videos are not listed while it holds."""
        _, reason = review.eval_unlocked(self.root)
        shown = list(self._allowed().values())
        return [(video_name(v, shown), v["video_id"]) for v in shown], reason

    def moments_table(self, video_id) -> tuple[list[list], list[str]]:
        """(table rows, moment IDs in row order). The IDs stay in app state; they are never shown."""
        if video_id not in self._allowed():
            return [], []
        found = [m for m in self.moments if m["video_id"] == video_id]
        name = lambda n: display_label(n, self.labels)  # noqa: E731
        table = [[_fmt_time(m["start_s"], m["end_s"]), name(m["primary_label"]),
                  ", ".join(name(lab) for lab, role in review.ai_labels(m) if role != review.PRIMARY),
                  _fmt_conf(m["confidence"]), review.moment_status(m, self.state)] for m in found]
        return table, [m["moment_id"] for m in found]

    def _cells(self, s) -> list:
        audio = s["audio_ok"] or {}
        if all(v is None for v in audio.values()):
            sounds = NA
        else:
            sounds = ", ".join(display_label(lab, self.labels) for lab, ok in audio.items() if ok) or "none"
        faces = s["faces_detected"]
        expression = (NA if faces is None else "no faces" if faces == 0 else
                      "none" if s["expr_top"] is None else EXPRESSION_PREFIX + display_label(s["expr_top"], self.labels))
        speech = NA if s["has_speech"] is None else "Yes" if s["has_speech"] else "No"
        people = NA if s["person_count"] is None else str(int(s["person_count"]))
        return [_fmt_time(s["start_s"], s["end_s"]), people, speech, sounds, expression, LED_BY[s["cls"]]]

    def segments_table(self, video_id, moment_id=None, show_staff=False) -> tuple[list[list], int]:
        """(rows, number of staff-led rows hidden). Staff-led rows are hidden unless show_staff."""
        if video_id not in self._allowed():
            return [], 0
        segs = [s for s in self.segments if s["video_id"] == video_id and (moment_id is None or s["moment_id"] == moment_id)]
        hidden = 0 if show_staff else sum(s["cls"] in STAFF for s in segs)
        return [self._cells(s) for s in segs if show_staff or s["cls"] not in STAFF], hidden

    def clip(self, video_id, moment_id):
        m = next((m for m in self.moments if m["moment_id"] == moment_id), None)
        if m is None:
            return None, ""
        return _cut(self.cutter, self._allowed().get(video_id), m["start_s"], m["end_s"],
                    self.pipeline["app"]["padding_s"])


# ---- Insights tab: reads 005 result folders (contracts/results.md) by file only ----
HEADLINE_COLUMNS = ("indicator", "phase", "value", "unit", "tag", "verified_share", "change_vs_early")
TABLE_HEADERS = ["indicator", "phase", "value", "tag", "change vs early"]
INDEX_HEADERS = ["phase", "Expression Index (0-100)", "label", "tag (weakest component)"]
ABLATION_HEADERS = ["system", "segments", "primary accuracy (95% CI)", "macro-F1 activity (95% CI)", "kappa primary",
                    "tag"]
EMPTY = "Insights are not available yet. Run the insights step first."
PHASE_ORDER = ("early", "mid", "showcase")


def _read_csv(path) -> list[dict]:
    import csv

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(v):
    return float(v) if v not in (None, "") else None


def _pct_ci(r, key) -> str:
    v, lo, hi = _num(r[key]), _num(r[f"{key}_ci_low"]), _num(r[f"{key}_ci_high"])
    if v is None:
        return "undefined"
    return f"{100 * v:.0f}%" + (f" (95% CI {100 * lo:.0f}–{100 * hi:.0f}%)" if lo is not None and hi is not None else "")


def indicator_name(indicator: str) -> str:
    from tpa_index.indicators import HEADLINE
    from tpa_index.report import HEADLINE_NAMES

    text = dict(zip(HEADLINE, HEADLINE_NAMES)).get(indicator, indicator.replace("_", " "))
    return text[:1].upper() + text[1:]


class InsightsController:
    """Newest complete insights folder (manifest.json present), plus the newest complete evaluate folder for the
    ablation. Columns are selected by name; tags are passed through verbatim."""

    def __init__(self, root=None):
        self.root = root
        self.load()

    def latest_folder(self):
        from tpa_index.report import latest_result

        return latest_result("insights", self.root)

    def load(self) -> dict:
        import json

        from tpa_index.report import SCHEMA_VERSION, latest_result

        self.state = {"status": "not available"}
        folder = self.latest_folder()
        if folder is None:
            return self.state
        manifest = json.loads((folder / "manifest.json").read_text())
        head = _read_csv(folder / "headline.csv") if (folder / "headline.csv").exists() else []
        if manifest.get("schema_version") != SCHEMA_VERSION or not head or not set(HEADLINE_COLUMNS) <= set(head[0]):
            self.state = {"status": "unsupported"}
            return self.state
        extra = json.loads((folder / "indicators.json").read_text()) if (folder / "indicators.json").exists() else {}
        index = (_read_csv(folder / "expression_index.csv") if (folder / "expression_index.csv").exists() else [])
        ev = latest_result("evaluate", self.root)
        self.state = {
            "status": "available", "created_at": manifest["created_at"], "limitations": manifest["limitations"],
            "headline": head, "model_tag": extra.get("model_tag", UNVALIDATED),
            "time": next((r for r in _read_csv(folder / "time_breakdown.csv") if r["scope_type"] == "all"), None),
            "index": [r for r in index if r["scope_type"] == "phase" and r["component"] == "expression_index"],
            "ablation": _read_csv(ev / "ablation.csv") if ev and (ev / "ablation.csv").exists() else [],
        }
        return self.state

    @property
    def available(self) -> bool:
        return self.state["status"] == "available"

    def message(self) -> str:
        if self.state["status"] == "unsupported":
            return f"{EMPTY} (unsupported results format)"
        return "" if self.available else EMPTY

    def source_line(self) -> str:
        if not self.available:
            return ""
        from datetime import datetime

        t = datetime.strptime(self.state["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        return f"Results from insights run of {t:%Y-%m-%d %H:%M} UTC"

    def table(self) -> list[list]:
        from tpa_index.report import show, show_change

        rows = []
        for r in self.state.get("headline", []):
            ok = r.get("status", "ok") in ("", "ok")
            rows.append([indicator_name(r["indicator"]), r["phase"],
                         show(_num(r["value"]), r["unit"]) if ok else r["status"], r["tag"],
                         show_change(_num(r["change_vs_early"]), r["unit"]) if ok else ""])
        return rows

    def figures(self) -> list:
        """One figure per headline indicator in file order: phases early, mid (only if present), showcase."""
        from matplotlib.figure import Figure

        from tpa_index.report import PALETTE, TEXT, _plt, show

        _plt()  # Agg backend and the theme's chart rcParams (white background, theme text colour)
        figs = []
        for ind in dict.fromkeys(r["indicator"] for r in self.state.get("headline", [])):
            rows = sorted((r for r in self.state["headline"] if r["indicator"] == ind),
                          key=lambda r: PHASE_ORDER.index(r["phase"]) if r["phase"] in PHASE_ORDER else 9)
            fig = Figure(figsize=(5, 3.2))
            ax = fig.add_subplot()
            top = max([_num(r["value"]) or 0 for r in rows] + [1e-9])
            for i, r in enumerate(rows):
                v = _num(r["value"])
                if r.get("status", "ok") not in ("", "ok") or v is None:
                    ax.text(i, 0, r["status"] or "n/a", ha="center", va="bottom", color=TEXT, fontsize=8)
                    continue
                ax.bar(i, v, color=PALETTE[i % len(PALETTE)])
                ax.text(i, v + 0.02 * top, f"{show(v, r['unit'])} · {r['tag']}", ha="center", va="bottom",
                        color=TEXT, fontsize=7, wrap=True)
            ax.set_xticks(range(len(rows)), [r["phase"] for r in rows])
            ax.set_ylim(0, top * 1.35)
            ax.set_title(indicator_name(ind))
            fig.tight_layout()
            figs.append(fig)
        return figs

    def context(self) -> str:
        if not self.available:
            return ""
        lines = []
        t = self.state["time"]
        if t:
            v, m, idle = (_num(t[k]) / 60 for k in ("staff_led_verified_s", "staff_led_model_s", "idle_s"))
            lines.append(f"**Excluded from youth indicators** (all videos): staff-led time {v + m:.1f} min "
                         f"({v:.1f} min staff-verified, {m:.1f} min {self.state['model_tag']}); idle / setup time "
                         f"{idle:.1f} min.")
        lines += ["", "**Limitations**", "", *[f"- {x}" for x in self.state["limitations"]]]
        return "\n".join(lines)

    def index_table(self) -> list[list]:
        """The optional Expression Index per phase, in 005's order (never ranked). Empty if the file is absent."""
        from tpa_index.report import show

        return [[r["scope"], show(_num(r["value"]), "index_0_100") if r["value"] != "" else r["status"], r["label"],
                 r["tag"]] for r in self.state.get("index", [])]

    def ablation_table(self) -> list[list]:
        return [[r["system"], r["n_segments"], _pct_ci(r, "primary_accuracy"), _pct_ci(r, "macro_f1_activity"),
                 f"{_num(r['kappa_primary']):.2f}" if r["kappa_primary"] != "" else "undefined", "metric"]
                for r in self.state.get("ablation", [])]
