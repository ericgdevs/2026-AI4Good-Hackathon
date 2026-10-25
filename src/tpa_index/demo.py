"""006 demo tabs: Timeline (moments plus per-segment 004 signals) and Insights (005 result folders).
Read-only. Segment classes come from 005's own chain (view -> rules -> indicators), never a restated rule. Rows are
built from an allow-list: no transcript text or counts, no filenames, no segment IDs."""
import re

from tpa_index import review
from tpa_index.app import LOCAL_BADGE, _cut, _fmt_conf, _fmt_time
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

        from tpa_index.schema import local_only

        # laptop-only videos are browsable here (badged, read-only); results never read them
        self.moments = sorted(read_official("moments", root=self.root, include_local=True).to_pylist(),
                              key=lambda m: (m["video_id"], m["start_s"]))
        self.local = set(local_only(self.root).get("predict", {}))
        self.state = review.latest_state(read_reviews(self.root), "review")
        self.segments, self.note = [], ""
        model = main_model(read_official("predictions", root=self.root))
        from tpa_index.config import threshold

        self.thresholds = {lab["name"]: threshold(self.pipeline, model, lab["name"]) for lab in self.labels["labels"]}
        if model is None:
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
        return {v["video_id"]: v for v in review.review_videos(self.root, include_local=True)}

    def _name(self, v: dict, shown: list[dict]) -> str:
        return video_name(v, shown) + (f" · {LOCAL_BADGE}" if v["video_id"] in self.local else "")

    def video_choices(self) -> tuple[list[tuple[str, str]], str]:
        """Same eval lock and message as the Review tab: eval videos are not listed while it holds."""
        _, reason = review.eval_unlocked(self.root)
        shown = list(self._allowed().values())
        return [(self._name(v, shown), v["video_id"]) for v in shown], reason

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

    # ---- Search moments and the Review queue's "AI noticed" box ----
    def strength(self, m) -> float | None:
        """The moment's confidence over its primary label's calibrated cut-off (never shown as a percentage).
        None for laptop runs: their model has no calibrated cut-offs."""
        if m["video_id"] in self.local:
            return None
        thr = self.thresholds.get(m["primary_label"])
        return m["confidence"] / thr if thr and m["confidence"] is not None else None

    def summary(self, moment_id) -> dict:
        """What the AI noticed across a moment's clips: counts and cues only."""
        from collections import Counter

        segs = [s for s in self.segments if s["moment_id"] == moment_id]
        people = Counter(int(s["person_count"]) for s in segs if s["person_count"] is not None).most_common(1)
        audio = [s["audio_ok"] or {} for s in segs]
        sounds = [lab for lab in dict.fromkeys(k for a in audio for k in a) if any(a.get(lab) for a in audio)]
        speech = [s["has_speech"] for s in segs if s["has_speech"] is not None]
        faces = [s for s in segs if s["faces_detected"]]
        expr = Counter(s["expr_top"] for s in faces if s["expr_top"]).most_common(1)
        staff = sum(s["cls"] in STAFF for s in segs)
        return {"people": people[0][0] if people else None,
                "sounds": None if all(not a or all(v is None for v in a.values()) for a in audio) else
                [display_label(lab, self.labels) for lab in sounds],
                "speech": None if not speech else any(speech),
                "expression": None if not segs or all(s["faces_detected"] is None for s in segs) else
                (display_label(expr[0][0], self.labels) if expr else "no faces"),
                "staff_led": bool(segs) and staff > len(segs) / 2}

    def search(self, query: str, video_id=None, show_staff=False) -> dict:
        """Local filter over moments in unlocked videos: label names (stored or readable), review status,
        "needs review", and "solo" / "group" (indicator labels, else the usual person count). No transcript."""
        from tpa_index.schema import read_reviews

        state = review.latest_state(read_reviews(self.root), "review")
        allowed = self._allowed()
        words = (query or "").lower().replace("status:", " ").split()
        text = " ".join(words)
        want_review = "needs review" in text
        if want_review:
            text = text.replace("needs review", " ")
        want_solo, want_group = "solo" in text.split(), "group" in text.split()
        rest = [w for w in text.split() if w not in ("solo", "group", "performance")] if (want_solo or want_group) \
            else text.split()
        results, hidden, staff = [], 0, 0
        for m in self.moments:
            if m["video_id"] not in allowed or (video_id and m["video_id"] != video_id):
                continue
            labs = [lab for lab, _ in review.ai_labels(m)]
            info = self.summary(m["moment_id"])
            status = queue_status(m, state)
            hay = " ".join([*labs, *(display_label(lab, self.labels).lower() for lab in labs), status.lower()])
            people = info["people"]
            ok = all(w in hay.replace("_", " ") or w in hay for w in rest)
            ok &= not want_review or bool(m["needs_review"])
            ok &= not want_solo or "solo_on_stage" in labs or people == 1
            ok &= not want_group or "group_performance" in labs or (people is not None and people >= 2)
            if not ok:
                continue
            staff += info["staff_led"]
            if info["staff_led"] and not show_staff:
                hidden += 1
                continue
            st = self.strength(m)
            detail = [f"{people} {'person' if people == 1 else 'people'}" if people is not None else "people n/a",
                      *(info["sounds"] or [])[:2], f"{st:.1f}×" if st is not None else ""]
            results.append({"video_id": m["video_id"], "moment_id": m["moment_id"], "start_s": m["start_s"],
                            "video": self._name(allowed[m["video_id"]], list(allowed.values())),
                            "time": _fmt_time(m["start_s"], m["end_s"]),
                            "label": display_label(m["primary_label"], self.labels),
                            "detail": " · ".join(d for d in detail if d), "review_status": status})
        marks, spans = self._marks(results, allowed, video_id)
        return {"results": results, "hidden": hidden, "staff": staff, "marks": marks, "spans": spans,
                "all_videos": not video_id}

    def _marks(self, results, allowed, video_id) -> tuple[list[float], list[tuple[str, float, float]]]:
        """Where each match falls (0-1) along the shown videos laid end to end in phase order (early -> mid ->
        showcase, then phase not set), and each phase's span on that line."""
        order = {"early": 0, "mid": 1, "showcase": 2}
        vids = sorted((v for v in allowed.values() if not video_id or v["video_id"] == video_id),
                      key=lambda v: (order.get(v["phase"], 3), v["video_id"]))
        length = {v["video_id"]: max([m["end_s"] for m in self.moments if m["video_id"] == v["video_id"]] + [0.0])
                  for v in vids}
        offset, total = {}, 0.0
        for v in vids:
            offset[v["video_id"]], total = total, total + length[v["video_id"]]
        if not total:
            return [], []
        spans: dict[str, list[float]] = {}
        for v in vids:
            ph = v["phase"] if v["phase"] in order else "unknown"
            lo, hi = offset[v["video_id"]] / total, (offset[v["video_id"]] + length[v["video_id"]]) / total
            spans.setdefault(ph, [lo, hi])[1] = hi
        return ([(offset[r["video_id"]] + r["start_s"]) / total for r in results],
                [(ph, lo, hi) for ph, (lo, hi) in spans.items() if hi > lo])


QUEUE_STATUS = {"unreviewed": "Not reviewed", "deferred": "Second look", "partly": "Partly reviewed"}


def queue_status(m: dict, state: dict) -> str:
    """Not reviewed | Second look | Partly reviewed | Edited | Accepted (from 003's moment_status)."""
    status = review.moment_status(m, state)
    if status != "done":
        return QUEUE_STATUS[status]
    edited = any(state.get((m["moment_id"], lab), {}).get("action") == "edit" for lab, _ in review.ai_labels(m))
    return "Edited" if edited else "Accepted"


def strength_text(st: float | None) -> str:
    if st is None:
        return ""
    return f"{st:.1f}× above its cut-off" if st >= 1 else f"{st:.1f}× its cut-off (below)"


# ---- Insights tab: reads 005 result folders (contracts/results.md) by file only ----
HEADLINE_COLUMNS = ("indicator", "phase", "value", "unit", "tag", "verified_share", "change_vs_early")
TABLE_HEADERS = ["indicator", "phase", "value", "tag", "change vs early"]
ABLATION_HEADERS = ["system", "segments", "primary accuracy (95% CI)", "macro-F1 activity (95% CI)", "kappa primary",
                    "tag"]
EMPTY = "Insights are not available yet. Run the insights step first."
PHASE_ORDER = ("early", "mid", "showcase")
PHASE_NAMES = {"early": "Early sessions", "mid": "Mid program", "showcase": "Showcase"}


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
        ev = latest_result("evaluate", self.root)
        inds = _read_csv(folder / "indicators.csv") if (folder / "indicators.csv").exists() else []
        self.state = {
            "videos": sorted({r["scope"] for r in inds if r["scope_type"] == "video"}),
            "coverage_reviewed": next((_num(r["value"]) for r in inds if r["indicator"] == "coverage_reviewed"
                                       and r["scope_type"] == "all"), None),
            # every clip in the result: the denominator of 005's coverage_reviewed over all videos
            "clips": next((round(_num(r["denominator"])) for r in inds if r["indicator"] == "coverage_reviewed"
                           and r["scope_type"] == "all" and r["denominator"] != ""), None),
            "status": "available", "created_at": manifest["created_at"], "limitations": manifest["limitations"],
            "folder": folder.name, "guideline_versions": manifest.get("guideline_versions", []),
            "headline": head, "model_tag": extra.get("model_tag", UNVALIDATED),
            "time": next((r for r in _read_csv(folder / "time_breakdown.csv") if r["scope_type"] == "all"), None),
            "ablation": _read_csv(ev / "ablation.csv") if ev and (ev / "ablation.csv").exists() else [],
        }
        return self.state

    # ---- Scorecard ----
    def phases(self) -> list[str]:
        found = {r["phase"] for r in self.state.get("headline", [])}
        return [p for p in PHASE_ORDER if p in found]

    def default_pair(self) -> tuple[str | None, str | None]:
        ph = self.phases()
        return (ph[0] if ph else None), (ph[-1] if len(ph) > 1 else None)

    def compare(self, a: str, b: str) -> dict:
        """The headline measures for phases a and b side by side (values and tags as written)."""
        rows = []
        for ind in dict.fromkeys(r["indicator"] for r in self.state.get("headline", [])):
            by = {r["phase"]: r for r in self.state["headline"] if r["indicator"] == ind}
            ra, rb = by.get(a), by.get(b)
            va = _num(ra["value"]) if ra and ra.get("status", "ok") in ("", "ok") else None
            vb = _num(rb["value"]) if rb and rb.get("status", "ok") in ("", "ok") else None
            ratio = vb / va if va and vb is not None else None
            unit = (rb or ra or {}).get("unit", "")
            rows.append({"indicator": ind, "a": va, "b": vb, "unit": unit, "ratio": ratio,
                         "a_status": ra["status"] if ra and va is None else "",
                         "b_status": rb["status"] if rb and vb is None else "",
                         "new": va == 0 and bool(vb), "tag": (rb if vb is not None else ra or {}).get("tag", "")})
        return {"a": a, "b": b, "rows": rows}

    def agreement_row(self) -> dict | None:
        """The ablation row for the AI version behind the insights numbers (learned when that was the fallback)."""
        rows = self.state.get("ablation", [])
        learned = self.state.get("model_tag") == "model estimate (learned)"
        pick = [r for r in rows if r["system"].startswith("learned") == learned and
                (learned or r["system"] == "zero-shot")]
        return (pick or rows or [None])[0]

    def facts(self) -> list[dict]:
        out = []
        r = self.agreement_row()
        v = _num(r["primary_accuracy"]) if r else None
        if v is None:
            out.append({"cap": "AI agreed with staff", "value": "n/a", "note": "no comparison with staff labels yet"})
        else:
            lo, hi = _num(r["primary_accuracy_ci_low"]), _num(r["primary_accuracy_ci_high"])
            m = round(_num(r["n_segments"]) or 0)
            rng = f" · range {100 * lo:.0f}–{100 * hi:.0f}%" if lo is not None and hi is not None else ""
            out.append({"cap": "AI agreed with staff", "value": f"{100 * v:.0f}%",
                        "note": f"{round(v * m)} of {m} blind-labelled clips matched the staff label{rng}"})
        out.append({"cap": "Checked on", "value": r["n_segments"] if r else "n/a",
                    "note": "blind-labelled clips, on videos the AI never tuned on" if r else "not available"})
        t = self.state.get("time")
        if t:
            youth, staff = _num(t["youth_s"]) / 60, (_num(t["staff_led_verified_s"]) + _num(t["staff_led_model_s"])) / 60
            out.append({"cap": "Youth time counted", "value": f"{youth:.0f} min",
                        "note": f"{staff:.0f} min staff-led left out"})
        cov = self.state.get("coverage_reviewed")
        out.append({"cap": "Staff-verified", "value": f"{100 * cov:.0f}%" if cov is not None else "n/a",
                    "note": "of clips have a staff-checked label"})
        return out

    def subtitle(self) -> str:
        """`<camp> · N videos · M clips · youth time only`; the camp only when every counted video shares one
        cohort."""
        from tpa_index.schema import read_official

        vids = self.state.get("videos", [])
        cohorts = {v["cohort"] for v in read_official("videos", root=self.root).to_pylist() if v["video_id"] in vids}
        camp = f"{cohorts.pop()} · " if len(cohorts) == 1 and None not in cohorts else ""
        n = self.state.get("clips")
        clips = f"{n} clip{'s' if n != 1 else ''} · " if n is not None else ""
        return f"{camp}{len(vids)} video{'s' if len(vids) != 1 else ''} · {clips}youth time only"

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

    def ablation_table(self) -> list[list]:
        return [[r["system"], r["n_segments"], _pct_ci(r, "primary_accuracy"), _pct_ci(r, "macro_f1_activity"),
                 f"{_num(r['kappa_primary']):.2f}" if r["kappa_primary"] != "" else "undefined", "metric"]
                for r in self.state.get("ablation", [])]
