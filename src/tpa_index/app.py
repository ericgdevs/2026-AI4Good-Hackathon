"""Local review app (Gradio): blind labeling, review, and the Timeline / Insights demo tabs. Binds 127.0.0.1 only; no sharing, analytics or monitoring.
Every label and group shown comes from labels.yaml / pipeline.yaml at runtime."""
import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # before gradio is imported anywhere

from tpa_index import review  # noqa: E402
from tpa_index.config import load_labels, load_pipeline, load_theme  # noqa: E402
from tpa_index.paths import redact  # noqa: E402

QUEUE_NAMES = {"Main sample": "main", "Overlap set (second labeler)": "overlap", "Top-up": "topup"}
REVIEWER_HINT = "staff first name or role"
LOCAL_BADGE = "local run (laptop models, uncalibrated)"
LOCAL_READ_ONLY = (f"This moment is from a {LOCAL_BADGE}. Reviews are saved only on official results, "
                   "so nothing was saved.")

# On load: force the light theme (Constitution X) and install the keyboard shortcuts: review queue buttons by element
# ID, blind checklist keys by the "[k]" suffix of each choice's text, so no label name is written here.
PAGE_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') { url.searchParams.set('__theme', 'light'); window.location.replace(url); return; }
  document.body.classList.remove('dark');
  const click = (sel) => { const el = document.querySelector(sel); if (el) { el.click(); return true; } return false; };
  document.addEventListener('keydown', (e) => {
    const t = e.target;
    if (t && (t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && !['radio', 'checkbox'].includes(t.type)))) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const review = document.querySelector('#review-decide');
    if (review && review.offsetParent !== null) {  // review queue: A / E / R / D and down arrow for the next item
      const k = { a: '#review-accept', e: '#review-edit', r: '#review-reject', d: '#review-defer' }[e.key.toLowerCase()];
      if (k) { if (click(k)) e.preventDefault(); return; }
      if (e.key === 'ArrowDown') { if (click('#review-next')) e.preventDefault(); }
      return;
    }
    const blind = document.querySelector('#blind-checklist');
    if (!blind || blind.offsetParent === null) return;  // the remaining keys belong to the blind tab only
    if (e.key === 'Enter') { if (click('#blind-save')) e.preventDefault(); return; }
    if (e.key === 'Backspace') { if (click('#blind-back')) e.preventDefault(); return; }
    if (e.key === ' ') { if (click('#blind-replay')) e.preventDefault(); return; }
    if (e.key.toLowerCase() === 'u') { if (click('#blind-uncertain input')) e.preventDefault(); return; }
    if (e.key.length !== 1) return;
    const suffix = '[' + e.key.toLowerCase() + ']';
    for (const lab of document.querySelectorAll('#blind-checklist label')) {
      // click the input itself: a label click toggles the box on screen but not Gradio's checkbox-group value
      if (lab.textContent.trim().endsWith(suffix)) { (lab.querySelector('input') || lab).click(); e.preventDefault(); return; }
    }
  });
}
"""


def _cut(cutter, video, start, end, pad=0) -> tuple[str | None, str]:
    """(clip path, message) for both tabs; errors shown to staff are redacted."""
    if video is None or cutter is None:
        return None, "Clip unavailable: video not in the official runs."
    try:
        return str(cutter.cut(video, start, end, pad_s=pad)), ""
    except Exception as e:
        return None, redact(f"Clip unavailable ({e}).")


class BlindController:
    """Blind-mode handlers (plain methods, so tests can drive them without a browser). Never reads predictions."""

    def __init__(self, cutter=None, root=None, labels=None, pipeline=None):
        self.cutter, self.root = cutter, root
        self.labels = labels or load_labels()
        self.pipeline = pipeline or load_pipeline(labels=self.labels)
        self.app = self.pipeline["app"]
        self.checklist = review.checklist(self.labels, self.pipeline)
        self.keys, self.unkeyed = review.key_map(self.checklist)
        self.key_of = {label: key for key, label in self.keys.items()}
        self.reload()

    def reload(self):
        self.videos = {v["video_id"]: v for v in review._official("videos", self.root).to_pylist()}

    def queue_choices(self) -> tuple[list[str], str]:
        """Queue picker entries; Top-up only once it can be built (official predict runs for the eval videos)."""
        ok, reason = review.topup_available(self.root)
        names = list(QUEUE_NAMES) if ok else list(QUEUE_NAMES)[:2]
        return names, "" if ok else f"Top-up: {reason}"

    # ---- display helpers ----
    def choice(self, label: str) -> tuple[str, str]:
        key = self.key_of.get(label)
        return (f"{label} [{key}]" if key else label, label)

    def _empty(self):
        return [None] + [[] for _ in self.checklist["groups"]] + [False]

    def _prefill(self, segment_id: str, reviewer: str):
        prev = {k: r for k, r in review.previous_blind(segment_id, reviewer, self.root).items() if r["action"] == "add"}
        primary = next((label for label, r in prev.items() if r["label_role"] == review.PRIMARY), None)
        groups = [[n for n in names if n in prev] for _, names in self.checklist["groups"]]
        return [primary] + groups + [self.checklist["uncertain"] in prev]

    def _progress(self, state) -> str:
        done, total = review.progress(state["queue"], state["reviewer"], self.root)
        return f"**{done} / {total}** done"

    def _clip(self, item):
        clip, msg = _cut(self.cutter, self.videos.get(item["video_id"]), item["start_s"], item["end_s"])
        return clip, msg and msg + " Use Skip."

    def _prefetch(self, queue, position):
        items = review.load_queue(queue, self.root)
        nxt = next((q for q in items if q["position"] > position), None)
        if nxt and self.cutter is not None and nxt["video_id"] in self.videos:
            self.cutter.prefetch(self.videos[nxt["video_id"]], nxt["start_s"], nxt["end_s"])

    def _show(self, state, position, prefill=False):
        """Outputs: state, clip, progress, primary, *groups, uncertain, status."""
        items = {q["position"]: q for q in review.load_queue(state["queue"], self.root)}
        if not items:
            state = {**state, "position": None}
            return [state, None, "", *self._empty(),
                    "This queue has no segments (for the top-up: every eval segment is already in the main sample)."]
        if position is None or position not in items:
            state = {**state, "position": None}
            return [state, None, self._progress(state), *self._empty(), "This queue is finished. Thank you."]
        item = items[position]
        state = {**state, "position": position, "segment_id": item["segment_id"]}
        clip, status = self._clip(item)
        self._prefetch(state["queue"], position)
        values = self._prefill(item["segment_id"], state["reviewer"]) if prefill else self._empty()
        return [state, clip, self._progress(state), *values, status]

    # ---- handlers ----
    def start(self, reviewer: str, queue_display: str):
        reviewer = (reviewer or "").strip().lower()  # "Ana" and "ana" are one reviewer (resume, eval lock)
        if not reviewer:
            return [None, None, "", *self._empty(), f"Enter a reviewer ({REVIEWER_HINT}) first."]
        queue = QUEUE_NAMES[queue_display]
        try:
            review.build_main_queues(self.app, self.root)
            if queue == "topup":
                review.build_topup_queue(self.app, self.labels, self.pipeline, self.root)
        except Exception as e:
            return [None, None, "", *self._empty(), redact(str(e))]
        state = {"reviewer": reviewer, "queue": queue, "position": None}
        position = review.next_position(queue, reviewer, self.root)
        return self._show(state, position, prefill=False)

    def save(self, state, primary, *rest):
        *groups, uncertain = rest
        if not state or state.get("position") is None:
            return [state, None, "", *self._empty(), "Start a queue first."]
        chosen = [n for values in groups for n in (values or [])]
        prev = review.previous_blind(state["segment_id"], state["reviewer"], self.root)
        try:
            rows = review.blind_rows({"segment_id": state["segment_id"]}, primary, chosen, bool(uncertain),
                                     state["reviewer"], self.labels, self.pipeline, previous=prev)
        except ValueError as e:  # nothing saved; keep the clip and everything ticked, show why
            import gradio as gr

            return [state, gr.update(), gr.update(), primary, *groups, uncertain, str(e)]
        if rows:
            from tpa_index.schema import append_reviews

            append_reviews(rows, self.root)
        return self._show(state, review.next_position(state["queue"], state["reviewer"], self.root))

    def back(self, state):
        if not state or state.get("position") in (None, 0):
            return self._show(state, 0, prefill=True) if state else [None, None, "", *self._empty(), ""]
        return self._show(state, state["position"] - 1, prefill=True)

    def replay(self, state):
        if not state or state.get("position") is None:
            return [state, None, "", *self._empty(), ""]
        return self._show(state, state["position"], prefill=True)

    def skip(self, state):
        if not state or state.get("position") is None:
            return [state, None, "", *self._empty(), "Start a queue first."]
        from tpa_index.schema import append_reviews

        append_reviews([review.skip_row({"segment_id": state["segment_id"]}, state["reviewer"], self.labels,
                                        self.pipeline)], self.root)
        return self._show(state, review.next_position(state["queue"], state["reviewer"], self.root))


ANY = "any"
TIMELINE_HEADERS = ["time", "primary", "review status", "needs review", "confidence"]
PANEL_HEADERS = ["label", "role", "confidence", "state"]
HISTORY_HEADERS = ["time", "reviewer", "label", "action", "note"]
REMINDER = "**No names. No judgments about a person.**"


def _fmt_time(start, end) -> str:
    mmss = lambda t: f"{int(t // 60)}:{int(t % 60):02d}"  # noqa: E731
    return f"{mmss(start)}–{mmss(end)}"


def _fmt_conf(c) -> str:
    return "" if c is None else f"{c:.3g}"  # real scores can be ~0.001: keep significant digits


class ReviewController:
    """Review-mode handlers (plain methods, testable without a browser). Reads official runs once per reload."""

    def __init__(self, cutter=None, root=None, labels=None, pipeline=None):
        self.cutter, self.root = cutter, root
        self.labels = labels or load_labels()
        self.pipeline = pipeline or load_pipeline(labels=self.labels)
        self.vocab = [lab["name"] for lab in self.labels["labels"] if review._selectable(lab, self.pipeline)]
        self.group_of = {lab["name"]: lab["group"] for lab in self.labels["labels"]}
        from tpa_index.config import observation_types

        self.observations = observation_types(self.labels)
        self.reload()

    def reload(self):
        from collections import Counter

        import pyarrow.compute as pc

        from tpa_index.config import threshold
        from tpa_index.schema import read_official

        from tpa_index.schema import local_only

        # laptop-only videos are shown (badged, read-only); scores and thresholds stay the official model's
        self.moments = read_official("moments", root=self.root, include_local=True).to_pylist()
        self.segments = read_official("segments", root=self.root, include_local=True).to_pylist()
        self.local = set(local_only(self.root).get("predict", {}))
        preds = read_official("predictions", root=self.root)
        main = Counter(m for m in preds["model"].to_pylist() if m != "rules").most_common(1)
        # shortcut: one main model across the official runs (as calibrate assumes); per-video models if they mix
        self.model = main[0][0] if main else None
        self.scores: dict[str, dict[str, float]] = {}
        if self.model:
            for r in preds.filter(pc.equal(preds["model"], self.model)).select(["segment_id", "label", "score"]).to_pylist():
                self.scores.setdefault(r["segment_id"], {})[r["label"]] = r["score"]
        self.thresholds = {lab: threshold(self.pipeline, self.model, lab) for lab in self.group_of}

    # ---- choices ----
    def video_choices(self) -> tuple[list[tuple[str, str]], str]:
        _, reason = review.eval_unlocked(self.root)
        videos = review.review_videos(self.root, include_local=True)
        return [(f"{v['video_id']} ({v['split']})", v["video_id"]) for v in videos], reason

    def label_filter_choices(self) -> list[tuple[str, str]]:
        from tpa_index.demo import display_label

        return [(ANY, ANY)] + [(display_label(n, self.labels), n) for n in self.vocab + self.observations]

    def named(self, names) -> list[tuple[str, str]]:
        """(readable name, stored name) pairs, sorted by the readable name (the Add and Edit pickers)."""
        from tpa_index.demo import display_label

        return sorted(((display_label(n, self.labels), n) for n in names), key=lambda c: c[0].lower())

    def edit_choices(self, current, label) -> list[str]:
        group = self.group_of.get(label)
        return [n for n in self.vocab if self.group_of[n] == group and n != label] if group else []

    # ---- data ----
    def _state(self) -> dict:
        from tpa_index.schema import read_reviews

        return review.latest_state(read_reviews(self.root), "review")

    def _confidences(self, m) -> dict[str, float]:
        # shortcut: mean over the non-idle segments inside the moment's span; merge_moments' gap segments may differ
        segs = [s["segment_id"] for s in self.segments if s["video_id"] == m["video_id"] and not s.get("is_idle")
                and s["start_s"] >= m["start_s"] and s["end_s"] <= m["end_s"]]
        out = {}
        for lab, _ in review.ai_labels(m):
            vals = [self.scores[s][lab] for s in segs if lab in self.scores.get(s, {})]
            if vals:
                out[lab] = sum(vals) / len(vals)
        return out

    def timeline(self, video_id, label=ANY, needs=ANY, status=ANY, show_idle=False):
        """Outputs: table values, rows (state), message."""
        allowed = {v["video_id"] for v in review.review_videos(self.root, include_local=True)}
        if not video_id or video_id not in allowed:
            return [], [], "Pick a video." if not video_id else review.eval_unlocked(self.root)[1]
        state = self._state()
        found = review.filter_moments([m for m in self.moments if m["video_id"] == video_id], state,
                                      label=None if label in (None, ANY) else label,
                                      needs_review=None if needs in (None, ANY) else needs == "yes",
                                      status=None if status in (None, ANY) else status)
        rows = [{"kind": "moment", **m, "review_status": review.moment_status(m, state)} for m in found]
        if show_idle:
            rows += [{"kind": "idle", "video_id": video_id, "start_s": s["start_s"], "end_s": s["end_s"]}
                     for s in self.segments if s["video_id"] == video_id and s.get("is_idle")]
        rows.sort(key=lambda r: r["start_s"])
        table = [[_fmt_time(r["start_s"], r["end_s"]), r["primary_label"], r["review_status"],
                  "yes" if r["needs_review"] else "no", _fmt_conf(r["confidence"])] if r["kind"] == "moment"
                 else [_fmt_time(r["start_s"], r["end_s"]), "idle", "", "", ""] for r in rows]
        return table, rows, "" if rows else "No moments match these filters."

    def _panel(self, m):
        state = self._state()
        rows = review.label_panel(m, state, self.labels, self._confidences(m), self.thresholds)
        panel = [[r["label"] + (" (retired)" if r["retired"] else ""), r["role"], _fmt_conf(r["confidence"]),
                  r["badge"]] for r in rows]
        history = [[review.local_time(r["created_at"]), r["reviewer"],
                    f"{r['label']} → {r['new_label']}" if r["action"] == "edit" else r["label"], r["action"],
                    r["note"] or ""]
                   for r in sorted(self._history_rows(m), key=lambda r: (r["created_at"], r["review_id"]))]
        status = f"Status: **{review.moment_status(m, state)}**"
        return panel, [lab for lab, _ in review.ai_labels(m)], history, status

    def _history_rows(self, m):
        from tpa_index.schema import read_reviews

        return [r for r in read_reviews(self.root).to_pylist()
                if r["mode"] == "review" and r["target_id"] == m["moment_id"]]

    # ---- handlers ----
    def open(self, rows, index):
        """Outputs: current, clip, panel, AI-label choices, history, moment status, message."""
        if not rows or index is None or not 0 <= index < len(rows):
            return [None, None, [], [], [], "", ""]
        r = rows[index]
        video = next((v for v in review.review_videos(self.root, include_local=True) if v["video_id"] == r["video_id"]), None)
        clip, msg = _cut(self.cutter, video, r["start_s"], r["end_s"], self.pipeline["app"]["padding_s"])
        if r["kind"] != "moment":
            return [None, clip, [], [], [], "Idle stretch (no labels).", msg]
        m = {k: v for k, v in r.items() if k not in ("kind", "review_status")}
        return [m, clip, *self._panel(m), msg]

    def _save(self, current, reviewer, make):
        """Outputs: panel, history, moment status, message."""
        reviewer = (reviewer or "").strip().lower()
        if current is None:
            return [None, None, None, "Select a moment first."]
        if current["video_id"] in self.local:
            return [None, None, None, LOCAL_READ_ONLY]
        if not reviewer:
            return [None, None, None, f"Enter a reviewer ({REVIEWER_HINT}) first."]
        try:
            row = make(reviewer)
        except ValueError as e:
            return [None, None, None, str(e)]
        from tpa_index.schema import append_reviews

        append_reviews([row], self.root)
        panel, _, history, status = self._panel(current)
        return [panel, history, status, f"Saved: {row['action']} {row['label']}."]

    def act(self, current, reviewer, label, action, edit_to=None, note=None):
        role = dict(review.ai_labels(current)).get(label) if current else None
        if current is not None and role is None:
            return [None, None, None, "Pick one of the moment's AI labels."]
        return self._save(current, reviewer, lambda who: review.review_row(
            current, label, role, action, who, self.labels, new_label=edit_to if action == "edit" else None,
            note=note))

    def add_label(self, current, reviewer, label, note=None):
        if label not in self.vocab:
            return [None, None, None, "Pick a label to add."]
        if current is not None and label in dict(review.ai_labels(current)):
            # same (moment, label) key as the AI suggestion: an add would hide its accept/edit/reject state
            return [None, None, None, f"{label} is already an AI suggestion here: use Accept or Edit."]
        role = review.role_of(label, self.labels, self.pipeline)
        return self._save(current, reviewer, lambda who: review.review_row(
            current, label, role, "add", who, self.labels, note=note))

    def add_observation(self, current, reviewer, obs, note=None):
        return self._save(current, reviewer, lambda who: review.observation_row(current, obs, who, self.labels, note))

    # ---- Review queue (three panes): queue entries, activity checkboxes, one-click decisions ----
    def queue_choices(self, rows) -> list[tuple[str, str]]:
        from tpa_index.demo import display_label, queue_status

        state = self._state()
        return [(f"{_fmt_time(r['start_s'], r['end_s'])} · {display_label(r['primary_label'], self.labels)}\n"
                 f"{queue_status(r, state)}" if r["kind"] == "moment" else
                 f"{_fmt_time(r['start_s'], r['end_s'])} · Idle\nno labels", str(i)) for i, r in enumerate(rows)]

    def queue_empty(self, video_id, rows, msg="") -> str:
        """The line shown instead of an empty queue: no moments at all in the video, or none match the filters."""
        if not video_id or rows:
            return ""
        return EMPTY_QUEUE if not self.timeline(video_id)[1] else msg

    def queue_head(self, video_name: str, rows) -> str:
        from tpa_index.demo import queue_status

        state = self._state()
        moments = [r for r in rows if r["kind"] == "moment"]
        done = sum(queue_status(r, state) in ("Accepted", "Edited") for r in moments)
        if not video_name:
            return "Pick a video."
        from html import escape

        # the video name may shorten with "…"; the done count always stays visible on the same line
        return (f'<div class="tpa-qhead"><strong class="tpa-qname">To review · {escape(video_name)}</strong>'
                f'<span class="tpa-qcount">· {done} / {len(moments)} done</span></div>')

    def suggestion(self, current) -> str:
        from tpa_index.demo import display_label, strength_text

        if current is None:
            return PICK_MOMENT
        if current["video_id"] in self.local:
            return (f"AI suggests **{display_label(current['primary_label'], self.labels)}** · {LOCAL_BADGE} · "
                    "view only")
        thr = self.thresholds.get(current["primary_label"])
        st = current["confidence"] / thr if thr and current["confidence"] is not None else None
        extra = f" · {strength_text(st)}" if st is not None else ""
        return f"AI suggests **{display_label(current['primary_label'], self.labels)}**{extra}"

    def activities(self, current) -> tuple[list[tuple[str, str]], list[str]]:
        """Checkbox choices (the moment's AI labels, main first, then other activities) and the ticked ones: AI labels
        not rejected, plus labels staff added."""
        from tpa_index.demo import display_label

        if current is None:
            return [], []
        ai = [lab for lab, _ in review.ai_labels(current)]
        others = [n for n in review.checklist(self.labels, self.pipeline)["primary"] if n not in ai]  # activities
        state = self._state()
        mid = current["moment_id"]
        ticked = [lab for lab in ai if state.get((mid, lab), {}).get("action") != "reject"]
        ticked += [lab for lab in others if state.get((mid, lab), {}).get("action") == "add"]
        main = current["primary_label"]
        return ([(display_label(n, self.labels) + (" (main)" if n == main else ""), n) for n in ai + others], ticked)

    def decide(self, current, reviewer, action, ticked=None, edit_to=None, note=None):
        """One decision on the main label, plus the checkbox changes: an unticked AI label is rejected, a ticked label
        the AI did not suggest is added by staff, and on accept or edit the other ticked AI labels are accepted (staff
        confirmed the set). Same rows as the per-label actions; the note goes on the main row."""
        reviewer = (reviewer or "").strip().lower()
        if current is None:
            return [None, None, None, "Select a moment first."]
        if current["video_id"] in self.local:
            return [None, None, None, LOCAL_READ_ONLY]
        if not reviewer:
            return [None, None, None, f"Enter a reviewer ({REVIEWER_HINT}) first."]
        state, mid, ticked = self._state(), current["moment_id"], set(ticked or [])
        ai = dict(review.ai_labels(current))
        main = current["primary_label"]
        rows = []
        try:
            for lab, role in ai.items():
                if lab == main:
                    continue
                latest = state.get((mid, lab), {}).get("action")
                if lab not in ticked and latest != "reject":
                    rows.append(review.review_row(current, lab, role, "reject", reviewer, self.labels))
                elif lab in ticked and action in ("accept", "edit") and latest != "accept":
                    rows.append(review.review_row(current, lab, role, "accept", reviewer, self.labels))
            for lab in sorted(ticked - set(ai)):
                if lab in self.vocab and state.get((mid, lab), {}).get("action") != "add":
                    rows.append(review.review_row(current, lab, review.role_of(lab, self.labels, self.pipeline), "add",
                                                  reviewer, self.labels))
            rows.append(review.review_row(current, main, ai[main], action, reviewer, self.labels,
                                          new_label=edit_to if action == "edit" else None, note=note))
        except ValueError as e:
            return [None, None, None, str(e)]
        from tpa_index.schema import append_reviews

        append_reviews(rows, self.root)
        panel, _, history, status = self._panel(current)
        return [panel, history, status, f"Saved: {action} {main}" + (f" (+{len(rows) - 1} label change(s))"
                                                                    if len(rows) > 1 else "") + "."]


def _download_dir() -> str:
    import tempfile

    # inside Gradio's temp folder (the private session folder once launched), so the file can be served
    return tempfile.mkdtemp(prefix="download-", dir=os.environ.get("GRADIO_TEMP_DIR") or None)


def report_download(root=None) -> tuple[str | None, str]:
    """(path of a fresh Word report, message). The plain-English reason when there is no insights result."""
    from pathlib import Path

    from tpa_index.config import ConfigError
    from tpa_index.word_report import write_word_report

    try:
        return str(write_word_report(Path(_download_dir()) / "TPA_insights_report.docx", root)), ""
    except ConfigError as e:
        return None, str(e)


def export_download(verified_only=False, root=None) -> tuple[list[str], str]:
    """(paths of the JSON/CSV pair copied for download, message). The originals stay under OUTPUT_DIR/exports."""
    import shutil
    from pathlib import Path

    from tpa_index.config import ConfigError
    from tpa_index.export import export_official

    try:
        counts, paths = export_official(root, verified_only=bool(verified_only))
    except ConfigError as e:
        return [], redact(str(e))
    out = Path(_download_dir())
    copies = [str(shutil.copy(p, out / p.name)) for p in paths]
    summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
    return copies, f"Exported {sum(counts.values())} moments ({summary})."


PROCESS_INTRO = ("Runs the laptop versions of the models on one video, on this machine. Nothing leaves it: no "
                 "cloud, no saved frames or faces, and no transcript text (speech yes/no only).")
PROCESS_HEADERS = ["time", "activity (AI suggestion, uncalibrated)", "needs review"]


def local_note(root=None) -> str:
    """Scorecard line: how many videos have laptop runs (never part of these numbers)."""
    from tpa_index.schema import load_official, local_runs_path, read_table

    ids = load_official(local_runs_path(root)).get("predict", [])
    n = len({v for v in read_table("videos", root, run_ids=ids)["video_id"].to_pylist()}) if ids else 0
    return (f"{n} video{'s' if n != 1 else ''} processed on this laptop {'are' if n != 1 else 'is'} not in these "
            "numbers.") if n else ""


def process_rows(moments: list[dict], labels: dict) -> list[list]:
    from tpa_index.demo import display_label

    return [[_fmt_time(m["start_s"], m["end_s"]), display_label(m["primary_label"], labels),
             "yes" if m["needs_review"] else "no"] for m in sorted(moments, key=lambda m: m["start_s"])]


def process_video(video_id, root=None, labels=None, poll=0.5, **makers):
    """Generator for the Process button: (progress HTML, preview HTML, message, moments table update) about twice a
    second until the run ends. `makers` replace the model loaders in tests."""
    import gradio as gr

    from tpa_index import process, ui_html

    labels = labels or load_labels()
    if not video_id:
        yield ui_html.process_status(None), ui_html.preview_panel(None, labels), "Pick a video first.", gr.update()
        return
    try:
        job = process.start(video_id, root, **makers)
    except Exception as e:  # noqa: BLE001 - config problems become a message, never a traceback
        job = process.plain("predict", e)
    if isinstance(job, str):
        yield ui_html.process_status(None), ui_html.preview_panel(None, labels), job, gr.update()
        return
    while not job.done.wait(poll):
        yield (ui_html.process_status(job.snapshot()), ui_html.preview_panel(job.live.view(), labels), "",
               gr.update())
    rows = process_rows(job.moments, labels)
    yield (ui_html.process_status(job.snapshot()), ui_html.preview_panel(job.live.view(), labels), job.message,
           gr.update(value=rows, visible=bool(rows)))


PICK_MOMENT = "Pick a moment from the queue."
EMPTY_QUEUE = "No moments found in this video. The AI didn't find a clear activity here."
QUICK_SEARCHES = {"Solo singing": "solo singing", "Spoken word": "spoken word", "Group performance": "group performance",
                  "Needs review": "needs review"}
SEARCH_HEADERS = ["video · time", "label", "details", "review status"]
OBSERVATION_PREFIX = "expression_"  # staff expression observation types in labels.yaml get a quick button each
SHORTCUT_HINT = ('<p class="tpa-hint"><kbd>A</kbd> accept · <kbd>E</kbd> edit · <kbd>R</kbd> reject · <kbd>D</kbd> defer'
                 ' · <kbd>↓</kbd> next</p>')


def build_app(root=None, cutter=None, labels=None, pipeline=None):
    import gradio as gr

    from tpa_index import demo as tabs
    from tpa_index import ui_html

    blind = BlindController(cutter, root, labels, pipeline)
    rv = ReviewController(cutter, root, blind.labels, blind.pipeline)
    tl = tabs.DemoController(cutter, root, blind.labels, blind.pipeline)
    ins = tabs.InsightsController(root)
    cl = blind.checklist
    palette = load_theme()

    def phase_choices():
        return [(ui_html.phase_name(ph), ph) for ph in ins.phases()]

    def scorecard(a, b):
        if not ins.available or not a or not b:
            return "", "", ""
        cmp = ins.compare(a, b)
        return ui_html.hero(cmp), ui_html.drives(cmp), ui_html.facts(ins.facts())

    def search(query, video_id, show_staff):
        found = tl.search(query, video_id or None, show_staff)
        rows = [[f"{r['video']} · {r['time']}", r["label"], r["detail"], r["review_status"]] for r in found["results"]]
        hits = [(r["video_id"], r["moment_id"]) for r in found["results"]]
        return ui_html.strip(found), rows, hits

    with gr.Blocks(analytics_enabled=False, title="This Is ME · TPA") as demo:
        demo.blind, demo.review, demo.timeline, demo.insights = blind, rv, tl, ins
        # the tab bar itself is the single top bar: app name and note are drawn around the tabs by page_css()
        with gr.Tabs(elem_id="app-tabs") as app_tabs:
            # ---- Scorecard (opens first) ----
            with gr.Tab("Scorecard", id="scorecard", elem_id="scorecard-tab"):
                a0, b0 = ins.default_pair()
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        gr.Markdown("# Program scorecard", elem_id="scorecard-title")
                        sc_sub = gr.Markdown(ins.subtitle() if ins.available else "", elem_id="scorecard-subtitle")
                    with gr.Column(scale=4):
                        with gr.Row():
                            sc_a = gr.Dropdown(phase_choices(), value=a0, label="Compare", elem_id="scorecard-from")
                            sc_b = gr.Dropdown(phase_choices(), value=b0, label="with", elem_id="scorecard-to")
                            sc_dl = gr.Button("Download report", variant="primary", elem_id="scorecard-download",
                                              visible=ins.available)
                sc_empty = gr.Markdown(ins.message(), elem_id="scorecard-empty")
                sc_file = gr.File(label="Word report", visible=False, interactive=False, elem_id="scorecard-file")
                g0, d0, f0 = scorecard(a0, b0)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=340):
                        sc_hero = gr.HTML(g0, elem_id="scorecard-hero")
                    with gr.Column(scale=3, min_width=480):
                        sc_drives = gr.HTML(d0, elem_id="scorecard-drives")
                sc_facts = gr.HTML(f0, elem_id="scorecard-facts")
                sc_local = gr.Markdown(local_note(root), elem_id="scorecard-local")
            # ---- Search moments (same data and eval lock as Review) ----
            with gr.Tab("Search moments", id="search", elem_id="search-tab"):
                s_choices, s_reason = tl.video_choices()
                with gr.Row():
                    s_box = gr.Textbox(placeholder="Search moments: a label, a status, solo or group",
                                       show_label=False, elem_id="search-box", scale=4)
                    s_video = gr.Dropdown([("All videos", "")] + s_choices, value="", label="Video",
                                          elem_id="search-video", scale=1)
                s_lock = gr.Markdown(s_reason, elem_id="search-lock")
                with gr.Row(elem_classes="tpa-quick"):
                    gr.HTML('<p class="tpa-cap">Try</p>', elem_id="search-try")
                    quick = [gr.Button(name, size="sm", scale=0, min_width=0, elem_id=f"search-q-{i}")
                             for i, name in enumerate(QUICK_SEARCHES)]
                s_staff = gr.Checkbox(False, label="Show staff-led moments", elem_id="search-show-staff")
                strip0, rows0, hits0 = search("", "", False)
                s_strip = gr.HTML(strip0, elem_id="search-strip")
                s_results = gr.Dataframe(rows0, headers=SEARCH_HEADERS, interactive=False,
                                         label="Results (select one to open it in the Review queue)",
                                         elem_id="search-results")
                s_hits = gr.State(hits0)
                gr.Markdown(tabs.NOTE, elem_id="search-note")
            # ---- Review queue: queue | clip and what the AI noticed | your review ----
            with gr.Tab("Review queue", id="review", elem_id="review-tab"):
                with gr.Row():
                    video = gr.Dropdown(s_choices, value=None, label="Video", elem_id="review-video", scale=2)
                    reviewer = gr.Textbox(label="Reviewer", placeholder=REVIEWER_HINT, elem_id="reviewer", scale=2)
                lock_note = gr.Markdown(s_reason, elem_id="review-lock")
                rows, current = gr.State([]), gr.State(None)
                with gr.Row(equal_height=False, elem_id="review-panes"):
                    with gr.Column(scale=1, min_width=260):
                        q_head = gr.HTML("Pick a video.", elem_id="review-queue-head")
                        q_empty = gr.Markdown("", elem_id="review-queue-empty")
                        with gr.Column(elem_id="review-queue-box"):
                            queue = gr.Radio([], show_label=False, elem_id="review-queue")
                        with gr.Accordion("Filter the queue", open=False, elem_id="review-filters"):
                            f_label = gr.Dropdown(rv.label_filter_choices(), value=ANY, label="Label",
                                                  elem_id="review-f-label")
                            f_needs = gr.Radio([ANY, "yes", "no"], value=ANY, label="Needs review",
                                               elem_id="review-f-needs")
                            f_status = gr.Dropdown([ANY, *review.STATUSES], value=ANY, label="Status",
                                                   elem_id="review-f-status")
                            f_idle = gr.Checkbox(False, label="Show idle", elem_id="review-f-idle")
                    with gr.Column(scale=4, min_width=420):
                        r_clip = gr.Video(autoplay=True, interactive=False, label="Moment (with context)", height=340,
                                          elem_id="review-clip")
                        r_noticed = gr.HTML(ui_html.noticed(None), elem_id="review-noticed")
                        r_msg = gr.Markdown(elem_id="review-msg")
                    with gr.Column(scale=2, min_width=320, elem_id="review-decide"):
                        r_suggest = gr.Markdown(PICK_MOMENT, elem_id="review-suggest")
                        r_acts = gr.CheckboxGroup([], label="Activities in this clip", elem_id="review-activities")
                        gr.Markdown("**Your observation** (optional)", elem_id="review-obs-title")
                        with gr.Row(elem_classes="tpa-pills"):
                            obs_btns = {o: gr.Button(o.removeprefix(OBSERVATION_PREFIX).capitalize(), size="sm",
                                                     scale=0, min_width=0,
                                                     elem_id=f"review-obs-{o.removeprefix(OBSERVATION_PREFIX)}")
                                        for o in rv.observations if o.startswith(OBSERVATION_PREFIX)}
                        note_box = gr.Textbox(label="Note (optional)", placeholder="Add a note or your own label",
                                              max_lines=2, elem_id="review-note")
                        gr.Markdown(REMINDER, elem_id="review-reminder")
                        with gr.Row(elem_classes="tpa-decide"):
                            accept_btn = gr.Button("Accept", variant="primary", min_width=0, interactive=False,
                                                   elem_id="review-accept")
                            edit_btn = gr.Button("Edit", min_width=0, interactive=False,
                                                   elem_id="review-edit")
                            reject_btn = gr.Button("Reject", min_width=0, interactive=False,
                                                   elem_id="review-reject")
                            defer_btn = gr.Button("Defer", min_width=0, interactive=False,
                                                   elem_id="review-defer")
                        edit_to = gr.Dropdown([], label="Edit main label to", elem_id="review-edit-to")
                        with gr.Row(equal_height=True):
                            gr.HTML(SHORTCUT_HINT, elem_id="review-hint")
                            next_btn = gr.Button("Next ↓", size="sm", scale=0, min_width=90, elem_id="review-next")
                        m_status = gr.Markdown(elem_id="review-status")
                        with gr.Accordion("More: other labels, observations and history", open=False,
                                          elem_id="review-more"):
                            with gr.Row():
                                add_pick = gr.Dropdown(rv.named(rv.vocab), label="Add label",
                                                       elem_id="review-add-label")
                                add_btn = gr.Button("Add label", elem_id="review-add")
                            with gr.Row():
                                obs_pick = gr.Dropdown(rv.named(rv.observations), label="Add observation",
                                                       elem_id="review-add-obs")
                                obs_btn = gr.Button("Add observation", elem_id="review-obs")
                            panel = gr.Dataframe(headers=PANEL_HEADERS, interactive=False, label="AI labels",
                                                 elem_id="review-panel")
                            history = gr.Dataframe(headers=HISTORY_HEADERS, interactive=False, label="Review history",
                                                   elem_id="review-history")
            # ---- Process a video (laptop models; runs register as local, never official) ----
            with gr.Tab("Process a video", id="process", elem_id="process-tab"):
                from tpa_index import process as proc

                gr.Markdown("# Process a video", elem_id="process-title")
                gr.Markdown(PROCESS_INTRO, elem_id="process-intro")
                p_choices, p_note = proc.video_choices(root)
                with gr.Row():
                    p_video = gr.Dropdown(p_choices, value=None, label="Video", elem_id="process-video", scale=4)
                    p_btn = gr.Button("Process", variant="primary", elem_id="process-run", scale=1)
                p_note_md = gr.Markdown(p_note, elem_id="process-note")
                p_status = gr.HTML(ui_html.process_status(None), elem_id="process-status")
                p_preview = gr.HTML(ui_html.preview_panel(None, blind.labels), elem_id="process-preview")
                p_msg = gr.Markdown(elem_id="process-msg")
                p_moments = gr.Dataframe(headers=PROCESS_HEADERS, interactive=False, visible=False,
                                         label="Moments in this laptop run", elem_id="process-moments")
            # ---- Blind labels (unchanged behaviour) ----
            with gr.Tab("Blind labels", id="blind", elem_id="blind-tab"):
                blind_reviewer = gr.Textbox(label="Reviewer", placeholder=REVIEWER_HINT, elem_id="blind-reviewer")
                names, note = blind.queue_choices()
                queue_pick = gr.Radio(names, value="Main sample", label="Queue", elem_id="blind-queue")
                topup_note = gr.Markdown(note)
                start_btn = gr.Button("Start / resume", variant="primary")
                progress = gr.Markdown()
                status = gr.Markdown()
                clip = gr.Video(autoplay=True, loop=True, interactive=False, label="Segment", elem_id="blind-clip")
                with gr.Column(elem_id="blind-checklist"):
                    primary = gr.Radio([blind.choice(n) for n in cl["primary"]], label="Primary activity",
                                       elem_id="blind-primary")
                    group_boxes = [gr.CheckboxGroup([blind.choice(n) for n in names], label=group,
                                                    elem_id=f"blind-group-{i}")
                                   for i, (group, names) in enumerate(cl["groups"])]
                uncertain = gr.Checkbox(label=f"{cl['uncertain']} [u]", elem_id="blind-uncertain")
                if blind.unkeyed:
                    gr.Markdown("No shortcut: " + ", ".join(blind.unkeyed))
                with gr.Row():
                    back_btn = gr.Button("Back [Backspace]", elem_id="blind-back")
                    replay_btn = gr.Button("Replay [Space]", elem_id="blind-replay")
                    skip_btn = gr.Button("Skip: clip unavailable", elem_id="blind-skip")
                    save_btn = gr.Button("Save & next [Enter]", variant="primary", elem_id="blind-save")
            # ---- Export ----
            with gr.Tab("Export", id="export", elem_id="export-tab"):
                gr.Markdown("### Annotation export\nOne record per moment, as JSON and CSV, in the case format. The "
                            "files are also kept in the outputs folder.", elem_id="export-intro")
                ex_verified = gr.Checkbox(False, label="Leave out moments whose main label staff rejected",
                                          elem_id="export-verified")
                ex_btn = gr.Button("Create annotation export", variant="primary", elem_id="export-run")
                ex_msg = gr.Markdown(elem_id="export-msg")
                ex_files = gr.File(label="Export files", file_count="multiple", interactive=False,
                                   elem_id="export-files")
            # ---- Help: static questions from config/help.yaml (no model, no network) ----
            with gr.Tab("Help", id="help", elem_id="help-tab"):
                from tpa_index.config import load_help

                help_items = load_help()
                gr.Markdown("# Help", elem_id="help-title")
                help_box = gr.Textbox(placeholder="Search the questions, for example: review, cut-off, export",
                                      show_label=False, elem_id="help-search")
                help_list = gr.HTML(ui_html.help_html(help_items), elem_id="help-list")
                help_box.change(lambda q: ui_html.help_html(help_items, q), help_box, help_list)
        with gr.Row(elem_id="app-footer", equal_height=True):
            gr.Markdown(palette["credit"], elem_id="app-credit-footer")
            # the credit lives in the footer; this hidden copy keeps the header credit ID for existing checks
            gr.Markdown(palette["credit"], elem_id="app-credit-header", visible=False)
            reload_btn = gr.Button("Reload data", size="sm", scale=0, min_width=130, elem_id="app-reload")

        # ---- Scorecard handlers ----
        for comp in (sc_a, sc_b):
            comp.change(scorecard, [sc_a, sc_b], [sc_hero, sc_drives, sc_facts])

        def download():
            path, msg = report_download(root)
            return gr.update(value=path, visible=path is not None), msg

        sc_dl.click(download, None, [sc_file, sc_empty])

        # ---- Search handlers ----
        s_inputs = [s_box, s_video, s_staff]
        s_box.submit(search, s_inputs, [s_strip, s_results, s_hits])
        for comp in (s_video, s_staff):
            comp.change(search, s_inputs, [s_strip, s_results, s_hits])
        for btn, q in zip(quick, QUICK_SEARCHES.values()):
            btn.click(lambda q=q: q, None, s_box).then(search, s_inputs, [s_strip, s_results, s_hits])

        # ---- Review handlers ----
        filters = [video, f_label, f_needs, f_status, f_idle]
        decide_btns = [accept_btn, edit_btn, reject_btn, defer_btn]
        pane = [current, r_clip, panel, history, m_status, r_msg, r_suggest, r_acts, r_noticed, edit_to, *decide_btns]

        def video_label(video_id):
            return next((n for n, v in tl.video_choices()[0] if v == video_id), "")

        def refresh(video_id, label, needs, status_v, idle, keep=None):
            _, rows_v, msg = rv.timeline(video_id, label, needs, status_v, idle)
            value = keep if keep is not None and keep in {str(i) for i in range(len(rows_v))} else None
            empty = rv.queue_empty(video_id, rows_v, msg)
            return [rows_v, gr.update(choices=rv.queue_choices(rows_v), value=value),
                    rv.queue_head(video_label(video_id), rows_v) if video_id else msg, empty]

        def open_item(rows_v, value):
            if value in (None, "") or not rows_v:
                return [None, None, [], [], "", "", PICK_MOMENT, gr.update(choices=[], value=[]),
                        ui_html.noticed(None), gr.update(choices=[], value=None),
                        *[gr.update(interactive=False)] * len(decide_btns)]
            m, clip_path, panel_v, _, history_v, status_v, msg = rv.open(rows_v, int(value))
            choices, ticked = rv.activities(m)
            info = tl.summary(m["moment_id"]) if m else None
            edit_choices = rv.edit_choices(m, m["primary_label"]) if m else []
            can_decide = m is not None and m["video_id"] not in rv.local  # laptop-only videos are view only
            return [m, clip_path, panel_v, history_v, status_v, msg, rv.suggestion(m),
                    gr.update(choices=choices, value=ticked), ui_html.noticed(info),
                    gr.update(choices=rv.named(edit_choices), value=None),
                    *[gr.update(interactive=can_decide)] * len(decide_btns)]

        for comp in filters:
            comp.input(refresh, filters, [rows, queue, q_head, q_empty])
        queue.change(open_item, [rows, queue], pane)

        def next_item(rows_v, value):
            i = int(value) + 1 if value not in (None, "") else 0
            return gr.update(value=str(i)) if i < len(rows_v) else gr.update()

        next_btn.click(next_item, [rows, queue], queue)

        def saved(out):
            panel_v, history_v, status_v, msg = out
            if panel_v is None:  # refused: nothing saved, keep everything (including the note) as it was
                return [gr.update(), gr.update(), gr.update(), msg, gr.update()]
            return [panel_v, history_v, status_v, msg, ""]

        save_out = [panel, history, m_status, r_msg, note_box]
        after = (lambda v, lab, nd, st, idl, keep: refresh(v, lab, nd, st, idl, keep), [*filters, queue],
                 [rows, queue, q_head, q_empty])
        for btn, action in ((accept_btn, "accept"), (edit_btn, "edit"), (reject_btn, "reject"), (defer_btn, "defer")):
            btn.click(lambda c, who, ticked, to, n, a=action: saved(rv.decide(c, who, a, ticked, to, n)),
                      [current, reviewer, r_acts, edit_to, note_box], save_out).then(*after)
        for obs_type, btn in obs_btns.items():
            btn.click(lambda c, who, n, o=obs_type: saved(rv.add_observation(c, who, o, n)),
                      [current, reviewer, note_box], save_out).then(*after)
        add_btn.click(lambda c, who, lab, n: saved(rv.add_label(c, who, lab, n)),
                      [current, reviewer, add_pick, note_box], save_out).then(*after)
        obs_btn.click(lambda c, who, o, n: saved(rv.add_observation(c, who, o, n)),
                      [current, reviewer, obs_pick, note_box], save_out).then(*after)

        def open_from_search(hits, evt: gr.SelectData):
            if not hits or not 0 <= evt.index[0] < len(hits):
                return [gr.update()] * 4
            video_id, moment_id = hits[evt.index[0]]
            _, rows_v, _ = rv.timeline(video_id)
            i = next((k for k, r in enumerate(rows_v) if r.get("moment_id") == moment_id), None)
            return [gr.Tabs(selected="review"), gr.update(value=video_id), rows_v,
                    gr.update(choices=rv.queue_choices(rows_v), value=None if i is None else str(i))]

        s_results.select(open_from_search, [s_hits], [app_tabs, video, rows, queue]).then(
            lambda v, r: (rv.queue_head(video_label(v), r), ""), [video, rows], [q_head, q_empty])

        # ---- Blind handlers (one reviewer name, typed in either tab) ----
        reviewer.input(lambda v: v, reviewer, blind_reviewer)
        blind_reviewer.input(lambda v: v, blind_reviewer, reviewer)
        b_state = gr.State(None)
        outputs = [b_state, clip, progress, primary, *group_boxes, uncertain, status]
        start_btn.click(blind.start, [blind_reviewer, queue_pick], outputs)
        save_btn.click(blind.save, [b_state, primary, *group_boxes, uncertain], outputs)
        back_btn.click(blind.back, [b_state], outputs)
        replay_btn.click(blind.replay, [b_state], outputs)
        skip_btn.click(blind.skip, [b_state], outputs)
        start_btn.click(lambda: blind.queue_choices()[1], None, topup_note)

        # ---- Export ----
        ex_btn.click(lambda v: export_download(v, root)[::-1], [ex_verified], [ex_msg, ex_files])

        # ---- Reload everything ----
        def reload():
            blind.reload()
            rv.reload()
            tl.reload()
            ins.load()
            names, note = blind.queue_choices()
            choices, reason = tl.video_choices()
            a, b = ins.default_pair()
            g, d, f = scorecard(a, b)
            strip_v, rows_v, hits_v = search("", "", False)
            p_choices_v, p_note_v = proc.video_choices(root)
            return (gr.update(choices=names), note, gr.update(choices=choices), reason,
                    gr.update(choices=[("All videos", "")] + choices, value=""), reason,
                    gr.update(choices=phase_choices(), value=a), gr.update(choices=phase_choices(), value=b),
                    ins.subtitle() if ins.available else "", ins.message(), gr.update(visible=ins.available),
                    g, d, f, strip_v, rows_v, hits_v, local_note(root),
                    gr.update(choices=p_choices_v), p_note_v)

        reload_outputs = [queue_pick, topup_note, video, lock_note, s_video, s_lock, sc_a, sc_b, sc_sub, sc_empty, sc_dl,
                          sc_hero, sc_drives, sc_facts, s_strip, s_results, s_hits, sc_local,
                          p_video, p_note_md]
        reload_btn.click(reload, None, reload_outputs)

        # ---- Process a video: one run at a time, then refresh the browse tabs ----
        def run_process(video_id):
            yield from process_video(video_id, root, blind.labels)

        p_btn.click(run_process, [p_video], [p_status, p_preview, p_msg, p_moments], concurrency_limit=1).then(
            reload, None, reload_outputs)
    return demo


def theme():
    """Light theme from config/theme.yaml (no colour is written in code). IBM Plex comes from the font files
    bundled with Gradio and served by the app itself: nothing is fetched from the internet."""
    import gradio as gr

    p = load_theme()
    sans = [gr.themes.LocalFont("IBM Plex Sans", weights=(400, 600, 700)), "system-ui", "sans-serif"]
    mono = [gr.themes.LocalFont("IBM Plex Mono", weights=(400, 500, 600)), "ui-monospace", "monospace"]
    return gr.themes.Default(font=sans, font_mono=mono, radius_size=gr.themes.sizes.radius_sm).set(
        body_background_fill=p["background"], background_fill_primary=p["surface"],
        background_fill_secondary=p["background"], block_background_fill=p["surface"],
        block_border_color=p["border"], block_border_width="1px", block_radius="6px",
        border_color_primary=p["border"], input_border_color=p["border_strong"],
        body_text_color=p["text"], body_text_color_subdued=p["text_secondary"], table_text_color=p["text"],
        block_title_text_color=p["accent_deep"], block_label_text_color=p["text_secondary"],
        color_accent=p["accent"], color_accent_soft=p["accent_soft"], border_color_accent=p["accent"],
        link_text_color=p["accent_deep"], link_text_color_hover=p["accent"], link_text_color_visited=p["accent_deep"],
        button_primary_background_fill=p["accent"], button_primary_background_fill_hover=p["accent_deep"],
        button_primary_border_color=p["accent"], button_primary_text_color=p["surface"],
        button_primary_text_color_hover=p["surface"],
    )


def page_css() -> str:
    from tpa_index.ui_html import page_css as css

    return css()  # every colour from theme.yaml


def launch_kwargs(session, port: int) -> dict:
    """The only launch settings the app ever uses: loopback, no sharing, no monitoring, light theme."""
    return dict(server_name="127.0.0.1", server_port=port, share=False, enable_monitoring=False, inbrowser=False,
                allowed_paths=[str(session)], theme=theme(), js=PAGE_JS, css=page_css(), ssr_mode=False, quiet=True,
                footer_links=[])


def launch(port: int | None = None) -> None:
    from tpa_index.clips import ClipCutter, register_cleanup, session_dir

    from tpa_index.config import check_guideline

    pipeline = load_pipeline()
    print(check_guideline(load_labels()))  # refuses a vocabulary change without a guideline bump
    session = session_dir()
    # clips already inside Gradio's cache folder are served in place (no copy), so cut()'s deletion applies
    os.environ["GRADIO_TEMP_DIR"] = str(session)
    cutter = ClipCutter(session, pipeline["app"]["clip_height"])
    register_cleanup(cutter)
    port = port or pipeline["app"]["port"]
    print(f"TPA review app: http://127.0.0.1:{port}  (this machine only; Ctrl+C to stop)")
    build_app(cutter=cutter, pipeline=pipeline).launch(**launch_kwargs(session, port))
