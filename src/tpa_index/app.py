"""Local review app (Gradio): blind labeling and review. Binds 127.0.0.1 only; no sharing, analytics or monitoring.
Every label and group shown comes from labels.yaml / pipeline.yaml at runtime."""
import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # before gradio is imported anywhere

from tpa_index import review  # noqa: E402
from tpa_index.config import load_labels, load_pipeline, load_theme  # noqa: E402
from tpa_index.paths import redact  # noqa: E402

QUEUE_NAMES = {"Main sample": "main", "Overlap set (second labeler)": "overlap", "Top-up": "topup"}
REVIEWER_HINT = "staff first name or role"

# On load: force the light theme (Constitution X) and install the checklist shortcuts. Keys are matched by the
# "[k]" suffix of each choice's text, so no label name is written here.
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
    const blind = document.querySelector('#blind-checklist');
    if (!blind || blind.offsetParent === null) return;  // keys belong to the blind tab only
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

        self.moments = read_official("moments", root=self.root).to_pylist()
        self.segments = read_official("segments", root=self.root).to_pylist()
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
        videos = review.review_videos(self.root)
        return [(f"{v['video_id']} ({v['split']})", v["video_id"]) for v in videos], reason

    def label_filter_choices(self) -> list[tuple[str, str]]:
        return [(ANY, ANY)] + [(n, n) for n in self.vocab + self.observations]

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
        allowed = {v["video_id"] for v in review.review_videos(self.root)}
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
        video = next((v for v in review.review_videos(self.root) if v["video_id"] == r["video_id"]), None)
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


def build_app(root=None, cutter=None, labels=None, pipeline=None):
    import gradio as gr

    blind = BlindController(cutter, root, labels, pipeline)
    rv = ReviewController(cutter, root, blind.labels, blind.pipeline)
    cl = blind.checklist
    palette = load_theme()
    with gr.Blocks(analytics_enabled=False, title="TPA video index") as demo:
        demo.blind, demo.review = blind, rv
        gr.Markdown("## TPA video index (local only)")
        gr.Markdown(palette["credit"], elem_id="app-credit-header")
        with gr.Row():
            reviewer = gr.Textbox(label="Reviewer", placeholder=REVIEWER_HINT, elem_id="reviewer", scale=3)
            reload_btn = gr.Button("Reload data", scale=1)
        state = gr.State(None)
        with gr.Tab("Blind labeling"):
            names, note = blind.queue_choices()
            queue = gr.Radio(names, value="Main sample", label="Queue", elem_id="blind-queue")
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
        with gr.Tab("Review"):
            v_choices, lock_reason = rv.video_choices()
            video = gr.Dropdown(v_choices, value=None, label="Video", elem_id="review-video")
            lock_note = gr.Markdown(lock_reason, elem_id="review-lock")
            with gr.Row():
                f_label = gr.Dropdown(rv.label_filter_choices(), value=ANY, label="Label", elem_id="review-f-label")
                f_needs = gr.Radio([ANY, "yes", "no"], value=ANY, label="Needs review", elem_id="review-f-needs")
                f_status = gr.Dropdown([ANY, *review.STATUSES], value=ANY, label="Status", elem_id="review-f-status")
                f_idle = gr.Checkbox(False, label="Show idle", elem_id="review-f-idle")
            timeline = gr.Dataframe(headers=TIMELINE_HEADERS, interactive=False, label="Moments (select one)",
                                    elem_id="review-timeline")
            rows = gr.State([])
            current = gr.State(None)
            r_msg = gr.Markdown(elem_id="review-msg")
            r_clip = gr.Video(autoplay=True, interactive=False, label="Moment (with context)", elem_id="review-clip")
            m_status = gr.Markdown(elem_id="review-status")
            panel = gr.Dataframe(headers=PANEL_HEADERS, interactive=False, label="AI labels", elem_id="review-panel")
            ai_pick = gr.Radio([], label="AI label to review", elem_id="review-ai-label")
            with gr.Row():
                accept_btn = gr.Button("Accept", elem_id="review-accept")
                reject_btn = gr.Button("Reject", elem_id="review-reject")
                defer_btn = gr.Button("Defer", elem_id="review-defer")
                edit_to = gr.Dropdown([], label="Edit to", elem_id="review-edit-to")
                edit_btn = gr.Button("Edit", elem_id="review-edit")
            with gr.Row():
                add_pick = gr.Dropdown(rv.vocab, label="Add label", elem_id="review-add-label")
                add_btn = gr.Button("Add label", elem_id="review-add")
                obs_pick = gr.Dropdown(rv.observations, label="Add observation", elem_id="review-add-obs")
                obs_btn = gr.Button("Add observation", elem_id="review-obs")
            gr.Markdown(REMINDER, elem_id="review-reminder")
            note_box = gr.Textbox(label="Note (optional)", max_lines=2, elem_id="review-note")
            history = gr.Dataframe(headers=HISTORY_HEADERS, interactive=False, label="Review history",
                                   elem_id="review-history")

        outputs = [state, clip, progress, primary, *group_boxes, uncertain, status]
        start_btn.click(blind.start, [reviewer, queue], outputs)
        save_btn.click(blind.save, [state, primary, *group_boxes, uncertain], outputs)
        back_btn.click(blind.back, [state], outputs)
        replay_btn.click(blind.replay, [state], outputs)
        skip_btn.click(blind.skip, [state], outputs)
        filters = [video, f_label, f_needs, f_status, f_idle]
        for comp in filters:
            comp.change(rv.timeline, filters, [timeline, rows, r_msg])

        def on_select(rows_value, evt: gr.SelectData):
            current_m, clip_path, panel_v, ai_choices, history_v, status_v, msg = rv.open(rows_value, evt.index[0])
            return [current_m, clip_path, panel_v, gr.update(choices=ai_choices, value=None), history_v, status_v,
                    msg, gr.update(choices=[], value=None)]

        timeline.select(on_select, [rows], [current, r_clip, panel, ai_pick, history, m_status, r_msg, edit_to])
        ai_pick.change(lambda c, lab: gr.update(choices=rv.edit_choices(c, lab), value=None), [current, ai_pick], edit_to)

        def saved(out):
            panel_v, history_v, status_v, msg = out
            if panel_v is None:  # refused: nothing saved, keep everything (including the note) as it was
                return [gr.update(), gr.update(), gr.update(), msg, gr.update()]
            return [panel_v, history_v, status_v, msg, ""]

        save_out = [panel, history, m_status, r_msg, note_box]
        for btn, action in ((accept_btn, "accept"), (reject_btn, "reject"), (defer_btn, "defer")):
            btn.click(lambda c, who, lab, n, a=action: saved(rv.act(c, who, lab, a, note=n)),
                      [current, reviewer, ai_pick, note_box], save_out).then(rv.timeline, filters, [timeline, rows, r_msg])
        edit_btn.click(lambda c, who, lab, to, n: saved(rv.act(c, who, lab, "edit", edit_to=to, note=n)),
                       [current, reviewer, ai_pick, edit_to, note_box], save_out).then(
            rv.timeline, filters, [timeline, rows, r_msg])
        add_btn.click(lambda c, who, lab, n: saved(rv.add_label(c, who, lab, n)),
                      [current, reviewer, add_pick, note_box], save_out).then(rv.timeline, filters, [timeline, rows, r_msg])
        obs_btn.click(lambda c, who, o, n: saved(rv.add_observation(c, who, o, n)),
                      [current, reviewer, obs_pick, note_box], save_out).then(rv.timeline, filters, [timeline, rows, r_msg])

        def reload():
            blind.reload()
            rv.reload()
            names, note = blind.queue_choices()
            v_choices, reason = rv.video_choices()
            return gr.update(choices=names), note, gr.update(choices=v_choices), reason

        reload_btn.click(reload, None, [queue, topup_note, video, lock_note])
        start_btn.click(lambda: blind.queue_choices()[1], None, topup_note)
        gr.Markdown(palette["credit"], elem_id="app-credit-footer")
    return demo


def theme():
    """Light theme from config/theme.yaml (no colour is written in code)."""
    import gradio as gr

    p = load_theme()
    return gr.themes.Default(font=["system-ui", "sans-serif"], font_mono=["ui-monospace", "monospace"]).set(
        body_background_fill=p["background"], background_fill_primary=p["surface"],
        background_fill_secondary=p["background"], block_background_fill=p["surface"],
        body_text_color=p["text"], body_text_color_subdued=p["text_secondary"], table_text_color=p["text"],
        block_title_text_color=p["accent_deep"], block_label_text_color=p["text_secondary"],
        color_accent=p["accent"], color_accent_soft=p["background"], border_color_accent=p["accent"],
        link_text_color=p["accent_deep"], link_text_color_hover=p["accent"], link_text_color_visited=p["accent_deep"],
        button_primary_background_fill=p["accent"], button_primary_background_fill_hover=p["accent_deep"],
        button_primary_border_color=p["accent"], button_primary_text_color=p["surface"],
        button_primary_text_color_hover=p["surface"],
    )


def page_css() -> str:
    return f"h1, h2, h3 {{ color: {load_theme()['accent_deep']}; }}"  # deep accent for headers, from theme.yaml


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
