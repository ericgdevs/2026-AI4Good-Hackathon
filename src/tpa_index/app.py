"""Local review app (Gradio): blind labeling and review. Binds 127.0.0.1 only; no sharing, analytics or monitoring.
Every label and group shown comes from labels.yaml / pipeline.yaml at runtime."""
import json
import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # before gradio is imported anywhere

from tpa_index import review  # noqa: E402
from tpa_index.config import load_labels, load_pipeline  # noqa: E402
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
        video = self.videos.get(item["video_id"])
        if video is None or self.cutter is None:
            return None, "Clip unavailable: video not in the official runs. Use Skip."
        try:
            return str(self.cutter.cut(video, item["start_s"], item["end_s"])), ""
        except Exception as e:
            return None, redact(f"Clip unavailable ({e}). Use Skip.")

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
        reviewer = (reviewer or "").strip()
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


def build_app(root=None, cutter=None, labels=None, pipeline=None):
    import gradio as gr

    blind = BlindController(cutter, root, labels, pipeline)
    cl = blind.checklist
    with gr.Blocks(analytics_enabled=False, title="TPA review") as demo:
        demo.blind = blind
        gr.Markdown("## TPA review (local only)")
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
            gr.Markdown("Review mode arrives in the next step.")

        outputs = [state, clip, progress, primary, *group_boxes, uncertain, status]
        start_btn.click(blind.start, [reviewer, queue], outputs)
        save_btn.click(blind.save, [state, primary, *group_boxes, uncertain], outputs)
        back_btn.click(blind.back, [state], outputs)
        replay_btn.click(blind.replay, [state], outputs)
        skip_btn.click(blind.skip, [state], outputs)
        def reload():
            blind.reload()
            names, note = blind.queue_choices()
            return gr.update(choices=names), note

        reload_btn.click(reload, None, [queue, topup_note])
        start_btn.click(lambda: blind.queue_choices()[1], None, topup_note)
    return demo


def theme():
    import gradio as gr

    return gr.themes.Default(font=["system-ui", "sans-serif"], font_mono=["ui-monospace", "monospace"])


def launch_kwargs(session, port: int) -> dict:
    """The only launch settings the app ever uses: loopback, no sharing, no monitoring, light theme."""
    return dict(server_name="127.0.0.1", server_port=port, share=False, enable_monitoring=False, inbrowser=False,
                allowed_paths=[str(session)], theme=theme(), js=PAGE_JS, ssr_mode=False, quiet=True,
                footer_links=[])


def launch(port: int | None = None) -> None:
    from tpa_index.clips import ClipCutter, register_cleanup, session_dir

    pipeline = load_pipeline()
    session = session_dir()
    os.environ["GRADIO_TEMP_DIR"] = str(session / "gradio")  # Gradio's own copies of clips stay in the session
    cutter = ClipCutter(session, pipeline["app"]["clip_height"])
    register_cleanup(cutter)
    port = port or pipeline["app"]["port"]
    print(f"TPA review app: http://127.0.0.1:{port}  (this machine only; Ctrl+C to stop)")
    build_app(cutter=cutter, pipeline=pipeline).launch(**launch_kwargs(session, port))


def page_config_json(demo) -> str:
    """The page's component config (for tests): what the browser receives."""
    return json.dumps(demo.config, default=str)
