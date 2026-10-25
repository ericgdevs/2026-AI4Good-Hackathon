"""Review-app logic: blind queues, label roles, review rows and state. Pure functions, no UI, no footage.
Every label and group name comes from labels.yaml / pipeline.yaml at runtime; none is written in this file."""
import string

from tpa_index.schema import LABEL_ROLES

PRIMARY, SECONDARY, INDICATOR, OBSERVATION = LABEL_ROLES

# key order for checklist shortcuts: primary labels first, then every other label (u is the uncertain toggle)
KEY_ORDER = list("1234567890") + [c for c in string.ascii_lowercase if c != "u"]


def _selectable(lab: dict, pipeline: dict) -> bool:
    """Blind labels can be model or staff labels, never rules flags or the null (noise) group."""
    return lab.get("source", "model") != "rules" and lab["group"] != pipeline["calibrate"]["null_group"]


def checklist(labels: dict, pipeline: dict) -> dict:
    """{"primary": [...], "groups": [(group, [...]), ...], "uncertain": label} in labels.yaml order."""
    app, primary_group = pipeline["app"], pipeline["moments"]["primary_group"]
    pick = lambda g: [lab["name"] for lab in labels["labels"] if lab["group"] == g and _selectable(lab, pipeline)]  # noqa: E731
    return {"primary": pick(primary_group),
            "groups": [(g, pick(g)) for g in app["blind_groups"] if g != primary_group],
            "uncertain": app["uncertain_label"]}


def key_map(cl: dict) -> tuple[dict[str, str], list[str]]:
    """(key -> label, labels left without a key). Primary labels get keys first."""
    order = cl["primary"] + [name for _, names in cl["groups"] for name in names]
    return dict(zip(KEY_ORDER, order)), order[len(KEY_ORDER):]


def role_of(label: str, labels: dict, pipeline: dict, chosen_primary: str | None = None) -> str:
    if label == chosen_primary:
        return PRIMARY
    group = next((lab["group"] for lab in labels["labels"] if lab["name"] == label), None)
    return INDICATOR if group in pipeline["app"]["indicator_groups"] else SECONDARY


def latest_state(rows, mode: str | None = None) -> dict[tuple[str, str], dict]:
    """(target_id, label) -> newest row (by created_at, then review_id), optionally for one mode."""
    rows = rows.to_pylist() if hasattr(rows, "to_pylist") else list(rows)
    out: dict[tuple[str, str], dict] = {}
    for r in sorted(rows, key=lambda r: (r["created_at"], r["review_id"])):
        if mode is None or r["mode"] == mode:
            out[(r["target_id"], r["label"])] = r
    return out


# ---- blind queues ------------------------------------------------------------------------------------------------
def _official(table: str, root=None):
    """Official rows; blind mode needs only segments, so fall back to segment-stage runs before predict exists."""
    from tpa_index.schema import read_official

    tbl = read_official(table, root=root)
    return tbl if tbl.num_rows else read_official(table, stage="segment", root=root)


def eval_segments(root=None) -> list[dict]:
    videos = _official("videos", root).to_pylist()
    split_of = {v["video_id"]: v["split"] for v in videos}
    return [s for s in _official("segments", root).to_pylist() if split_of.get(s["video_id"]) == "eval"]


def load_queue(queue: str, root=None) -> list[dict]:
    from tpa_index.schema import read_blind_queue

    return read_blind_queue(queue, root).to_pylist()


def build_main_queues(app_cfg: dict, root=None) -> int:
    """Seeded random sample of eval segments, saved once; the overlap set is its first overlap_size items."""
    from datetime import datetime, timezone

    import numpy as np

    from tpa_index.config import ConfigError
    from tpa_index.schema import append_blind_queue

    existing = load_queue("main", root)
    if existing:
        return len(existing)
    segs = sorted(eval_segments(root), key=lambda s: s["segment_id"])
    if not segs:
        raise ConfigError("no eval-split segments in official runs")
    order = np.random.default_rng(app_cfg["seed"]).permutation(len(segs))[:app_cfg["sample_size"]]
    now = datetime.now(timezone.utc)

    def rows(queue, picked):
        return [{"queue": queue, "position": i, "segment_id": s["segment_id"], "video_id": s["video_id"],
                 "start_s": s["start_s"], "end_s": s["end_s"], "target_label": None, "seed": app_cfg["seed"],
                 "created_at": now} for i, s in enumerate(picked)]

    main = [segs[int(i)] for i in order]
    append_blind_queue(rows("main", main), root)
    append_blind_queue(rows("overlap", main[:app_cfg["overlap_size"]]), root)
    return len(main)


def _done_segments(queue: str, reviewer: str, root=None) -> set[str]:
    """Blind-labeled segments that count as done. The main sample is shared: a segment any reviewer labeled is done,
    so reviewers split it. Overlap and top-up stay per reviewer (overlap measures agreement between labelers)."""
    from tpa_index.schema import read_reviews

    return {r["target_id"] for r in read_reviews(root).to_pylist()
            if r["mode"] == "blind" and (queue == "main" or r["reviewer"] == reviewer)}


def next_position(queue: str, reviewer: str, root=None) -> int | None:
    """First queue position still to label (resume point); None when finished."""
    done = _done_segments(queue, reviewer, root)
    return next((q["position"] for q in load_queue(queue, root) if q["segment_id"] not in done), None)


def progress(queue: str, reviewer: str, root=None) -> tuple[int, int]:
    """(done, total); for the main sample, done is the team total."""
    items, done = load_queue(queue, root), _done_segments(queue, reviewer, root)
    return sum(q["segment_id"] in done for q in items), len(items)


def queue_missing(queue: str, root=None) -> list[str]:
    """Queue segments no longer in the official runs (reported and skipped, never replaced)."""
    present = {s["segment_id"] for s in _official("segments", root).to_pylist()}
    return [q["segment_id"] for q in load_queue(queue, root) if q["segment_id"] not in present]


# ---- blind save rows -----------------------------------------------------------------------------------------------
def _row(target_type, target_id, label, role, action, new_label, source, mode, reviewer, labels, note=None):
    return {"target_type": target_type, "target_id": target_id, "label": label, "label_role": role, "action": action,
            "new_label": new_label, "source": source, "mode": mode, "guideline_version": labels["guideline_version"],
            "reviewer": reviewer, "note": note}


def blind_rows(segment: dict, primary: str | None, chosen: list[str], uncertain: bool, reviewer: str, labels: dict,
               pipeline: dict, previous: dict[str, dict] | None = None) -> list[dict]:
    """Rows for one blind save. previous: this reviewer's latest blind rows on the segment (label -> row);
    unchanged labels write nothing, new or re-roled labels write add, unticked labels write reject."""
    if not primary and not uncertain:
        raise ValueError("choose a primary activity or the uncertain option")
    wanted = {label: role_of(label, labels, pipeline, primary) for label in chosen}
    if primary:
        wanted[primary] = PRIMARY
    if uncertain:
        wanted[pipeline["app"]["uncertain_label"]] = role_of(pipeline["app"]["uncertain_label"], labels, pipeline)
    previous = {k: r for k, r in (previous or {}).items() if r["action"] == "add"}
    sid = segment["segment_id"]
    rows = [_row("segment", sid, label, role, "add", label, "staff", "blind", reviewer, labels)
            for label, role in wanted.items() if previous.get(label, {}).get("label_role") != role]
    rows += [_row("segment", sid, label, r["label_role"], "reject", None, "staff", "blind", reviewer, labels)
             for label, r in previous.items() if label not in wanted]
    return rows


def skip_row(segment: dict, reviewer: str, labels: dict, pipeline: dict) -> dict:
    obs = pipeline["app"]["skip_observation"]
    return _row("segment", segment["segment_id"], obs, OBSERVATION, "add", obs, "staff", "blind", reviewer, labels)


def previous_blind(segment_id: str, reviewer: str, root=None) -> dict[str, dict]:
    """This reviewer's latest blind row per label on one segment."""
    from tpa_index.schema import read_reviews

    rows = [r for r in read_reviews(root).to_pylist()
            if r["mode"] == "blind" and r["reviewer"] == reviewer and r["target_id"] == segment_id]
    return {label: r for (_, label), r in latest_state(rows).items()}


# ---- top-up (rare labels, model-picked, shown blind) ----------------------------------------------------------------
def topup_available(root=None) -> tuple[bool, str]:
    """The top-up picks segments by model score, so it needs official predict runs for every eval video."""
    from tpa_index.schema import _official_by_video, load_official

    eval_videos = {q["video_id"] for q in load_queue("main", root)}
    if not eval_videos:
        return False, "Start the main sample first."
    covered = set(_official_by_video(load_official(), "predict", root))
    missing = eval_videos - covered
    if missing:
        return False, "The top-up needs official predict runs for the eval videos (missing for some videos)."
    return True, ""


def build_topup_queue(app_cfg: dict, labels: dict, pipeline: dict, root=None) -> int:
    """Up to topup_per_label unlabeled eval segments per rare label, by official model score; shuffled, saved once.
    The target label is stored for metrics (precision only) and never shown."""
    from datetime import datetime, timezone

    import numpy as np

    from tpa_index.config import ConfigError
    from tpa_index.schema import append_blind_queue, read_official, read_reviews

    existing = load_queue("topup", root)
    if existing:
        return len(existing)
    ok, reason = topup_available(root)
    if not ok:
        raise ConfigError(reason)
    main = load_queue("main", root)
    queued = {q["segment_id"] for q in main} | {q["segment_id"] for q in load_queue("overlap", root)}
    blind = [r for r in read_reviews(root).to_pylist() if r["mode"] == "blind" and r["target_id"] in queued]
    positives: dict[str, int] = {}
    for (_, label), r in latest_state(blind).items():
        if r["action"] == "add":
            positives[label] = positives.get(label, 0) + 1
    candidates = [lab["name"] for lab in labels["labels"]
                  if lab["group"] in app_cfg["blind_groups"] and lab.get("source", "model") == "model"
                  and lab["group"] != pipeline["calibrate"]["null_group"]]
    rare = [name for name in candidates if positives.get(name, 0) < app_cfg["rare_min_positives"]]
    eval_ids = {s["segment_id"]: s for s in eval_segments(root)}
    scores: dict[str, list[tuple[float, str]]] = {name: [] for name in rare}
    for p in read_official("predictions", root=root).to_pylist():
        if p["model"] != "rules" and p["label"] in scores and p["segment_id"] in eval_ids and p["segment_id"] not in queued:
            scores[p["label"]].append((p["score"], p["segment_id"]))
    picked: dict[str, str] = {}
    for name in rare:
        for _, sid in sorted(scores[name], reverse=True)[:app_cfg["topup_per_label"]]:
            picked.setdefault(sid, name)  # first rare label wins a shared segment
    ids = sorted(picked)
    order = np.random.default_rng(app_cfg["seed"] + 1).permutation(len(ids)) if ids else []
    now = datetime.now(timezone.utc)
    rows = [{"queue": "topup", "position": pos, "segment_id": ids[int(i)], "video_id": eval_ids[ids[int(i)]]["video_id"],
             "start_s": eval_ids[ids[int(i)]]["start_s"], "end_s": eval_ids[ids[int(i)]]["end_s"],
             "target_label": picked[ids[int(i)]], "seed": app_cfg["seed"], "created_at": now}
            for pos, i in enumerate(order)]
    if rows:
        append_blind_queue(rows, root)
    return len(rows)


# ---- review mode: moments, state, filters (research §9) -----------------------------------------------------------
UNVALIDATED = "AI suggestion, unvalidated"
STATUSES = ("unreviewed", "partly", "done", "deferred")
DONE_ACTIONS = {"accept", "edit", "reject"}


def ai_labels(moment: dict) -> list[tuple[str, str]]:
    """(label, role) for every AI label on a moment: primary, then secondaries, then indicators."""
    return ([(moment["primary_label"], PRIMARY)] + [(lab, SECONDARY) for lab in moment["secondary_labels"] or []]
            + [(lab, INDICATOR) for lab in moment["indicators"] or []])


def moment_status(moment: dict, state: dict) -> str:
    """unreviewed | deferred | done | partly, from the latest review row per AI label."""
    actions = [state[(moment["moment_id"], lab)]["action"] for lab, _ in ai_labels(moment)
               if (moment["moment_id"], lab) in state]
    if not actions:
        return "unreviewed"
    if "defer" in actions:
        return "deferred"
    if len(actions) == len(ai_labels(moment)) and set(actions) <= DONE_ACTIONS:
        return "done"
    return "partly"


def filter_moments(moments, state: dict, label=None, needs_review=None, status=None) -> list[dict]:
    """Moments sorted by time. label matches an AI label or a label/observation staff added or edited to."""
    staff = {t for (t, _), r in state.items() if r["new_label"] == label}

    def has(m):
        return m["moment_id"] in staff or label in {lab for lab, _ in ai_labels(m)}

    return sorted((m for m in moments
                   if (label is None or has(m)) and (needs_review is None or m["needs_review"] == needs_review)
                   and (status is None or moment_status(m, state) == status)),
                  key=lambda m: (m["video_id"], m["start_s"]))


def eval_unlocked(root=None) -> tuple[bool, str]:
    """Eval videos open for review only after the main sample has a blind save per segment and the overlap set
    has blind saves from at least two different reviewers per segment (FR-014a)."""
    from tpa_index.schema import read_reviews

    main, overlap = load_queue("main", root), load_queue("overlap", root)
    if not main:
        return False, "Eval videos are hidden until blind labeling of the main sample has started and finished."
    reviewers: dict[str, set[str]] = {}
    for r in read_reviews(root).to_pylist():
        if r["mode"] == "blind":
            reviewers.setdefault(r["target_id"], set()).add(r["reviewer"])
    main_left = sum(not reviewers.get(q["segment_id"]) for q in main)
    overlap_left = sum(len(reviewers.get(q["segment_id"], ())) < 2 for q in overlap)
    if main_left or overlap_left:
        return False, (f"Eval videos are hidden until blind labeling is done: {main_left} main-sample segment(s) "
                       f"unlabeled, {overlap_left} overlap segment(s) without a second reviewer.")
    return True, ""


def review_videos(root=None, include_local: bool = False) -> list[dict]:
    """Official videos (plus laptop-only videos when include_local), sorted by id, without eval-split videos while
    blind labeling is unfinished."""
    from tpa_index.schema import read_official

    unlocked, _ = eval_unlocked(root)
    videos = read_official("videos", root=root, include_local=include_local).to_pylist()
    return sorted((v for v in videos if unlocked or v["split"] != "eval"), key=lambda v: v["video_id"])


def badge(row: dict | None) -> str:
    """The unvalidated marker, or the latest review state with reviewer and time."""
    if row is None:
        return UNVALIDATED
    what = f"edit → {row['new_label']}" if row["action"] == "edit" else row["action"]
    return f"{what} by {row['reviewer']}, {local_time(row['created_at'])}"


def local_time(ts) -> str:
    """Stored times are UTC; staff read them in the machine's local time."""
    return f"{ts.astimezone():%Y-%m-%d %H:%M}"


def label_panel(moment: dict, state: dict, labels: dict, confidences: dict[str, float],
                thresholds: dict[str, float]) -> list[dict]:
    """One row per AI label: primary first, then each role by strength (score / threshold), strongest first.
    The primary's confidence is the moment's own; labels missing from labels.yaml are flagged retired."""
    from tpa_index.score import strengths

    known = {lab["name"] for lab in labels["labels"]}
    conf = {**confidences, moment["primary_label"]: moment["confidence"]}
    st = strengths({k: v for k, v in conf.items() if k in thresholds}, thresholds)
    order = {PRIMARY: 0, SECONDARY: 1, INDICATOR: 2}
    pairs = sorted(ai_labels(moment), key=lambda p: (order[p[1]], -st.get(p[0], 0.0)))
    return [{"label": lab, "role": role, "confidence": conf.get(lab), "retired": lab not in known,
             "badge": badge(state.get((moment["moment_id"], lab)))} for lab, role in pairs]


# ---- review rows -------------------------------------------------------------------------------------------------
REVIEW_ACTIONS = ("accept", "edit", "reject", "defer", "add")


def review_row(moment: dict, label: str, role: str, action: str, reviewer: str, labels: dict, new_label=None,
               note=None, source="ai") -> dict:
    """One mode=review row on a moment. edit needs new_label; accept, reject and defer forbid it; add (staff)
    records the label itself as new_label."""
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"unknown review action {action!r}")
    if action == "edit" and not new_label:
        raise ValueError("edit needs the new label")
    if action in ("accept", "reject", "defer") and new_label:
        raise ValueError(f"{action} takes no new label")
    if action == "add":
        new_label, source = label, "staff"
    return _row("moment", moment["moment_id"], label, role, action, new_label, source, "review", reviewer, labels,
                note or None)


def observation_row(moment: dict, obs_type: str, reviewer: str, labels: dict, note=None) -> dict:
    from tpa_index.config import observation_types

    if obs_type not in observation_types(labels):
        raise ValueError(f"unknown observation type {obs_type!r}")
    return _row("moment", moment["moment_id"], obs_type, OBSERVATION, "add", obs_type, "staff", "review", reviewer,
                labels, note or None)
