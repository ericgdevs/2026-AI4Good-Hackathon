"""005: one per-segment view of everything evaluate and insights need, built from the official runs and the review
log. Aggregates and opaque IDs only: video filenames and transcript text are never read into it."""
from collections import Counter

import pandas as pd

from tpa_index.config import ConfigError, model_labels, required_threshold

GROUPS = ("activity", "category", "actor")
UNCERTAIN = "uncertain_mixed"


def main_model(predictions) -> str | None:
    """Most common non-rules model in a predictions table (the visual model)."""
    counts = Counter(m for m in predictions["model"].to_pylist() if m != "rules")
    return counts.most_common(1)[0][0] if counts else None


def current_phases() -> dict[str, str]:
    """video_id -> phase from the local videos config when it exists (validated against video_phases), so a phase
    edited after segmenting is used; {} otherwise, and the phase stored in the official runs applies. Only the phase
    is kept: filenames never leave this function. Never falls back to the example config."""
    from tpa_index import config

    if not config.LOCAL_VIDEOS.exists():
        return {}
    return {v["video_id"]: v["phase"] for v in config.load_videos(config.LOCAL_VIDEOS)}


def _labels_by_group(labels) -> dict[str, list[str]]:
    return {g: [lab["name"] for lab in model_labels(labels, {g})] for g in GROUPS}


def _rule_clears(pipeline) -> set[str]:
    """Visual labels named in `clears:` conditions anywhere in the insights rules."""
    found = set()

    def walk(c):
        if isinstance(c, dict):
            for k, v in c.items():
                if k == "clears":
                    found.add(v)
                else:
                    walk(v)
        elif isinstance(c, list):
            for x in c:
                walk(x)
    walk(pipeline["insights"])
    return found


def check_inputs(kind: str, pipeline: dict, labels: dict, root=None) -> None:
    """Fail loudly (one ConfigError listing every problem) if an official stage or a calibrated visual threshold
    is missing. Audio uses the fixed insights.audio_threshold, so it needs no calibrated value."""
    from tpa_index.schema import _official_by_video, load_official, read_official

    videos = read_official("videos", root=root).to_pylist()
    scope = sorted(v["video_id"] for v in videos if kind == "insights" or v["split"] == "eval")
    if not scope:
        raise ConfigError("no official videos" + (" on the eval split" if kind == "evaluate" else ""))
    official, problems = load_official(), []
    covered = set(_official_by_video(official, "predict", root)) | set(_official_by_video(official, "moments", root))
    if set(scope) - covered:
        problems.append(f"official predict runs missing for: {', '.join(sorted(set(scope) - covered))}")
    stages = ["audio", "people"] + (["transcript", "expression"] if kind == "insights" else [])
    for stage in stages:
        missing = sorted(set(scope) - set(_official_by_video(official, stage, root)))
        if missing:
            problems.append(f"official {stage} runs missing for: {', '.join(missing)}")
    model = main_model(read_official("predictions", root=root))
    if model is None:
        problems.append("no model predictions in the official runs")
    else:
        needed = [lab for g in GROUPS for lab in _labels_by_group(labels)[g]] + sorted(_rule_clears(pipeline))
        for label in dict.fromkeys(needed):
            try:
                required_threshold(pipeline, model, label)
            except ConfigError as e:
                problems.append(str(e))
    if kind == "evaluate" and not problems:
        human = human_labels(labels, pipeline, root)
        eval_ids = {v["video_id"] for v in videos if v["split"] == "eval"}
        if human.empty or not (human["video_id"].isin(eval_ids) & (human["queue"] == "main")).any():
            problems.append("no blind labels on eval videos")
    if problems:
        raise ConfigError("; ".join(problems))


def _pivot(table) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for sid, label, score in zip(table["segment_id"].to_pylist(), table["label"].to_pylist(),
                                 table["score"].to_pylist()):
        out.setdefault(sid, {})[label] = score
    return out


def segment_view(pipeline: dict, labels: dict, root=None) -> pd.DataFrame:
    """One row per official segment: time slice, model calls, flags and 004 signals (no filenames, no text)."""
    import pyarrow.compute as pc

    from tpa_index.schema import _official_by_video, load_official, read_official, read_signals
    from tpa_index.score import strengths

    videos = {v["video_id"]: v for v in read_official("videos", root=root).to_pylist()}
    current = current_phases()
    segs = sorted(read_signals(root).to_pylist(), key=lambda s: (s["video_id"], s["start_s"]))
    preds = read_official("predictions", root=root)
    model = main_model(preds)
    scores = _pivot(preds.filter(pc.equal(preds["model"], model)))
    flags = _pivot(preds.filter(pc.and_(pc.equal(preds["model"], "rules"), pc.equal(preds["label"], UNCERTAIN))))
    groups = _labels_by_group(labels)
    thr = {lab: required_threshold(pipeline, model, lab) for g in GROUPS for lab in groups[g]}
    audio = _pivot(read_official("predictions", stage="audio", root=root))
    audio_videos = set(_official_by_video(load_official(), "audio", root))
    expr = _pivot(read_official("predictions", stage="expression", root=root))
    expr_labels = list(pipeline["signals"]["expression"]["mapping"])
    cut = pipeline["insights"]["audio_threshold"]
    audio_labels = [lab["name"] for lab in model_labels(labels, {"audio"})]
    moments: dict[str, list[dict]] = {}
    for m in read_official("moments", root=root).to_pylist():
        moments.setdefault(m["video_id"], []).append(m)

    rows = []
    for i, s in enumerate(segs):
        vid = s["video_id"]
        nxt = segs[i + 1] if i + 1 < len(segs) and segs[i + 1]["video_id"] == vid else None
        sc = scores.get(s["segment_id"], {})
        st = {g: strengths({lab: sc.get(lab, 0.0) for lab in groups[g]}, thr) for g in GROUPS}
        moment = next((m for m in moments.get(vid, []) if m["start_s"] <= s["start_s"] and s["end_s"] <= m["end_s"]),
                      None)
        primary = max(st["activity"], key=st["activity"].get) if st["activity"] and moment else UNCERTAIN
        a = audio.get(s["segment_id"], {})
        e = expr.get(s["segment_id"], {})
        faces = s["faces_detected"]
        rows.append({
            "segment_id": s["segment_id"], "video_id": vid, "split": videos[vid]["split"],
            "phase": current.get(vid, videos[vid]["phase"]),
            "phase_source": "local config" if vid in current else "stored runs",
            "start_s": s["start_s"], "end_s": s["end_s"],
            "slice_s": (min(s["end_s"], nxt["start_s"]) if nxt else s["end_s"]) - s["start_s"],
            "is_idle": bool(s["is_idle"]), "person_count": s["person_count"], "has_speech": s["has_speech"],
            "faces_detected": faces,
            "strengths": st, "model_cleared": {g: set(st[g]) for g in GROUPS},
            "model_primary": primary, "model_strength": st["activity"].get(primary, 0.0),
            "uncertain_flag": flags.get(s["segment_id"], {}).get(UNCERTAIN, 0.0) >= 0.5,
            "needs_review": bool(moment and moment["needs_review"]), "no_call": moment is None,
            "moment_id": moment["moment_id"] if moment else None,
            "audio_ok": {lab: (a.get(lab, 0.0) >= cut if vid in audio_videos else None) for lab in audio_labels},
            "expr_top": (max(expr_labels, key=lambda lab: e.get(lab, 0.0)) if faces and e else None),
            "transcript_chars": s["transcript_chars"],  # a count only; transcript text is never read
            "visual_scores": dict(sc), "audio_scores": dict(a) if vid in audio_videos else None,
            "expr_scores": dict(e) if faces and e else None,
        })
    return pd.DataFrame(rows)


def human_labels(labels: dict, pipeline: dict, root=None) -> pd.DataFrame:
    """Blind labels per segment (latest state per label; a label counts when its latest action is add)."""
    from tpa_index.review import latest_state
    from tpa_index.schema import read_blind_queue, read_reviews

    group_of = {lab["name"]: lab["group"] for lab in labels["labels"]}
    skip = pipeline["app"]["skip_observation"]
    per: dict[str, dict] = {}
    for (sid, label), r in latest_state(read_reviews(root), mode="blind").items():
        if r["target_type"] != "segment" or r["action"] != "add":
            continue
        d = per.setdefault(sid, {"primary": None, "uncertain": False, "labels": set(), "skip": False})
        if label == skip:
            d["skip"] = True
        elif label == UNCERTAIN:
            d["uncertain"] = True
        else:
            d["labels"].add(label)
            if r["label_role"] == "primary":
                d["primary"] = label
    queues = {q: set(read_blind_queue(q, root)["segment_id"].to_pylist()) for q in ("main", "overlap", "topup")}
    rows = []
    for sid, d in per.items():
        queue = "main" if sid in queues["main"] | queues["overlap"] else "topup" if sid in queues["topup"] else None
        if queue is None:
            continue
        by = {g: {lab for lab in d["labels"] if group_of.get(lab) == g} for g in (*GROUPS, "indicator")}
        rows.append({"segment_id": sid, "video_id": sid.rsplit("_", 1)[0], "queue": queue,
                     "skipped": d["skip"] and not d["labels"] and not d["uncertain"],
                     "primary": d["primary"] or (UNCERTAIN if d["uncertain"] else None),
                     "activities": by["activity"], "categories": by["category"], "actors": by["actor"],
                     "indicators": by["indicator"]})
    return pd.DataFrame(rows, columns=["segment_id", "video_id", "queue", "skipped", "primary", "activities",
                                       "categories", "actors", "indicators"])


def verified_labels(view: pd.DataFrame, labels: dict, pipeline: dict, root=None) -> dict[str, dict]:
    """segment_id -> {group: verified label set}. A blind label verifies every group of its segment (the labeler saw
    the full checklist); otherwise review-mode decisions on the containing moment adjust the model's set for the
    groups they touch. Segments with neither are absent (the model call is used)."""
    from tpa_index.review import latest_state
    from tpa_index.schema import read_reviews

    group_of = {lab["name"]: lab["group"] for lab in labels["labels"]}
    out: dict[str, dict] = {}
    human = human_labels(labels, pipeline, root)
    for h in human[~human["skipped"]].to_dict("records"):
        out[h["segment_id"]] = {"activity": h["activities"], "category": h["categories"], "actor": h["actors"],
                                "indicator": h["indicators"]}
    decisions: dict[str, list[dict]] = {}
    for (mid, _), r in latest_state(read_reviews(root), mode="review").items():
        if r["target_type"] == "moment":
            decisions.setdefault(mid, []).append(r)
    for seg in view.to_dict("records"):
        if seg["segment_id"] in out or seg["moment_id"] not in decisions:
            continue
        sets = {g: set(seg["model_cleared"].get(g, set())) for g in (*GROUPS, "indicator")}
        touched = set()
        for r in decisions[seg["moment_id"]]:
            g = group_of.get(r["label"])
            if g not in sets:
                continue
            touched.add(g)
            if r["action"] == "accept":
                sets[g].add(r["label"])
            elif r["action"] == "reject":
                sets[g].discard(r["label"])
            elif r["action"] in ("edit", "add") and r["new_label"]:
                if r["action"] == "edit":
                    sets[g].discard(r["label"])
                sets[group_of.get(r["new_label"], g)].add(r["new_label"])
        if touched:
            out[seg["segment_id"]] = {g: sets[g] for g in touched}
    return out
