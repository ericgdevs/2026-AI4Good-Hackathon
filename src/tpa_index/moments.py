"""Merge per-segment label scores into reviewable moments (rules in pipeline.yaml: moments)."""
from collections import defaultdict
from statistics import mean

from tpa_index.config import threshold


def merge_moments(segments, predictions, pipeline: dict, run_id: str, window_s=None, model=None) -> list[dict]:
    """Moments for the given segments. window_s: the video's effective window; model: whose scores to use
    (default: every non-rules row). Idle segments never start or extend a moment."""
    rows = lambda t: t.to_pylist() if hasattr(t, "to_pylist") else list(t)  # noqa: E731
    segments, predictions = rows(segments), rows(predictions)
    cfg = pipeline["moments"]
    review_factor = 1 + cfg["needs_review_margin"]
    window = window_s or pipeline["window_s"]
    max_gap, min_len = cfg["max_gap_windows"] * window, cfg["min_windows"] * window

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    group_of: dict[str, str] = {}
    flagged: set[str] = set()  # segments with an uncertain_mixed rules row = 1
    for p in predictions:
        if p["model"] == "rules":
            if p["label"] == "uncertain_mixed" and p["score"] >= 0.5:
                flagged.add(p["segment_id"])
            continue
        if model is not None and p["model"] != model:
            continue
        scores[p["segment_id"]][p["label"]] = p["score"]
        group_of[p["label"]] = p["label_group"]
    thr = {label: threshold(pipeline, model, label) for label in group_of}
    in_group = lambda g: [label for label, grp in group_of.items() if grp == g]  # noqa: E731
    primary_labels = in_group(cfg["primary_group"])
    secondary = [label for g in cfg["secondary_groups"] for label in in_group(g)]
    indicator_labels = in_group("indicator")

    def build(video_id, label, segs):
        start, end = segs[0]["start_s"], max(s["end_s"] for s in segs)
        if end - start < min_len:
            return None
        avg = lambda lab: mean(scores[s["segment_id"]].get(lab, 0.0) for s in segs)  # noqa: E731

        def above(labs):
            ranked = sorted(((avg(x), x) for x in labs), reverse=True)
            return [lab for score, lab in ranked if score >= thr[lab]]

        confidence = avg(label)
        return {
            "moment_id": f"{video_id}_{round(start * 1000)}_{label}",
            "video_id": video_id, "start_s": start, "end_s": end, "primary_label": label,
            "secondary_labels": above(secondary), "indicators": above(indicator_labels),
            "confidence": confidence,
            # close to the bar (within the margin above it), or any segment too close to call
            "needs_review": confidence < thr[label] * review_factor or any(s["segment_id"] in flagged for s in segs),
            "run_id": run_id,
        }

    by_video = defaultdict(list)
    for s in segments:
        by_video[s["video_id"]].append(s)
    moments = []
    for video_id, segs in by_video.items():
        current_label, current = None, []
        for seg in sorted(segs, key=lambda s: s["start_s"]):
            if seg.get("is_idle") is True:  # idle closes the current moment
                if current:
                    moments.append(build(video_id, current_label, current))
                current_label, current = None, []
                continue
            seg_scores = scores.get(seg["segment_id"], {})
            cleared = [lab for lab in primary_labels if seg_scores.get(lab, 0.0) >= thr[lab]]
            top = max(cleared, key=lambda lab: seg_scores[lab], default=None)
            if top is None:
                continue
            gap = seg["start_s"] - max(s["end_s"] for s in current) if current else None
            if current and top == current_label and gap <= max_gap:
                current.append(seg)
                continue
            if current:
                moments.append(build(video_id, current_label, current))
            current_label, current = top, [seg]
        if current:
            moments.append(build(video_id, current_label, current))
    return [m for m in moments if m]
