"""Merge per-segment label scores into reviewable moments (rules in pipeline.yaml: moments)."""
from collections import defaultdict
from statistics import mean


def merge_moments(segments, predictions, pipeline: dict, run_id: str) -> list[dict]:
    rows = lambda t: t.to_pylist() if hasattr(t, "to_pylist") else list(t)  # noqa: E731
    segments, predictions = rows(segments), rows(predictions)
    cfg, thr = pipeline["moments"], pipeline["label_threshold"]
    window = pipeline["window_s"]  # shortcut: ignores per-video overrides; pass effective settings once moments run per video
    max_gap, min_len = cfg["max_gap_windows"] * window, cfg["min_windows"] * window

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    group_of: dict[str, str] = {}
    for p in predictions:
        scores[p["segment_id"]][p["label"]] = p["score"]
        group_of[p["label"]] = p["label_group"]
    in_group = lambda g: [label for label, grp in group_of.items() if grp == g]  # noqa: E731
    primary_labels = in_group(cfg["primary_group"])
    secondary = [label for g in cfg["secondary_groups"] for label in in_group(g)]
    indicator_labels = in_group("indicator")

    def build(video_id, label, segs):
        start, end = segs[0]["start_s"], max(s["end_s"] for s in segs)
        if end - start < min_len:
            return None
        avg = lambda lab: mean(scores[s["segment_id"]].get(lab, 0.0) for s in segs)  # noqa: E731
        confidence = avg(label)
        runner_up = max((avg(lab) for lab in primary_labels if lab != label), default=0.0)
        def above(labs):
            ranked = sorted(((avg(x), x) for x in labs), reverse=True)
            return [lab for score, lab in ranked if score >= thr]

        return {
            "moment_id": f"{video_id}_{round(start * 1000)}_{label}",
            "video_id": video_id, "start_s": start, "end_s": end, "primary_label": label,
            "secondary_labels": above(secondary), "indicators": above(indicator_labels),
            "confidence": confidence,
            "needs_review": confidence < cfg["needs_review_below"] or confidence - runner_up < cfg["conflict_margin"],
            "run_id": run_id,
        }

    by_video = defaultdict(list)
    for s in segments:
        by_video[s["video_id"]].append(s)
    moments = []
    for video_id, segs in by_video.items():
        current_label, current = None, []
        for seg in sorted(segs, key=lambda s: s["start_s"]):
            seg_scores = scores.get(seg["segment_id"], {})
            top = max(primary_labels, key=lambda lab: seg_scores.get(lab, 0.0), default=None)
            if top is None or seg_scores.get(top, 0.0) < thr:
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
