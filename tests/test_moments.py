import copy

from tpa_index.moments import merge_moments
from tpa_index.schema import validate

PIPE = {"window_s": 8, "label_threshold": 0.5, "label_thresholds": {},
        "moments": {"primary_group": "activity", "secondary_groups": ["audio"], "max_gap_windows": 1,
                    "min_windows": 1, "needs_review_margin": 0.2}}
GROUPS = {"singing": "activity", "dancing": "activity", "music": "audio", "participation": "indicator"}


def make(spec, length=8, idle=(), flagged=()):
    """spec: list of (start_s, {label: score}); non-overlapping windows of `length` seconds."""
    segs, preds = [], []
    for start, sc in spec:
        sid = f"t01_{start * 1000}"
        segs.append({"segment_id": sid, "video_id": "t01", "start_s": start, "end_s": start + length,
                     "is_idle": start in idle})
        for label, group in GROUPS.items():
            preds.append({"segment_id": sid, "label": label, "label_group": group, "score": sc.get(label, 0.1),
                          "model": "m1"})
        preds.append({"segment_id": sid, "label": "uncertain_mixed", "label_group": "status",
                      "score": 1.0 if start in flagged else 0.0, "model": "rules"})
    return segs, preds


def test_adjacent_segments_merge():
    m = merge_moments(*make([(0, {"singing": 0.9}), (8, {"singing": 0.9}), (16, {"singing": 0.9})]), PIPE, "r1")
    assert len(m) == 1 and (m[0]["start_s"], m[0]["end_s"]) == (0, 24)


def test_one_window_gap_merges_two_windows_split():
    assert len(merge_moments(*make([(0, {"singing": 0.9}), (16, {"singing": 0.9})]), PIPE, "r1")) == 1
    assert len(merge_moments(*make([(0, {"singing": 0.9}), (24, {"singing": 0.9})]), PIPE, "r1")) == 2


def test_video_window_changes_gap_and_min_length():
    segs, preds = make([(0, {"singing": 0.9}), (12, {"singing": 0.9})], length=4)
    assert len(merge_moments(segs, preds, PIPE, "r1", window_s=8)) == 1  # gap 8 <= 1 x 8 s
    assert len(merge_moments(segs, preds, PIPE, "r1", window_s=4)) == 2  # gap 8 > 1 x 4 s
    short = make([(40, {"singing": 0.9})], length=5)
    assert merge_moments(*short, PIPE, "r1") == []  # 5 s < one 8 s window
    assert len(merge_moments(*short, PIPE, "r1", window_s=4)) == 1  # 5 s >= one 4 s window


def test_idle_segment_breaks_moment():
    spec = [(0, {"singing": 0.9}), (8, {"singing": 0.9}), (16, {"singing": 0.9})]
    m = merge_moments(*make(spec, idle={8}), PIPE, "r1")
    assert [(x["start_s"], x["end_s"]) for x in m] == [(0, 8), (16, 24)]
    assert merge_moments(*make(spec, idle={0, 8, 16}), PIPE, "r1") == []


def test_confidence_near_threshold_needs_review():
    # threshold 0.5, margin 0.2 -> review below 0.6
    p = {**PIPE, "label_thresholds": {"m1": {"singing": 0.5}}}
    (near,) = merge_moments(*make([(0, {"singing": 0.55}), (8, {"singing": 0.58})]), p, "r1", model="m1")
    (clear,) = merge_moments(*make([(0, {"singing": 0.65}), (8, {"singing": 0.70})]), p, "r1", model="m1")
    assert near["needs_review"] and not clear["needs_review"]
    (edge,) = merge_moments(*make([(0, {"singing": 0.6})]), p, "r1", model="m1")
    assert not edge["needs_review"]  # exactly threshold * 1.2 is not below it


def test_review_margin_zero_and_per_label_threshold():
    p = {**PIPE, "label_thresholds": {"m1": {"singing": 0.3}},
         "moments": {**PIPE["moments"], "needs_review_margin": 0.0}}
    (m,) = merge_moments(*make([(0, {"singing": 0.31})]), p, "r1", model="m1")
    assert not m["needs_review"]  # margin 0: a moment that clears its threshold is not flagged for confidence
    p["moments"]["needs_review_margin"] = 0.2
    (m,) = merge_moments(*make([(0, {"singing": 0.31})]), p, "r1", model="m1")
    assert m["needs_review"]  # 0.31 < 0.3 * 1.2 = 0.36, using the label's own threshold


def test_uncertain_mixed_segment_sets_needs_review():
    spec = [(0, {"singing": 0.9}), (8, {"singing": 0.9})]  # confident: only the flagged segment can trigger review
    (clean,) = merge_moments(*make(spec), PIPE, "r1")
    (flag,) = merge_moments(*make(spec, flagged={8}), PIPE, "r1")
    assert not clean["needs_review"] and flag["needs_review"]


def test_per_label_threshold_and_model_filter():
    p = {**PIPE, "label_thresholds": {"m1": {"singing": 0.95}}}
    segs, preds = make([(0, {"singing": 0.9})])
    assert merge_moments(segs, preds, p, "r1", model="m1") == []  # 0.9 < 0.95 for m1
    assert len(merge_moments(segs, preds, p, "r1", model=None)) == 1  # global 0.5 applies without a model
    other = [dict(x, model="m2", score=0.99) for x in preds if x["model"] == "m1" and x["label"] == "dancing"]
    assert merge_moments(segs, preds + other, p, "r1", model="m1") == []  # m2 rows ignored for m1


def test_secondary_and_indicators_listed_and_valid():
    m = merge_moments(*make([(0, {"singing": 0.9, "music": 0.8, "participation": 0.7})]), PIPE, "r1")
    assert m[0]["secondary_labels"] == ["music"] and m[0]["indicators"] == ["participation"]
    assert validate("moments", m).num_rows == 1
