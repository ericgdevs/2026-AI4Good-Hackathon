from tpa_index.moments import merge_moments
from tpa_index.schema import validate

PIPE = {"window_s": 8, "label_threshold": 0.5,
        "moments": {"primary_group": "activity", "secondary_groups": ["audio"], "max_gap_windows": 1,
                    "min_windows": 1, "needs_review_below": 0.6, "conflict_margin": 0.1}}


def make(spec, end=None):
    """spec: list of (start_s, {label: score}); non-overlapping 8 s windows."""
    segs, preds = [], []
    groups = {"singing": "activity", "dancing": "activity", "music": "audio", "participation": "indicator"}
    for start, sc in spec:
        sid = f"t01_{start * 1000}"
        segs.append({"segment_id": sid, "video_id": "t01", "start_s": start, "end_s": end or start + 8})
        for label, group in groups.items():
            preds.append({"segment_id": sid, "label": label, "label_group": group, "score": sc.get(label, 0.1)})
    return segs, preds


def test_adjacent_segments_merge():
    m = merge_moments(*make([(0, {"singing": 0.9}), (8, {"singing": 0.9}), (16, {"singing": 0.9})]), PIPE, "r1")
    assert len(m) == 1 and (m[0]["start_s"], m[0]["end_s"]) == (0, 24)


def test_one_window_gap_merges_two_windows_split():
    assert len(merge_moments(*make([(0, {"singing": 0.9}), (16, {"singing": 0.9})]), PIPE, "r1")) == 1
    assert len(merge_moments(*make([(0, {"singing": 0.9}), (24, {"singing": 0.9})]), PIPE, "r1")) == 2


def test_moment_shorter_than_one_window_dropped():
    segs, preds = make([(40, {"singing": 0.9})], end=45)
    assert merge_moments(segs, preds, PIPE, "r1") == []


def test_low_confidence_needs_review():
    (m,) = merge_moments(*make([(0, {"singing": 0.55})]), PIPE, "r1")
    assert m["needs_review"]


def test_close_runner_up_needs_review():
    (m,) = merge_moments(*make([(0, {"singing": 0.9, "dancing": 0.85})]), PIPE, "r1")
    assert m["needs_review"]
    (m,) = merge_moments(*make([(0, {"singing": 0.9, "dancing": 0.2})]), PIPE, "r1")
    assert not m["needs_review"]


def test_secondary_and_indicators_listed_and_valid():
    m = merge_moments(*make([(0, {"singing": 0.9, "music": 0.8, "participation": 0.7})]), PIPE, "r1")
    assert m[0]["secondary_labels"] == ["music"] and m[0]["indicators"] == ["participation"]
    assert validate("moments", m).num_rows == 1
