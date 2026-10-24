import copy

import pytest

from tests.fakes import FakeEmbedder
from tpa_index.config import model_labels
from tpa_index.schema import validate
from tpa_index.score import predict_video

VISUAL = {"category", "actor", "activity"}


def run(video, pipeline, labels, fake=None, **kw):
    return predict_video(video, pipeline, labels, fake or FakeEmbedder(), "predict-test", **kw)


def model_rows(out, model="fake"):
    return [p for p in out["predictions"] if p["model"] == model]


def test_one_score_per_segment_and_visual_label(clip_videos, pipeline, labels):
    out = run(clip_videos["t01"], pipeline, labels)
    visual = {lab["name"] for lab in model_labels(labels, VISUAL)}
    rows = model_rows(out)
    assert len(rows) == len(out["segments"]) * len(visual)
    assert {r["label"] for r in rows} == visual
    assert {r["label_group"] for r in rows} <= VISUAL  # no audio/expression/indicator/status model rows
    assert all(0 <= r["score"] <= 1 and r["run_id"] == "predict-test" for r in rows)
    assert validate("predictions", out["predictions"]).num_rows == len(out["predictions"])
    assert validate("segments", out["segments"]).num_rows == len(out["segments"])
    assert validate("videos", out["videos"]).num_rows == 1


def test_deterministic(clip_videos, pipeline, labels):
    assert run(clip_videos["t01"], pipeline, labels)["predictions"] == \
        run(clip_videos["t01"], pipeline, labels)["predictions"]


def test_frames_arrive_whole_and_square(clip_videos, pipeline, labels):
    fake = FakeEmbedder()
    run(clip_videos["t02"], pipeline, labels, fake)  # portrait 90x160
    assert fake.seen_shapes and set(fake.seen_shapes) == {(160, 160, 3)}


@pytest.mark.parametrize("batch", [1, 64])
def test_batch_size_does_not_change_scores(clip_videos, pipeline, labels, batch):
    p = copy.deepcopy(pipeline)
    p["models"]["batch_windows"]["cpu"] = batch
    base = run(clip_videos["t01"], pipeline, labels)["predictions"]
    got = run(clip_videos["t01"], p, labels)["predictions"]
    assert [r["label"] for r in got] == [r["label"] for r in base]
    assert all(abs(a["score"] - b["score"]) < 1e-6 for a, b in zip(got, base))


def _steer_activities(fake, labels, names):
    """Point every prompt of `names` at the fake image direction, so those labels match every window."""
    for lab in labels["labels"]:
        if lab["name"] in names:
            for prompt in lab["prompts"]:
                fake.steer(prompt, fake.base)


def _rules(out):
    return {p["segment_id"]: p["score"] for p in out["predictions"] if p["model"] == "rules"}


def test_clear_activity_is_primary_and_certain(clip_videos, pipeline, labels):
    fake = FakeEmbedder()
    _steer_activities(fake, labels, {"singing"})
    out = run(clip_videos["t01"], pipeline, labels, fake)
    assert set(_rules(out).values()) == {0.0}
    assert out["moments"] and {m["primary_label"] for m in out["moments"]} == {"singing"}
    assert not any(m["needs_review"] for m in out["moments"])
    assert {m["video_id"] for m in out["moments"]} <= {v["video_id"] for v in out["videos"]}
    assert validate("moments", out["moments"]).num_rows == len(out["moments"])


def test_two_equal_activities_are_uncertain_and_need_review(clip_videos, pipeline, labels):
    fake = FakeEmbedder()
    _steer_activities(fake, labels, {"singing", "dancing"})
    out = run(clip_videos["t01"], pipeline, labels, fake)
    assert set(_rules(out).values()) == {1.0}
    assert out["moments"] and all(m["needs_review"] for m in out["moments"])


def test_thresholds_above_every_score_give_no_moments(clip_videos, pipeline, labels):
    fake = FakeEmbedder()
    _steer_activities(fake, labels, {"singing"})
    p = copy.deepcopy(pipeline)
    p["label_thresholds"] = {"fake": {lab["name"]: 1.0 for lab in model_labels(labels, VISUAL)}}
    out = run(clip_videos["t01"], p, labels, fake)
    assert set(_rules(out).values()) == {1.0} and out["moments"] == []


def test_clip_kind_moments_follow_its_window(clip_videos, pipeline, labels):
    fake = FakeEmbedder()
    _steer_activities(fake, labels, {"acting"})
    out = run(clip_videos["t02"], pipeline, labels, fake)  # kind clip: 4 s windows, 3 s stride
    starts = sorted(s["start_s"] for s in out["segments"])
    assert starts[:3] == [0, 3, 6]
    (m,) = out["moments"]
    assert (m["start_s"], m["end_s"]) == (0, pytest.approx(out["segments"][-1]["end_s"]))


def _steer_idle(fake, pipeline):
    for prompt in pipeline["idle"]["prompts"]:
        fake.steer(prompt, fake.base)


def test_static_clip_with_idle_prompt_is_idle(clip_videos, pipeline, labels):
    fake = FakeEmbedder()
    _steer_idle(fake, pipeline)
    _steer_activities(fake, labels, {"singing"})
    out = run(clip_videos["t04"], pipeline, labels, fake)
    assert out["segments"] and all(s["is_idle"] for s in out["segments"])
    assert out["moments"] == []  # idle never starts a moment


def test_changing_clip_is_not_idle_even_if_prompt_matches(clip_videos, pipeline, labels):
    fake = FakeEmbedder()
    _steer_idle(fake, pipeline)
    out = run(clip_videos["t01"], pipeline, labels, fake)  # colour changes every second
    assert not any(s["is_idle"] for s in out["segments"])


def test_idle_prompt_alone_without_match_is_not_idle(clip_videos, pipeline, labels):
    out = run(clip_videos["t04"], pipeline, labels)  # static, but idle prompts not steered
    assert not any(s["is_idle"] for s in out["segments"])


def test_single_frame_windows_are_never_idle(clip_videos, pipeline, labels):
    fake = FakeEmbedder()
    _steer_idle(fake, pipeline)
    p = copy.deepcopy(pipeline)
    p["overrides"]["by_video"] = {"t04": {"frames_per_window": 1}}
    out = run(clip_videos["t04"], p, labels, fake)
    assert out["segments"] and not any(s["is_idle"] for s in out["segments"])


def test_motion_model_adds_rows_under_its_own_name(clip_videos, pipeline, labels):
    from tests.fakes import FakeMotion

    base = run(clip_videos["t01"], pipeline, labels)
    out = run(clip_videos["t01"], pipeline, labels, motion=FakeMotion(["dancing", "acting"]))
    motion_rows = [p for p in out["predictions"] if p["model"] == "fake-motion"]
    assert {r["label"] for r in motion_rows} == {"dancing", "acting"}
    assert len(motion_rows) == 2 * len(out["segments"])
    assert {r["segment_id"] for r in motion_rows} == {s["segment_id"] for s in out["segments"]}
    assert model_rows(out) == model_rows(base)  # main-model rows unchanged
    assert out["moments"] == base["moments"]  # moments use the main model only
    assert validate("predictions", out["predictions"]).num_rows == len(out["predictions"])
