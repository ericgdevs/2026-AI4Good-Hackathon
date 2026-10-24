"""005 per-segment view: time slices, model calls, flags, human and verified labels, input checks (mock world)."""
import pandas as pd
import pytest
import yaml

from tests.insights_fixtures import AST, build_world
from tpa_index import config
from tpa_index.config import ConfigError
from tpa_index.view import check_inputs, human_labels, segment_view, verified_labels


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


def _view(world):
    return segment_view(world["pipeline"], world["labels"])


def test_slices_partition_duration(world):
    """SC-003 (part 1): non-overlapping slices sum to each video's duration."""
    view = _view(world)
    assert (view["slice_s"] > 0).all()
    totals = view.groupby("video_id")["slice_s"].sum()
    assert totals.round(6).to_dict() == {v: 120.0 for v in ("d01", "e01", "e02", "x01")}
    assert view[view["segment_id"] == "e01_0"]["slice_s"].item() == 6.0
    assert view[view["segment_id"] == "e01_114000"]["slice_s"].item() == 6.0


def test_no_moment_is_uncertain_and_needs_review_inherited(world):
    """SC-010: a segment in no moment is "model made no call" -> uncertain_mixed; needs_review comes from the
    moment that contains the segment."""
    view = _view(world).set_index("segment_id")
    for sid in world["idle"] + world["instruction"]:
        assert view.loc[sid, "no_call"] and view.loc[sid, "model_primary"] == "uncertain_mixed"
        assert view.loc[sid, "model_strength"] == 0.0
    dance = "e01_30000"  # e01 dance at 0.55: its moment's confidence is below threshold * 1.2
    assert view.loc[dance, "model_primary"] == "dance" and view.loc[dance, "needs_review"]
    assert not view.loc["e01_6000", "needs_review"] and not view.loc["e01_6000", "no_call"]


def test_model_calls_and_signals(world):
    view = _view(world).set_index("segment_id")
    row = view.loc["e02_0"]
    assert row["model_primary"] == "singing" and "performance" in row["model_cleared"]["category"]
    assert row["person_count"] == 1 and row["faces_detected"] == 2 and row["expr_top"] == "positive"
    assert pd.isna(view.loc["e01_66000", "expr_top"])  # idle: no faces
    assert row["audio_ok"]["singing_voice"] is True and row["audio_ok"]["music"] is False
    assert "split" in view.columns and "filename" not in view.columns


def test_audio_uses_fixed_threshold_not_calibrated(world):
    """Audio conditions use insights.audio_threshold (0.5) even if the local file holds a 0.95 audio threshold."""
    data = yaml.safe_load(config.THRESHOLDS_LOCAL.read_text())
    data["label_thresholds"][AST] = {"singing_voice": 0.95, "music": 0.95, "speech": 0.95}
    config.THRESHOLDS_LOCAL.write_text(yaml.safe_dump(data))
    pipeline = config.load_pipeline()
    view = segment_view(pipeline, world["labels"]).set_index("segment_id")
    assert view.loc["e02_0", "audio_ok"]["singing_voice"] is True  # 0.8 >= 0.5


def test_check_inputs_lists_every_missing_item(tmp_path, monkeypatch):
    world = build_world(tmp_path, monkeypatch, overrides={"skip_stages": [("audio", "e02")]})
    data = yaml.safe_load(config.THRESHOLDS_LOCAL.read_text())
    del data["label_thresholds"]["mock-siglip"]["dance"]
    config.THRESHOLDS_LOCAL.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError) as err:
        check_inputs("evaluate", config.load_pipeline(), world["labels"])
    msg = str(err.value)
    assert "official audio runs missing for: e02" in msg
    assert "no calibrated threshold for dance (model mock-siglip)" in msg


def test_check_inputs_passes_on_complete_world(world):
    check_inputs("evaluate", world["pipeline"], world["labels"])
    check_inputs("insights", world["pipeline"], world["labels"])


def test_human_labels_latest_state_and_special_cases(world):
    human = human_labels(world["labels"], world["pipeline"]).set_index("segment_id")
    first = human.loc["e01_0"]
    assert first["primary"] == "singing" and "dance" not in first["activities"]  # added, then rejected
    assert "solo_on_stage" in first["indicators"]
    assert human.loc[world["two_categories"], "categories"] == {"performance", "participation"}
    assert human.loc[world["no_category"], "categories"] == set()
    assert human.loc[world["skip_segment"], "skipped"]
    assert human.loc[world["idle"][0], "primary"] == "uncertain_mixed"
    assert {human.loc[s, "queue"] for s in world["topup"]} == {"topup"}
    assert human.loc["e01_6000", "queue"] == "main"


def test_verified_labels_blind_first(world):
    view = _view(world)
    verified = verified_labels(view, world["labels"], world["pipeline"])
    assert verified["e01_78000"]["actor"] == {"actor_staff_facilitation"}  # blind: instruction stretch
    assert world["skip_segment"] not in verified
    assert "e02_60000" not in verified  # model-only staff stretch, not labeled
