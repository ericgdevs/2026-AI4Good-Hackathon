import copy
import re

import pytest
import yaml

from tpa_index.config import (ConfigError, EXAMPLE_VIDEOS, _check_labels, _check_pipeline, _check_videos,
                              load_labels, load_pipeline, load_videos)

BANNED = ["anxious", "depressed", "adhd", "autism", "autistic", "disabled", "disability", "trauma", "mental",
          "diagnosis", "race", "ethnicity", "gender", "age", "religion", "iq", "risk"]


def test_shipped_configs_load():
    assert load_labels()["guideline_version"]
    assert load_pipeline()["window_s"] == 8
    assert [v["video_id"] for v in load_videos(EXAMPLE_VIDEOS)] == ["mock01", "mock02", "mock03", "mock04"]


def test_example_videos_are_mock_only():
    assert all(v["video_id"].startswith("mock") and v["filename"].startswith("mock/")
               for v in load_videos(EXAMPLE_VIDEOS))


def _labels_with(labels, **changes):
    bad = copy.deepcopy(labels)
    bad["labels"][0].update(changes)
    return bad


@pytest.mark.parametrize("changes,match", [
    ({"prompts": ["a", "b"]}, "3-5 prompt"),
    ({"prompts": list("abcdef")}, "3-5 prompt"),
    ({"group": "mood"}, "unknown group"),
    ({"color": "blue"}, "color"),
])
def test_bad_label_named(labels, changes, match):
    with pytest.raises(ConfigError, match=rf"label {labels['labels'][0]['name']}: .*{match}"):
        _check_labels(_labels_with(labels, **changes))


def test_missing_guideline_version(labels):
    bad = copy.deepcopy(labels)
    del bad["guideline_version"]
    with pytest.raises(ConfigError, match="guideline_version"):
        _check_labels(bad)


def test_expression_set_is_fixed(labels):
    bad = copy.deepcopy(labels)
    bad["labels"].append({"name": "anxious_look", "group": "expression", "prompts": ["x", "y", "z"],
                          "color": "#000000"})
    with pytest.raises(ConfigError, match="expression"):
        _check_labels(bad)


def test_overlap_must_be_below_window(pipeline):
    with pytest.raises(ConfigError, match="overlap_s"):
        _check_pipeline({**pipeline, "overlap_s": 8})


def test_unknown_override_key(pipeline):
    bad = copy.deepcopy(pipeline)
    bad["overrides"]["by_kind"]["clip"]["label_threshold"] = 0.2
    with pytest.raises(ConfigError, match="by_kind.clip"):
        _check_pipeline(bad)


def test_override_combination_checked(pipeline):
    bad = copy.deepcopy(pipeline)
    bad["overrides"]["by_video"] = {"v09": {"overlap_s": 5}}  # fine at 8 s, broken with clip's 4 s window
    with pytest.raises(ConfigError, match="by_video.v09"):
        _check_pipeline(bad)


def test_bad_videos_named(pipeline):
    good = load_videos(EXAMPLE_VIDEOS)
    phases = pipeline["video_phases"]
    for change, match in (({"phase": "finale"}, "phase"), ({"filename": "../x.mp4"}, "filename"),
                          ({"video_id": "Mock 1"}, "video_id")):
        bad = copy.deepcopy(good)
        bad[0].update(change)
        with pytest.raises(ConfigError, match=match):
            _check_videos(bad, phases)


def test_bad_yaml_file_raises_config_error(tmp_path):
    (tmp_path / "p.yaml").write_text("window_s: [unclosed")
    with pytest.raises(ConfigError):
        load_pipeline(tmp_path / "p.yaml")


def test_vocabulary_has_no_banned_terms(labels):
    words = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.I)
    text = yaml.safe_dump(labels["labels"]).replace("_", " ")
    assert not words.findall(text)


def test_shipped_labels_are_guideline_0_2_0(labels):
    from tpa_index.config import model_labels, rules_labels

    assert labels["guideline_version"] == "0.2.0"
    assert {lab["name"] for lab in rules_labels(labels)} == {"uncertain_mixed", "needs_review"}
    assert not {"uncertain", "mixed"} & {lab["name"] for lab in labels["labels"]}
    assert all(lab["group"] != "status" for lab in model_labels(labels))


def test_rules_label_with_prompts_fails(labels):
    bad = copy.deepcopy(labels)
    rule = next(lab for lab in bad["labels"] if lab["name"] == "uncertain_mixed")
    rule["prompts"] = ["a", "b", "c"]
    with pytest.raises(ConfigError, match="label uncertain_mixed: rules labels must not have prompts"):
        _check_labels(bad)


def test_model_label_without_prompts_fails(labels):
    bad = copy.deepcopy(labels)
    del bad["labels"][0]["prompts"]
    with pytest.raises(ConfigError, match="3-5 prompt"):
        _check_labels(bad)


def test_unknown_source_fails(labels):
    with pytest.raises(ConfigError, match="source"):
        _check_labels(_labels_with(labels, source="staff"))


@pytest.mark.parametrize("change,match", [
    (lambda p: p.update(visual_groups=["category", "mood"]), "visual_groups"),
    (lambda p: p["models"]["batch_windows"].update(mps=0), "batch_windows.mps"),
    (lambda p: p.update(label_thresholds={"m": {"singing": 1.5}}), "within 0-1"),
    (lambda p: p.update(uncertain_margin=1.0), "uncertain_margin"),
    (lambda p: p["idle"].update(prompts=[]), "idle"),
])
def test_bad_pipeline_keys(pipeline, change, match):
    bad = copy.deepcopy(pipeline)
    change(bad)
    with pytest.raises(ConfigError, match=match):
        _check_pipeline(bad)


@pytest.mark.parametrize("change,match", [
    (lambda p: p["models"]["motion"].update(labels=["music"]), "music is not an activity label"),
    (lambda p: p.update(label_thresholds={"m": {"nope": 0.2}}), "nope: not a model-scored label"),
    (lambda p: p.update(label_thresholds={"m": {"uncertain_mixed": 0.2}}), "uncertain_mixed: not a model-scored"),
    (lambda p: p.update(visual_groups=["status"]), "status has no model-scored labels"),
])
def test_pipeline_checks_against_labels(pipeline, labels, change, match):
    from tpa_index.config import _check_pipeline_labels

    bad = copy.deepcopy(pipeline)
    change(bad)
    with pytest.raises(ConfigError, match=match):
        _check_pipeline_labels(_check_pipeline(bad), labels)


def test_threshold_fallback_and_overrides(tmp_path, monkeypatch):
    from tpa_index import config
    from tpa_index.config import threshold

    p = load_pipeline()
    assert threshold(p, "m1", "singing") == p["label_threshold"]
    local = tmp_path / "thresholds.local.yaml"
    local.write_text("label_thresholds:\n  m1: {singing: 0.12}\n")
    monkeypatch.setattr(config, "THRESHOLDS_LOCAL", local)
    p = load_pipeline()
    assert threshold(p, "m1", "singing") == 0.12
    assert threshold(p, "m1", "dancing") == p["label_threshold"]
    assert threshold(p, "m2", "singing") == p["label_threshold"]  # another model's threshold never applies


def test_local_thresholds_win_over_pipeline(tmp_path, monkeypatch):
    from tpa_index import config
    from tpa_index.config import threshold

    pipe = tmp_path / "pipeline.yaml"
    base = yaml.safe_load((config.CONFIG_DIR / "pipeline.yaml").read_text())
    base["label_thresholds"] = {"m1": {"singing": 0.3, "dancing": 0.4}}
    pipe.write_text(yaml.safe_dump(base))
    local = tmp_path / "thresholds.local.yaml"
    local.write_text("label_thresholds:\n  m1: {singing: 0.12}\n")
    monkeypatch.setattr(config, "THRESHOLDS_LOCAL", local)
    p = load_pipeline(pipe)
    assert (threshold(p, "m1", "singing"), threshold(p, "m1", "dancing")) == (0.12, 0.4)
