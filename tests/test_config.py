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
