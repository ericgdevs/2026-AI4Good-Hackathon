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


def test_shipped_labels_are_guideline_0_6_0(labels):
    from tpa_index.config import model_labels, rules_labels

    assert labels["guideline_version"] == "0.6.0"
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
        _check_labels(_labels_with(labels, source="robot"))


@pytest.mark.parametrize("change,match", [
    (lambda p: p.update(visual_groups=["category", "mood"]), "visual_groups"),
    (lambda p: p["models"]["batch_windows"].update(mps=0), "batch_windows.mps"),
    (lambda p: p.update(label_thresholds={"m": {"singing": 1.5}}), "within 0-1"),
    (lambda p: p["moments"].update(close_ratio=1.5), "close_ratio"),
    (lambda p: p["moments"].pop("close_ratio"), "close_ratio"),
    (lambda p: p["idle"].update(prompts=[]), "idle"),
    (lambda p: p["moments"].update(needs_review_margin=1.5), "needs_review_margin"),
    (lambda p: p["moments"].update(needs_review_margin=-0.1), "needs_review_margin"),
    (lambda p: p["moments"].pop("needs_review_margin"), "needs_review_margin"),
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
    (lambda p: p.update(visual_groups=["status"]), "group status \\(visual or null\\) has no model-scored labels"),
    (lambda p: p.pop("app"), "pipeline: app section missing"),
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
    assert threshold(p, "m1", "dance") == p["label_threshold"]
    assert threshold(p, "m2", "singing") == p["label_threshold"]  # another model's threshold never applies


def test_local_thresholds_win_over_pipeline(tmp_path, monkeypatch):
    from tpa_index import config
    from tpa_index.config import threshold

    pipe = tmp_path / "pipeline.yaml"
    base = yaml.safe_load((config.CONFIG_DIR / "pipeline.yaml").read_text())
    base["label_thresholds"] = {"m1": {"singing": 0.3, "dance": 0.4}}
    pipe.write_text(yaml.safe_dump(base))
    local = tmp_path / "thresholds.local.yaml"
    local.write_text("label_thresholds:\n  m1: {singing: 0.12}\n")
    monkeypatch.setattr(config, "THRESHOLDS_LOCAL", local)
    p = load_pipeline(pipe)
    assert (threshold(p, "m1", "singing"), threshold(p, "m1", "dance")) == (0.12, 0.4)


@pytest.mark.parametrize("change,match", [
    (lambda p: p["calibrate"].update(top_pct=0), "top_pct"),
    (lambda p: p["calibrate"].update(top_pct=100), "top_pct"),
    (lambda p: p["idle"].update(calibrate_top_pct=150), "calibrate_top_pct"),
    (lambda p: p["calibrate"].update(min_threshold=1.2), "min_threshold"),
    (lambda p: p["calibrate"].update(min_segments=0), "min_segments"),
    (lambda p: p.pop("calibrate"), "calibrate section"),
])
def test_bad_calibrate_settings(pipeline, change, match):
    bad = copy.deepcopy(pipeline)
    change(bad)
    with pytest.raises(ConfigError, match=match):
        _check_pipeline(bad)


def test_approved_vocabulary(labels):
    from tpa_index.config import model_labels

    by_group = lambda g: [lab["name"] for lab in labels["labels"] if lab["group"] == g]  # noqa: E731
    assert by_group("category") == ["performance", "creative_practice", "participation", "instruction"]
    assert by_group("actor") == ["actor_participant", "actor_staff_facilitation"]
    assert len(by_group("activity")) == 13 and "audience_watching" in by_group("activity")
    assert all(lab.get("source") == "staff" and not lab.get("prompts")
               for lab in labels["labels"] if lab["group"] == "indicator")
    assert len(model_labels(labels, {"null"})) == 5
    assert all(len(lab["prompts"]) == 3 for lab in model_labels(labels, {"category", "actor", "activity"}))


def test_staff_label_with_prompts_fails(labels):
    bad = copy.deepcopy(labels)
    staff = next(lab for lab in bad["labels"] if lab.get("source") == "staff")
    staff["prompts"] = ["a", "b", "c"]
    with pytest.raises(ConfigError, match="staff labels must not have prompts"):
        _check_labels(bad)


@pytest.mark.parametrize("change,match", [
    (lambda p: p["calibrate"].update(null_group=None), "quoted group name"),
    (lambda p: p["calibrate"].update(null_group="activity"), "must not be a visual"),
    (lambda p: p["calibrate"].update(null_percentile=0), "null_percentile"),
])
def test_bad_null_settings(pipeline, change, match):
    bad = copy.deepcopy(pipeline)
    change(bad)
    with pytest.raises(ConfigError, match=match):
        _check_pipeline(bad)



@pytest.mark.parametrize("change,match", [
    (lambda a, p: a.update(overlap_size=a["sample_size"] + 1), "overlap_size"),
    (lambda a, p: a.update(blind_groups=a["blind_groups"] + ["nope"]), "unknown group"),
    (lambda a, p: a.update(blind_groups=[g for g in a["blind_groups"] if g != p["moments"]["primary_group"]]),
     "primary group"),
    (lambda a, p: a.update(blind_groups=a["blind_groups"] + [p["calibrate"]["null_group"]]), "null group"),
    (lambda a, p: a.update(uncertain_label="no_such_rule"), "uncertain_label"),
    (lambda a, p: a.update(skip_observation="no_such_type"), "skip_observation"),
    (lambda a, p: a.update(padding_s=31), "padding_s"),
    (lambda a, p: a.update(port=80), "port"),
])
def test_bad_app_settings(pipeline, labels, change, match):
    from tpa_index.config import _check_app

    bad = copy.deepcopy(pipeline)
    change(bad["app"], bad)
    with pytest.raises(ConfigError, match=match):
        _check_app(bad["app"], bad, labels)


def test_rules_only_group_cannot_be_blind_labeled(pipeline, labels):
    from tpa_index.config import _check_app

    rules_group = next(lab["group"] for lab in labels["labels"] if lab.get("source") == "rules")
    bad = copy.deepcopy(pipeline)
    bad["app"]["blind_groups"].append(rules_group)
    with pytest.raises(ConfigError, match="no model or staff labels"):
        _check_app(bad["app"], bad, labels)


@pytest.mark.parametrize("change,match", [
    (lambda o, names: o.append(dict(o[0])), "duplicate"),
    (lambda o, names: o.append({"name": names[0], "description": "x"}), "clashes with a label"),
    (lambda o, names: o.append({"name": "Bad-Name", "description": "x"}), "lowercase"),
    (lambda o, names: o.append({"name": "fine_name", "description": " "}), "description"),
])
def test_bad_observation_types(labels, change, match):
    bad = copy.deepcopy(labels)
    change(bad["observation_types"], [lab["name"] for lab in bad["labels"]])
    with pytest.raises(ConfigError, match=match):
        _check_labels(bad)


def test_observation_types_listed(labels, pipeline):
    from tpa_index.config import observation_types

    assert pipeline["app"]["skip_observation"] in observation_types(labels)
    assert len(observation_types(labels)) == len(set(observation_types(labels)))


# ---- 004 signals -------------------------------------------------------------------------------------------------
def test_singing_voice_is_an_audio_label(labels):
    by_name = {lab["name"]: lab for lab in labels["labels"]}
    assert by_name["singing_voice"]["group"] == "audio"
    assert by_name["singing"]["group"] == "activity"
    assert [lab["name"] for lab in labels["labels"] if lab["group"] == "audio"] == \
        ["music", "speech", "applause", "silence", "singing_voice"]


@pytest.mark.parametrize("change,match", [
    (lambda p: p["signals"]["audio"]["mapping"].update(nope=["Music"]), "mapping.nope: not an allowed label"),
    (lambda p: p["signals"]["audio"]["mapping"].pop("silence"), "audio label silence has no classes"),
    (lambda p: p["signals"]["expression"]["mapping"].update(confident=["Happiness"]), "confident: not an allowed"),
    (lambda p: p["signals"]["expression"]["mapping"].update(upset=["Sadness", "Neutral"]), "'Neutral' is mapped to more"),
    (lambda p: p["signals"]["audio"]["mapping"].update(music=[]), "music: must be a non-empty list"),
    (lambda p: p["models"]["people"]["hpc"].update(sha256="xyz"), "people.hpc.sha256"),
    (lambda p: p["models"]["people"]["mac"].update(file="../yolo11n.pt"), "people.mac.file: must be a bare"),
    (lambda p: p["models"]["expression"]["model"].update(file="a/b.onnx"), "expression.model.file"),
    (lambda p: p["signals"]["expression"].update(face_score=1.5), "face_score must be within"),
    (lambda p: p["signals"]["expression"].update(detect_max_side=100), "detect_max_side: must be an integer"),
    (lambda p: p["signals"]["people"].update(person_conf=0), "person_conf must be within"),
    (lambda p: p["signals"]["people"].update(min_box_height_frac=1), "min_box_height_frac must be within"),
    (lambda p: p["signals"]["people"].pop("min_box_height_frac"), "min_box_height_frac must be a number"),
    (lambda p: p["models"]["transcript"].update(hpc_device="tpu"), "hpc_device: must be cuda or cpu"),
    (lambda p: p["models"].pop("audio"), "models.audio"),
    (lambda p: p.pop("signals"), "signals section missing"),
])
def test_bad_signal_settings(pipeline, labels, change, match):
    from tpa_index.config import _check_pipeline_labels

    bad = copy.deepcopy(pipeline)
    change(bad)
    with pytest.raises(ConfigError, match=match):
        _check_pipeline_labels(_check_pipeline(bad), labels)


def test_pinned_weight(tmp_path, monkeypatch):
    import hashlib

    from tpa_index.config import pinned_weight, weights_dir

    monkeypatch.setenv("TPA_WEIGHTS_DIR", str(tmp_path))
    assert weights_dir() == tmp_path
    with pytest.raises(ConfigError, match="weights not found: w.pt"):
        pinned_weight({"file": "w.pt", "sha256": "0" * 64})
    (tmp_path / "w.pt").write_bytes(b"weights")
    with pytest.raises(ConfigError, match="weights not pinned: set sha256 for w.pt"):
        pinned_weight({"file": "w.pt", "sha256": None})
    with pytest.raises(ConfigError, match="checksum mismatch: w.pt"):
        pinned_weight({"file": "w.pt", "sha256": "0" * 64})
    good = hashlib.sha256(b"weights").hexdigest()
    assert pinned_weight({"file": "w.pt", "sha256": good}) == tmp_path / "w.pt"


def test_weights_dir_defaults_to_hf_home(tmp_path, monkeypatch):
    from tpa_index.config import weights_dir

    monkeypatch.delenv("TPA_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert weights_dir() == tmp_path / "tpa-weights"


def test_stage_config_hash_is_scoped(pipeline, labels):
    from tpa_index.config import stage_config_hash

    video = {"video_id": "v01", "kind": "live"}
    base = {s: stage_config_hash(s, labels, pipeline, video) for s in ("transcript", "audio", "people", "expression")}
    assert all(re.fullmatch(r"[0-9a-f]{12}", h) for h in base.values())
    assert len(set(base.values())) == 4

    prompts = copy.deepcopy(labels)
    for lab in prompts["labels"]:
        if lab.get("prompts"):
            lab["prompts"] = ["x", "y", "z"]
    assert {s: stage_config_hash(s, prompts, pipeline, video) for s in base} == base

    other = copy.deepcopy(pipeline)
    other["signals"]["people"]["person_conf"] = 0.5
    other["moments"]["close_ratio"] = 0.5
    other["models"]["main"]["hpc"] = "other"
    changed = {s: stage_config_hash(s, labels, other, video) for s in base}
    assert changed["people"] != base["people"]
    assert {s: changed[s] for s in ("transcript", "audio", "expression")} == \
        {s: base[s] for s in ("transcript", "audio", "expression")}

    mapped = copy.deepcopy(pipeline)
    mapped["signals"]["audio"]["mapping"]["music"] = ["Music", "Song"]
    assert stage_config_hash("audio", labels, mapped, video) != base["audio"]
    boxed = copy.deepcopy(pipeline)
    boxed["signals"]["people"]["min_box_height_frac"] = 0.3
    assert stage_config_hash("people", labels, boxed, video) != base["people"]
    assert stage_config_hash("expression", labels, boxed, video) == base["expression"]
    sized = copy.deepcopy(pipeline)
    sized["models"]["transcript"]["mac"] = "medium"  # default hardware is mac; an hpc edit must not change it
    assert stage_config_hash("transcript", labels, sized, video) != base["transcript"]


def test_stage_config_hash_by_video_override_only_that_video(pipeline, labels):
    from tpa_index.config import stage_config_hash

    v1, v2 = {"video_id": "v01", "kind": "live"}, {"video_id": "v02", "kind": "live"}
    over = copy.deepcopy(pipeline)
    over.setdefault("overrides", {}).setdefault("by_video", {})["v01"] = {"frames_per_window": 8}
    assert stage_config_hash("people", labels, over, v1) != stage_config_hash("people", labels, pipeline, v1)
    assert stage_config_hash("people", labels, over, v2) == stage_config_hash("people", labels, pipeline, v2)
    with pytest.raises(ConfigError, match="unknown signal stage"):
        stage_config_hash("predict", labels, pipeline, v1)


def test_shipped_weights_are_pinned(pipeline):
    m = pipeline["models"]
    entries = [m["people"]["hpc"], m["people"]["mac"], m["expression"]["model"], m["expression"]["face_detector"]]
    assert all(re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) for e in entries)
    assert re.fullmatch(r"[0-9a-f]{40}", m["expression"]["face_detector"]["revision"])


def test_face_detector_needs_revision(pipeline, labels):
    from tpa_index.config import _check_pipeline_labels

    bad = copy.deepcopy(pipeline)
    bad["models"]["expression"]["face_detector"]["revision"] = "main"
    with pytest.raises(ConfigError, match="face_detector.revision"):
        _check_pipeline_labels(_check_pipeline(bad), labels)


def test_stage_config_hash_ignores_the_other_hardware(pipeline, labels):
    from tpa_index.config import stage_config_hash

    video = {"video_id": "v01", "kind": "live"}
    hpc_before = stage_config_hash("transcript", labels, pipeline, video, "hpc")
    mac_before = stage_config_hash("people", labels, pipeline, video, "mac")
    edited = copy.deepcopy(pipeline)
    edited["models"]["transcript"]["mac"] = "tiny"  # a Mac-only edit
    edited["models"]["people"]["hpc"]["file"] = "yolo11l.pt"  # an HPC-only edit
    assert stage_config_hash("transcript", labels, edited, video, "hpc") == hpc_before
    assert stage_config_hash("transcript", labels, edited, video, "mac") != \
        stage_config_hash("transcript", labels, pipeline, video, "mac")
    assert stage_config_hash("people", labels, edited, video, "mac") == mac_before
    assert stage_config_hash("people", labels, edited, video, "hpc") != \
        stage_config_hash("people", labels, pipeline, video, "hpc")
    with pytest.raises(ConfigError, match="hardware must be hpc or mac"):
        stage_config_hash("audio", labels, pipeline, video, "gpu")


# ---- 005 insights ------------------------------------------------------------------------------------------------
def test_shipped_insights_block_loads(pipeline):
    ins = pipeline["insights"]
    assert ins["audio_threshold"] == 0.5 and ins["min_support"] == 5
    assert list(ins["indicator_rules"]) == ["solo_on_stage", "group_performance", "voice_projection"]


@pytest.mark.parametrize("change,match", [
    (lambda i: i["indicator_rules"].update(solo_on_stage={"volume": 3}), "unknown condition 'volume'"),
    (lambda i: i["fusion"].append({"label": "singing", "require_any": [{"audio": "dance"}]}),
     "audio: 'dance' is not an audio label"),
    (lambda i: i["indicator_rules"].update(solo_on_stage={"clears": "music"}), "clears: 'music' is not a model-scored"),
    (lambda i: i["indicator_rules"].update(nope={"has_speech": True}), "indicator_rules.nope: not an indicator"),
    (lambda i: i["indicator_rules"].update(
        solo_on_stage={"indicator": "voice_projection"}), "'voice_projection' is not an earlier indicator rule"),
    (lambda i: i.update(audio_threshold=1.5), "audio_threshold must be within"),
    (lambda i: i.update(min_support=0), "min_support must be an integer >= 1"),
    (lambda i: i["indicator_rules"].update(solo_on_stage={"person_count": {"gt": 1}}), "person_count: must be"),
    (lambda i: i["fusion"].append({"label": "music", "require_any": [{"has_speech": True}]}),
     "'music' is not a model-scored activity label"),
])
def test_bad_insights_settings(pipeline, labels, change, match):
    from tpa_index.config import _check_pipeline_labels

    bad = copy.deepcopy(pipeline)
    change(bad["insights"])
    with pytest.raises(ConfigError, match=match):
        _check_pipeline_labels(_check_pipeline(bad), labels)


def test_required_threshold_no_fallback(pipeline):
    from tpa_index.config import required_threshold

    p = copy.deepcopy(pipeline)
    p["label_threshold"] = 0.5
    p["label_thresholds"] = {"m1": {"singing": 0.12}}
    assert required_threshold(p, "m1", "singing") == 0.12
    with pytest.raises(ConfigError, match=r"no calibrated threshold for dance \(model m1\).*calibrate --stage predict"):
        required_threshold(p, "m1", "dance")
    with pytest.raises(ConfigError, match="calibrate --stage audio"):
        required_threshold(p, "m2", "music", stage_hint="audio")
