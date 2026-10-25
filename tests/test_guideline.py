"""006 US4: staff add an observation type in config; a guideline bump is required; the blind checklist never changes."""
import copy
import json

import pytest
import yaml

from tpa_index import config, review, schema
from tpa_index.config import CONFIG_DIR, ConfigError, check_guideline, guideline_fingerprint, load_labels

INTERVIEW = {"name": "interview", "description": "a youth speaking to camera about their work"}


def _write(tmp_path, data, name="labels.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_new_observation_type(tmp_path, monkeypatch, labels, pipeline):
    """SC-006: the new type appears in review mode after a bump; without a bump it is refused; the blind checklist
    is byte-identical."""
    raw = yaml.safe_load((CONFIG_DIR / "labels.yaml").read_text())
    added = copy.deepcopy(raw)
    added["observation_types"].append(INTERVIEW)
    with pytest.raises(ConfigError, match="without bumping guideline_version"):
        check_guideline(load_labels(_write(tmp_path, added)))

    added["guideline_version"] = "0.7.0"
    new = load_labels(_write(tmp_path, added))
    line = check_guideline(new)
    assert line.startswith("guideline: 0.7.0 not recorded") and guideline_fingerprint(new) in line

    before = json.dumps(review.checklist(labels, pipeline), sort_keys=True)
    assert json.dumps(review.checklist(new, pipeline), sort_keys=True) == before

    from tpa_index.app import ReviewController, build_app
    from tpa_index.mock import generate

    run_id = generate(seed=0, n_videos=2)
    official = tmp_path / "official.yaml"
    official.write_text(f"predict: [{run_id}]\n")
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", official)
    assert "interview" in ReviewController(labels=new, pipeline=pipeline).observations
    demo = build_app(labels=new, pipeline=pipeline)
    by_id = {c["props"].get("elem_id"): c["props"] for c in demo.config["components"]}
    assert "interview" in [v for _, v in by_id["review-add-obs"]["choices"]]
    shipped = build_app(labels=labels, pipeline=pipeline)
    blind = lambda d: [(c["props"].get("elem_id"), c["props"].get("choices"), c["props"].get("label"))  # noqa: E731
                       for c in d.config["components"] if str(c["props"].get("elem_id", "")).startswith("blind-")]
    assert blind(demo) == blind(shipped)


def test_fingerprint_rules(tmp_path, labels, capsys, monkeypatch):
    base = guideline_fingerprint(labels)
    cosmetic = copy.deepcopy(labels)
    lab = cosmetic["labels"][0]
    lab["prompts"] = ["something else", "another", "a third"]
    lab["color"] = "#000000"
    lab["display"] = "Nicer name"
    cosmetic["observation_types"][0]["description"] = "reworded"
    assert guideline_fingerprint(cosmetic) == base

    for change in ("rename", "group", "obs_add", "obs_remove"):
        c = copy.deepcopy(labels)
        if change == "rename":
            c["labels"][0]["name"] += "_x"
        elif change == "group":
            c["labels"][0]["group"] = "category" if c["labels"][0]["group"] != "category" else "activity"
        elif change == "obs_add":
            c["observation_types"].append(INTERVIEW)
        else:
            c["observation_types"].pop()
        assert guideline_fingerprint(c) != base, change

    history = yaml.safe_load(config.GUIDELINE_HISTORY.read_text())
    assert history[labels["guideline_version"]] == base  # the shipped vocabulary matches its recorded version
    assert check_guideline(labels) == f"guideline: ok ({labels['guideline_version']})"

    from tpa_index import cli

    assert cli.main(["check"]) == 0
    assert f"guideline: ok ({labels['guideline_version']})" in capsys.readouterr().out


def test_app_launch_refuses_unbumped_vocabulary(tmp_path, monkeypatch):
    from tpa_index import app

    history = tmp_path / "history.yaml"
    history.write_text(f'"{load_labels()["guideline_version"]}": "000000000000"\n')
    monkeypatch.setattr(config, "GUIDELINE_HISTORY", history)
    monkeypatch.setattr("tpa_index.clips.session_dir", lambda: pytest.fail("launch must stop before any setup"))
    with pytest.raises(ConfigError, match="without bumping"):
        app.launch(port=7999)
