import numpy as np
import pytest
import yaml

from tpa_index import config
from tpa_index.calibrate import null_floor, propose_idle_threshold, propose_thresholds, write_local_thresholds
from tpa_index.config import idle_threshold, load_pipeline

VIDEOS = [{"video_id": "mock01", "split": "dev"}, {"video_id": "mock02", "split": "dev"},
          {"video_id": "mock04", "split": "eval"}]


def preds(label, scores_by_video, model="m1"):
    return [{"segment_id": f"{vid}_{i * 6000}", "label": label, "score": s, "model": model}
            for vid, scores in scores_by_video.items() for i, s in enumerate(scores)]


def test_percentile_on_known_scores():
    rows = preds("singing", {"mock01": list(np.linspace(0, 1, 51)), "mock02": list(np.linspace(0, 1, 50))})
    ((value, by),) = propose_thresholds(rows, VIDEOS, "m1", 10, 20, 0.0).values()
    assert value == pytest.approx(np.percentile(list(np.linspace(0, 1, 51)) + list(np.linspace(0, 1, 50)), 90))
    assert by is None


def test_eval_split_never_changes_thresholds():
    dev = {"mock01": list(np.linspace(0, 1, 40))}
    a = propose_thresholds(preds("singing", {**dev, "mock04": [0.0] * 40}), VIDEOS, "m1", 10, 20, 0.0)
    b = propose_thresholds(preds("singing", {**dev, "mock04": [1.0] * 40}), VIDEOS, "m1", 10, 20, 0.0)
    assert a == b
    noisy = preds("null_a", {"mock01": [0.1] * 40, "mock04": [0.99] * 40})  # eval null scores never move the floor
    assert null_floor(noisy, VIDEOS, "m1", {"null_a"}, 99, 20)[0] == pytest.approx(0.1)


def test_label_matching_only_noise_is_floored_by_null_floor():
    rng = np.random.default_rng(0)
    nulls = [f"null_{i}" for i in range(5)]
    rows = []
    for name in nulls:  # unrelated prompts: noise between 0.02 and 0.1
        rows += preds(name, {"mock01": list(rng.uniform(0.02, 0.1, 40)), "mock02": list(rng.uniform(0.02, 0.1, 40))})
    rows += preds("acting_theater", {"mock01": list(rng.uniform(0.0, 0.09, 40)),
                                     "mock02": list(rng.uniform(0.0, 0.09, 40))})
    rows += preds("singing", {"mock01": [0.05] * 30 + [0.6] * 10, "mock02": [0.05] * 30 + [0.6] * 10})
    floor, n = null_floor(rows, VIDEOS, "m1", set(nulls), 99, 20)
    assert n == 80 and 0.09 < floor <= 0.1
    got = propose_thresholds(rows, VIDEOS, "m1", 10, 20, 0.0, set(nulls), floor)
    assert set(got) == {"acting_theater", "singing"}  # null labels are never proposed
    assert got["acting_theater"] == (floor, "null")  # matches only noise: held at the null floor
    assert got["singing"][1] is None and got["singing"][0] > floor  # a real signal beats it


def test_label_below_min_threshold_gets_exactly_the_floor():
    rows = preds("acting_theater", {"mock01": list(np.linspace(0.0, 0.0008, 40))})  # top 10% below 0.001
    assert propose_thresholds(rows, VIDEOS, "m1", 10, 20, 0.001, floor=0.0) == {"acting_theater": (0.001, "min")}


def test_min_threshold_still_works_as_hand_override():
    rows = preds("acting_theater", {"mock01": [0.001] * 30})
    assert propose_thresholds(rows, VIDEOS, "m1", 10, 20, 0.05) == {"acting_theater": (0.05, "min")}
    assert propose_thresholds(rows, VIDEOS, "m1", 10, 20, 0.05, floor=0.2) == {"acting_theater": (0.2, "null")}


def test_too_few_segments_and_other_models_ignored():
    rows = preds("singing", {"mock01": [0.5] * 5}) + preds("singing", {"mock01": [0.9] * 40}, model="m2") \
        + preds("uncertain_mixed", {"mock01": [1.0] * 40}, model="rules")
    assert propose_thresholds(rows, VIDEOS, "m1", 10, 20, 0.0) == {"singing": (None, None)}
    assert null_floor(rows, VIDEOS, "m1", {"null_a"}, 99, 20) == (None, 0)


def test_idle_proposal_uses_dev_low_change_segments_only():
    segs = [{"video_id": "mock01", "idle_score": s, "visual_change": 0.001} for s in np.linspace(0.1, 0.5, 30)]
    segs += [{"video_id": "mock01", "idle_score": 0.99, "visual_change": 0.5}] * 30  # moving: ignored
    segs += [{"video_id": "mock04", "idle_score": 0.99, "visual_change": 0.001}] * 30  # eval: ignored
    segs += [{"video_id": "mock01", "idle_score": None, "visual_change": None}] * 5  # old rows: ignored
    got, floored = propose_idle_threshold(segs, VIDEOS, 20, 0.02, 20)
    assert got == pytest.approx(np.percentile(np.linspace(0.1, 0.5, 30), 80)) and not floored
    assert propose_idle_threshold(segs[:10], VIDEOS, 20, 0.02, 20) == (None, False)


def test_idle_proposal_below_min_threshold_is_floored():
    segs = [{"video_id": "mock01", "idle_score": s, "visual_change": 0.001} for s in np.linspace(0.0, 0.0005, 30)]
    assert propose_idle_threshold(segs, VIDEOS, 20, 0.02, 20, min_threshold=0.001) == (0.001, True)
    assert propose_idle_threshold(segs, VIDEOS, 20, 0.02, 20, min_threshold=0.0)[1] is False


def test_write_keeps_other_models(tmp_path):
    path = tmp_path / "t.local.yaml"
    path.write_text(yaml.safe_dump({"label_thresholds": {"mac": {"singing": 0.2}}, "idle_thresholds": {"mac": 0.3}}))
    write_local_thresholds("hpc", {"singing": 0.12, "acting_theater": None}, 0.04, path)
    data = yaml.safe_load(path.read_text())
    assert data["label_thresholds"] == {"mac": {"singing": 0.2}, "hpc": {"singing": 0.12}}
    assert data["idle_thresholds"] == {"mac": 0.3, "hpc": 0.04}
    assert path.read_text().startswith("# written by tpa-index calibrate")


def test_idle_threshold_fallback_and_override(tmp_path, monkeypatch):
    p = load_pipeline()
    assert idle_threshold(p, "hpc") == p["idle"]["score_threshold"]
    local = tmp_path / "t.local.yaml"
    local.write_text("idle_thresholds: {hpc: 0.04}\n")
    monkeypatch.setattr(config, "THRESHOLDS_LOCAL", local)
    p = load_pipeline()
    assert idle_threshold(p, "hpc") == 0.04 and idle_threshold(p, "mac") == p["idle"]["score_threshold"]


# ---- 004: audio and expression stages through the unchanged calibrate rules ---------------------------------------
AST, ENET = "ast-finetuned-audioset-10-10-0.4593", "enet_b0_8_va_mtl"
STAGE_LABELS = {"audio": ["singing_voice", "music", "speech", "applause", "silence"],
                "expression": ["positive", "neutral", "negative", "upset"]}


def _stage_runs(tmp_path, monkeypatch, eval_score=0.5):
    """One official audio and one expression run per video (2 dev, 1 eval), 30 segments each."""
    from datetime import datetime, timezone

    from tpa_index import cli, schema
    from tpa_index.schema import write_table

    monkeypatch.setattr(cli, "install_redaction", lambda: None)
    rng = np.random.default_rng(0)
    official = {"audio": [], "expression": []}
    for vid, split in (("v1", "dev"), ("v2", "dev"), ("v3", "eval")):
        for stage, model in (("audio", AST), ("expression", ENET)):
            rid = f"{stage}-20261024T000000-{vid}"
            write_table("videos", [{"video_id": vid, "filename": f"mock/{vid}.mp4", "duration_s": 200.0, "fps": 30.0,
                                    "orientation": "landscape", "kind": "live", "phase": "early", "cohort": None,
                                    "split": split}], rid)
            segs = [{"segment_id": f"{vid}_{i * 6000}", "video_id": vid, "start_s": i * 6.0, "end_s": i * 6.0 + 8,
                     "person_count": None, "faces_detected": 1 if stage == "expression" else None,
                     "has_speech": None, "transcript_chars": None, "is_idle": None, "idle_score": None,
                     "visual_change": None} for i in range(30)]
            write_table("segments", segs, rid)
            write_table("predictions", [
                {"segment_id": s["segment_id"], "label": label, "label_group": stage, "model": model, "run_id": rid,
                 "score": eval_score if split == "eval" else float(rng.uniform(0, 1))}
                for s in segs for label in STAGE_LABELS[stage]], rid)
            write_table("runs", [{"run_id": rid, "stage": stage, "model": model, "config_hash": "0" * 12,
                                  "hardware": "test", "n_segments": 30, "runtime_s": 1.0, "peak_mem_gb": None,
                                  "started_at": datetime(2026, 10, 24, tzinfo=timezone.utc), "git_commit": None}], rid)
            official[stage].append(rid)
    path = tmp_path / "official.yaml"
    path.write_text(yaml.safe_dump(official))
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)
    return cli


def test_calibrate_audio_and_expression_stages(tmp_path, monkeypatch, capsys):
    cli = _stage_runs(tmp_path, monkeypatch)
    config.THRESHOLDS_LOCAL.write_text(yaml.safe_dump({"label_thresholds": {"siglip2-base-patch16-224": {"singing": 0.1}},
                                                       "idle_thresholds": {"siglip2-base-patch16-224": 0.2}}))
    assert cli.main(["calibrate", "--stage", "audio", "--split", "dev"]) == 0
    out = capsys.readouterr().out
    assert "labels in scored groups without scores: none" in out
    assert cli.main(["calibrate", "--stage", "expression", "--split", "dev"]) == 0
    out = capsys.readouterr().out
    assert "labels in scored groups without scores: confident" in out
    data = yaml.safe_load(config.THRESHOLDS_LOCAL.read_text())
    assert set(data["label_thresholds"][AST]) == set(STAGE_LABELS["audio"])
    assert set(data["label_thresholds"][ENET]) == set(STAGE_LABELS["expression"])
    assert data["label_thresholds"]["siglip2-base-patch16-224"] == {"singing": 0.1}  # visual block kept
    assert data["idle_thresholds"] == {"siglip2-base-patch16-224": 0.2}  # no idle threshold for these stages


def test_calibrate_stage_thresholds_ignore_eval(tmp_path, monkeypatch):
    """SC-009: changing every eval-split score changes no proposed threshold."""
    proposed = []
    for eval_score in (0.0, 1.0):
        (tmp_path / str(eval_score) / "out").mkdir(parents=True)
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / str(eval_score) / "out"))
        cli = _stage_runs(tmp_path / str(eval_score), monkeypatch, eval_score)
        target = tmp_path / f"thr-{eval_score}.yaml"
        monkeypatch.setattr(config, "THRESHOLDS_LOCAL", target)
        assert cli.main(["calibrate", "--stage", "audio", "--split", "dev"]) == 0
        proposed.append(yaml.safe_load(target.read_text())["label_thresholds"][AST])
    assert proposed[0] == proposed[1]


def test_calibrate_stage_refuses_eval(tmp_path, monkeypatch):
    cli = _stage_runs(tmp_path, monkeypatch)
    assert cli.main(["calibrate", "--stage", "expression", "--split", "eval"]) == 3
