import pytest

from tpa_index import schema
from tpa_index.schema import SchemaError, list_runs, read_official, write_table
from tests.test_schema import row


def _run(run_id, video_ids, stage="predict", rules_score=0.0, model="m1"):
    write_table("videos", [row("videos", video_id=v, filename=f"mock/{v}.mp4") for v in video_ids], run_id)
    segs = [row("segments", segment_id=f"{v}_0", video_id=v) for v in video_ids]
    write_table("segments", segs, run_id)
    preds = [row("predictions", segment_id=s["segment_id"], model=model, run_id=run_id) for s in segs]
    preds += [row("predictions", segment_id=s["segment_id"], label="uncertain_mixed", label_group="status",
                  score=rules_score, model="rules", run_id=run_id) for s in segs]
    write_table("predictions", preds, run_id)
    write_table("moments", [row("moments", moment_id=f"{v}_0_singing", video_id=v, run_id=run_id)
                            for v in video_ids], run_id)
    write_table("runs", [row("runs", run_id=run_id, stage=stage)], run_id)


def _official(tmp_path, monkeypatch, text):
    path = tmp_path / "official.yaml"
    path.write_text(text)
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)


def test_default_reads_only_official_runs(tmp_path, monkeypatch):
    for rid, vid in (("p1", "mock01"), ("p2", "mock02"), ("bench", "mock01")):
        _run(rid, [vid])
    _official(tmp_path, monkeypatch, "predict: [p1, p2]\n")
    assert sorted(read_official("segments")["video_id"].to_pylist()) == ["mock01", "mock02"]
    assert set(read_official("predictions")["run_id"].to_pylist()) == {"p1", "p2"}
    assert sorted(read_official("runs")["run_id"].to_pylist()) == ["p1", "p2"]


def test_empty_list_gives_empty_table():
    _run("p1", ["mock01"])
    assert read_official("moments").num_rows == 0


def test_unknown_or_incomplete_run_names_the_id(tmp_path, monkeypatch):
    _run("p1", ["mock01"])
    write_table("segments", [row("segments")], "half")  # no runs.parquet
    for bad in ("nope", "half"):
        _official(tmp_path, monkeypatch, f"predict: [p1, {bad}]\n")
        with pytest.raises(SchemaError, match=f"run {bad} not found or incomplete"):
            read_official("moments")


def test_two_runs_for_one_video_name_both(tmp_path, monkeypatch):
    _run("p1", ["mock01"])
    _run("p2", ["mock01"])
    _official(tmp_path, monkeypatch, "predict: [p1, p2]\n")
    with pytest.raises(SchemaError, match="p1 and p2 both cover video mock01"):
        read_official("segments")


def test_moments_run_takes_precedence_and_replaces_rules_rows(tmp_path, monkeypatch):
    _run("p1", ["mock01"], rules_score=0.0)
    _run("p2", ["mock02"], rules_score=0.0)
    _run("m1run", ["mock01"], stage="moments", rules_score=1.0)
    _official(tmp_path, monkeypatch, "predict: [p1, p2]\nmoments: [m1run]\n")
    moments = read_official("moments").to_pylist()
    assert {(m["video_id"], m["run_id"]) for m in moments} == {("mock01", "m1run"), ("mock02", "p2")}
    assert {s["video_id"] for s in read_official("segments").to_pylist()} == {"mock01", "mock02"}
    rules = {(p["segment_id"], p["score"]) for p in read_official("predictions").to_pylist() if p["model"] == "rules"}
    assert rules == {("mock01_0", 1.0), ("mock02_0", 0.0)}
    model_rows = [p for p in read_official("predictions").to_pylist() if p["model"] == "m1"]
    assert {p["run_id"] for p in model_rows} == {"p1", "p2"}  # model rows always come from predict runs


def test_explicit_stage(tmp_path, monkeypatch):
    _run("p1", ["mock01"])
    _run("m1run", ["mock01"], stage="moments")
    _official(tmp_path, monkeypatch, "predict: [p1]\nmoments: [m1run]\n")
    assert set(read_official("moments", stage="predict")["run_id"].to_pylist()) == {"p1"}


def test_list_runs_skips_incomplete():
    _run("p1", ["mock01", "mock02"])
    write_table("segments", [row("segments")], "half")
    (r,) = list_runs()
    assert r["run_id"] == "p1" and r["video_ids"] == "mock01,mock02" and r["n_predictions"] == 4
    assert r["n_segments"] == 1 and r["hardware"] == "mock-hardware"


def test_stage_value_must_be_a_list(tmp_path, monkeypatch):
    _official(tmp_path, monkeypatch, "predict: predict-20261024T000000-abcd\n")
    with pytest.raises(SchemaError, match="must be a list of run IDs"):
        read_official("moments")


def _stage_run(run_id, video_id, stage="people", model="yolo11n", cfg="aaaaaaaaaaaa", complete=True, started=None):
    from datetime import datetime, timezone

    write_table("videos", [row("videos", video_id=video_id, filename=f"mock/{video_id}.mp4")], run_id)
    if complete:
        write_table("runs", [row("runs", run_id=run_id, stage=stage, model=model, config_hash=cfg,
                                 started_at=started or datetime.now(timezone.utc))], run_id)


def test_find_complete_run_matches_stage_model_hash_and_video():
    from datetime import datetime, timezone

    from tpa_index.schema import find_complete_run

    _stage_run("people-20261024T000000-0001", "v01", started=datetime(2026, 10, 24, tzinfo=timezone.utc))
    _stage_run("people-20261024T010000-0002", "v01", started=datetime(2026, 10, 24, 1, tzinfo=timezone.utc))
    _stage_run("people-20261024T020000-0003", "v02", complete=False)
    assert find_complete_run("people", "v01", "yolo11n", "aaaaaaaaaaaa") == "people-20261024T010000-0002"
    assert find_complete_run("people", "v01", "yolo11n", "bbbbbbbbbbbb") is None  # other config
    assert find_complete_run("people", "v01", "yolo11m", "aaaaaaaaaaaa") is None  # other model
    assert find_complete_run("people", "v02", "yolo11n", "aaaaaaaaaaaa") is None  # incomplete folder
    assert find_complete_run("people", "v03", "yolo11n", "aaaaaaaaaaaa") is None  # other video
    assert find_complete_run("audio", "v01", "yolo11n", "aaaaaaaaaaaa") is None  # other stage


def test_moments_and_calibrate_rebuild_runs_with_old_config_hash(tmp_path, monkeypatch):
    """004: official 002 runs written under another config (0.4.0 labels) still rebuild; nothing compares hashes."""
    from tpa_index import cli, mock
    from tpa_index.schema import read_table

    monkeypatch.setattr(cli, "install_redaction", lambda: None)
    monkeypatch.setattr(mock, "config_hash", lambda labels, pipeline: "000000000000")
    mock.generate(seed=0)
    assert {r["config_hash"] for r in read_table("runs").to_pylist()} == {"000000000000"}
    _official(tmp_path, monkeypatch, "predict: [mock-seed0]\n")
    assert cli.main(["moments"]) == 0
    assert cli.main(["calibrate", "--stage", "predict", "--split", "dev", "--dry-run"]) == 0
    rebuilt = [r["run_id"] for r in read_table("runs").to_pylist() if r["stage"] == "moments"]
    assert rebuilt and read_table("moments", run_ids=rebuilt).num_rows > 0


# ---- read_signals (004) -------------------------------------------------------------------------------------------
def _signal_run(run_id, stage, video_id, segment_ids, **values):
    write_table("videos", [row("videos", video_id=video_id, filename=f"mock/{video_id}.mp4")], run_id)
    write_table("segments", [row("segments", segment_id=s, video_id=video_id, **values) for s in segment_ids], run_id)
    write_table("runs", [row("runs", run_id=run_id, stage=stage)], run_id)


def test_read_signals_joins_official_stage_columns(tmp_path, monkeypatch):
    from tpa_index.schema import read_signals

    _run("p1", ["v1", "v2", "v3"])
    _signal_run("people-1", "people", "v1", ["v1_0"], person_count=4)
    _signal_run("people-2", "people", "v2", ["v2_0"], person_count=7)
    _signal_run("expression-1", "expression", "v1", ["v1_0"], faces_detected=2)
    _official(tmp_path, monkeypatch, "predict: [p1]\npeople: [people-1, people-2]\nexpression: [expression-1]\n")
    rows = {r["video_id"]: r for r in read_signals().to_pylist()}
    assert len(rows) == 3
    assert (rows["v1"]["person_count"], rows["v2"]["person_count"], rows["v3"]["person_count"]) == (4, 7, None)
    assert rows["v1"]["faces_detected"] == 2 and rows["v2"]["faces_detected"] is None
    assert all(r["has_speech"] is None and r["transcript_chars"] is None for r in rows.values())  # no transcript runs


def test_read_signals_missing_segment_names_stage_and_video(tmp_path, monkeypatch):
    from tpa_index.schema import read_signals

    _run("p1", ["v1"])
    _signal_run("transcript-1", "transcript", "v1", ["v1_6000"], has_speech=True, transcript_chars=3)
    _official(tmp_path, monkeypatch, "predict: [p1]\ntranscript: [transcript-1]\n")
    with pytest.raises(SchemaError, match="official transcript run transcript-1 misses segments of video v1"):
        read_signals()


def test_read_signals_two_runs_for_one_video(tmp_path, monkeypatch):
    from tpa_index.schema import read_signals

    _run("p1", ["v1"])
    _signal_run("people-1", "people", "v1", ["v1_0"], person_count=1)
    _signal_run("people-2", "people", "v1", ["v1_0"], person_count=2)
    _official(tmp_path, monkeypatch, "predict: [p1]\npeople: [people-1, people-2]\n")
    with pytest.raises(SchemaError, match="both cover video v1"):
        read_signals()


def test_default_reads_ignore_signal_runs(tmp_path, monkeypatch):
    """US3-AS2: listing audio/expression runs changes nothing in the default official reads."""
    _run("p1", ["v1"])
    _official(tmp_path, monkeypatch, "predict: [p1]\n")
    before = {t: read_official(t).to_pylist() for t in ("segments", "predictions", "moments")}
    _signal_run("audio-1", "audio", "v1", ["v1_0"])
    write_table("predictions", [row("predictions", segment_id="v1_0", label="music", label_group="audio",
                                    model="ast", run_id="audio-x")], "audio-x")
    _official(tmp_path, monkeypatch, "predict: [p1]\naudio: [audio-1]\nexpression: []\n")
    assert {t: read_official(t).to_pylist() for t in before} == before
    assert read_official("segments", stage="audio").num_rows == 1


def test_repo_official_runs_has_signal_sections():
    """004: the repo file lists the four signal stages (empty until the team adds run IDs); 002 lists untouched."""
    from tpa_index.paths import repo_root
    from tpa_index.schema import load_official

    data = load_official(repo_root() / "config/official_runs.yaml")
    assert all(data[s] == [] for s in ("transcript", "audio", "people", "expression"))
    assert len(data["segment"]) == len(data["predict"]) == len(data["moments"]) == 12
    assert all(r.startswith(f"{s}-") for s in ("segment", "predict", "moments") for r in data[s])
