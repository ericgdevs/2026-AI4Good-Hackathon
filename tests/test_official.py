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
