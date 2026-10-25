"""006 US1: official annotation export and the Devpost sample (mock data only)."""
import csv
import json
import re
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from tpa_index import export, review, schema
from tpa_index.config import EXAMPLE_VIDEOS, load_labels
from tpa_index.export import FIELDS, build_records, write_export
from tpa_index.paths import repo_root

T0 = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
SENTINEL = "TRANSCRIPT_SENTINEL_006"


def moment(mid, primary="dance", secondaries=("performance", "actor_participant", "music"), indicators=(),
           video_id="mock01", start=0.0, end=8.0, confidence=0.7):
    return {"moment_id": mid, "video_id": video_id, "start_s": start, "end_s": end, "primary_label": primary,
            "secondary_labels": list(secondaries), "indicators": list(indicators), "confidence": confidence,
            "needs_review": False, "run_id": "r1"}


def rows(labels, *specs):
    """specs: (moment, label, role, action, new_label, note, minute[, guideline]) -> review rows with fixed times."""
    out = []
    for i, (m, label, role, action, new_label, note, minute, *gv) in enumerate(specs):
        if role == review.OBSERVATION:
            r = review.observation_row(m, label, "mock-staff-a", labels, note)
        else:
            r = review.review_row(m, label, role, action, "mock-staff-a", labels, new_label=new_label, note=note)
        r.update(review_id=f"r{i:03d}", created_at=T0 + timedelta(minutes=minute))
        if gv:
            r["guideline_version"] = gv[0]
        out.append(r)
    return out


def by_id(records, moments):
    """Records come back sorted by (video, start); map them to moment ids via start time."""
    starts = {m["start_s"]: m["moment_id"] for m in moments}
    return {starts[r["start_s"]]: r for r in records}


P, S, I = review.PRIMARY, review.SECONDARY, review.INDICATOR


def test_fields_exact_order(labels, tmp_path):
    """SC-001: every record has exactly the 13 case fields, in order, in JSON and CSV."""
    ms = [moment(f"m{i}", start=i * 10.0, end=i * 10.0 + 8, indicators=("group_performance",) if i % 2 else ())
          for i in range(5)]
    rs = rows(labels, (ms[1], "dance", P, "accept", None, None, 1), (ms[2], "dance", P, "edit", "singing", "x", 2),
              (ms[3], "camera_problem", review.OBSERVATION, "add", None, "blurry", 3))
    records = build_records(ms, rs, labels)
    assert FIELDS == ("video_id", "start_s", "end_s", "primary_label", "secondary_labels", "indicators",
                      "contextual_observation", "confidence", "source", "review_decision", "guideline_version",
                      "reviewer_note", "review_date")
    assert all(tuple(r) == FIELDS for r in records)
    json_path, csv_path = write_export(records, tmp_path, "0.5.0")
    assert all(tuple(obj) == FIELDS for obj in json.loads(json_path.read_text()))
    with open(csv_path, newline="", encoding="utf-8") as f:
        assert tuple(next(csv.reader(f))) == FIELDS


def test_decision_table(labels):
    """SC-002: every review state maps to the expected record (latest per (moment, label) wins)."""
    names = ["none", "accept", "edit", "reject", "defer", "twice", "tie", "sec_reject", "sec_edit", "sec_defer",
             "add", "add_rejected", "obs_only", "old_guideline"]
    ms = [moment(n, start=i * 10.0, end=i * 10.0 + 8, indicators=("group_performance",)) for i, n in enumerate(names)]
    m = {x["moment_id"]: x for x in ms}
    rs = rows(
        labels,
        (m["accept"], "dance", P, "accept", None, None, 1),
        (m["edit"], "dance", P, "edit", "singing", "it is singing", 2),
        (m["reject"], "dance", P, "reject", None, None, 3),
        (m["defer"], "dance", P, "defer", None, None, 4),
        (m["twice"], "dance", P, "accept", None, "first", 5),
        (m["twice"], "dance", P, "reject", None, None, 6),
        (m["sec_reject"], "dance", P, "accept", None, None, 7),
        (m["sec_reject"], "music", S, "reject", None, None, 8),
        (m["sec_edit"], "dance", P, "accept", None, None, 9),
        (m["sec_edit"], "music", S, "edit", "applause", None, 10),
        (m["sec_defer"], "music", S, "defer", None, None, 11),
        (m["add"], "dance", P, "accept", None, None, 12),
        (m["add"], "solo_on_stage", I, "add", None, "added", 13),
        (m["add_rejected"], "singing_voice", S, "add", None, None, 14),
        (m["add_rejected"], "singing_voice", S, "reject", None, None, 15),
        (m["obs_only"], "camera_problem", review.OBSERVATION, "add", None, None, 16),
        (m["old_guideline"], "dance", P, "accept", None, None, 17, "0.4.0"),
    )
    # tie: two rows with the same created_at; review_id decides, the same on every run
    tie = rows(labels, (m["tie"], "dance", P, "defer", None, None, 20), (m["tie"], "dance", P, "accept", None, None, 20))
    tie[0]["review_id"], tie[1]["review_id"] = "zz-last", "aa-first"
    got = by_id(build_records(ms, rs + tie, labels), ms)

    assert (got["none"]["source"], got["none"]["review_decision"]) == ("ai", "pending")
    assert got["none"]["reviewer_note"] == got["none"]["review_date"] == "" and got["none"]["guideline_version"] == load_labels()["guideline_version"]
    assert got["accept"]["review_decision"] == "accepted" and got["accept"]["source"] == "ai"
    assert (got["edit"]["review_decision"], got["edit"]["primary_label"], got["edit"]["source"]) == (
        "edited", "singing", "staff")
    assert got["edit"]["reviewer_note"] == "it is singing" and got["edit"]["review_date"] == "2026-10-24"
    assert got["reject"]["review_decision"] == "rejected" and got["reject"]["primary_label"] == "dance"
    assert got["defer"]["review_decision"] == "deferred"
    assert got["twice"]["review_decision"] == "rejected"
    assert got["twice"]["reviewer_note"] == "" and got["twice"]["review_date"] == "2026-10-24"  # the reject decides
    assert got["sec_reject"]["review_date"] == "2026-10-24" and got["sec_reject"]["reviewer_note"] == ""
    assert got["tie"]["review_decision"] == "deferred"  # same time: the larger review_id is the later row
    assert got["sec_reject"]["secondary_labels"] == ["performance", "actor_participant"]
    assert got["sec_edit"]["secondary_labels"] == ["performance", "actor_participant", "applause"]
    assert got["sec_defer"]["secondary_labels"] == ["performance", "actor_participant", "music"]
    assert got["sec_defer"]["review_decision"] == "pending"
    assert got["add"]["indicators"] == ["group_performance", "solo_on_stage"] and got["add"]["source"] == "staff"
    assert "singing_voice" not in got["add_rejected"]["secondary_labels"] and got["add_rejected"]["source"] == "ai"
    assert got["obs_only"]["source"] == "ai" and got["obs_only"]["contextual_observation"] == "camera_problem"
    assert got["old_guideline"]["guideline_version"] == "0.4.0"

    verified = build_records(ms, rs + tie, labels, verified_only=True)
    assert len(verified) == len(ms) - 2 and all(r["review_decision"] != "rejected" for r in verified)


def test_blind_and_segment_rows_are_ignored(labels):
    m = moment("m1")
    r = rows(labels, (m, "dance", P, "reject", None, None, 1))[0]
    blind = {**r, "mode": "blind", "target_type": "segment", "review_id": "b1"}
    assert build_records([m], [blind], labels)[0]["review_decision"] == "pending"


def test_files_never_overwritten(labels, tmp_path):
    ms = [moment("m1"), moment("m2", start=10, end=18.456, confidence=0.123456)]
    note = 'comma, "quote"\nnew line, café ñ'
    rs = rows(labels, (ms[0], "dance", P, "accept", None, note, 1))
    records = build_records(ms, rs, labels)
    first = write_export(records, tmp_path, "0.5.0")
    second = write_export(records[:1], tmp_path, "0.5.0")
    names = sorted(p.name for p in tmp_path.iterdir())
    pattern = re.compile(r"annotations-\d{8}T\d{6}-[0-9a-f]{4}-g0\.5\.0\.(json|csv)")
    timestamped = [n for n in names if pattern.fullmatch(n)]
    assert len(timestamped) == 4 and set(first).isdisjoint(second)
    assert all(p.exists() for p in (*first, *second))
    assert (tmp_path / "annotations-latest.json").read_text() == second[0].read_text()
    assert (tmp_path / "annotations-latest.csv").read_text() == second[1].read_text()

    with open(first[1], newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert got[0]["reviewer_note"] == note
    assert got[0]["secondary_labels"] == "performance;actor_participant;music"
    assert got[0]["indicators"] == "" and got[1]["review_date"] == ""
    assert got[1]["end_s"] == "18.46" and got[1]["confidence"] == "0.123"
    obj = json.loads(first[0].read_text())
    assert obj[1]["reviewer_note"] == "" and obj[1]["contextual_observation"] == "" and obj[1]["indicators"] == []


def _scan_targets():
    names = [v["filename"] for v in yaml.safe_load(EXAMPLE_VIDEOS.read_text())["videos"]]
    return names + [n.rsplit("/", 1)[-1] for n in names] + [".mp4", SENTINEL, "mock-staff"]


def _assert_clean(text: str, extra=()):
    for needle in [*_scan_targets(), *extra]:
        assert needle not in text, needle
    assert not re.search(r'(^|[\s",])/(Users|home|private|tmp|Volumes)\b', text)


def test_sample_is_mock_only_and_current(tmp_path, monkeypatch):
    """SC-003: the checked-in sample is reproduced exactly, covers every decision, and holds no real filename,
    path or transcript text."""
    from tpa_index import mock

    real = mock.generate

    def with_transcript(out_root=None, **kw):  # a transcript sitting next to the mock runs must never leak
        run_id = real(out_root=out_root, **kw)
        (schema._runs_root(out_root) / run_id / "transcript.json").write_text(json.dumps({"text": SENTINEL}))
        return run_id

    monkeypatch.setattr(mock, "generate", with_transcript)
    json_path, csv_path = export.write_sample(dest=tmp_path)
    shipped = repo_root() / "docs/samples"
    assert json_path.read_bytes() == (shipped / "annotation_export_sample.json").read_bytes()
    assert csv_path.read_bytes() == (shipped / "annotation_export_sample.csv").read_bytes()

    records = json.loads(json_path.read_text())
    assert len(records) == export.SAMPLE_SIZE
    assert {r["review_decision"] for r in records} == {"accepted", "edited", "rejected", "deferred", "pending"}
    assert any(r["source"] == "staff" and r["review_decision"] != "edited" for r in records)  # staff-added label
    assert any(r["contextual_observation"] for r in records)
    assert {r["video_id"] for r in records} <= {"mock01", "mock02", "mock03", "mock04"}
    for path in (json_path, csv_path):
        _assert_clean(path.read_text())


@pytest.fixture
def official_mock(tmp_path, monkeypatch):
    from tpa_index.mock import generate

    run_id = generate(seed=0, n_videos=2)
    path = tmp_path / "official.yaml"
    path.write_text(f"predict: [{run_id}]\n")
    monkeypatch.setattr(schema, "OFFICIAL_RUNS", path)
    (schema._runs_root(None) / run_id / "transcript.json").write_text(json.dumps({"text": SENTINEL}))
    return run_id


def test_export_official_cli(official_mock, labels, capsys):
    from tpa_index import cli
    from tpa_index.paths import output_dir

    moments = schema.read_official("moments").to_pylist()
    schema.append_reviews(export.sample_reviews(moments[:20], labels))
    assert cli.main(["export"]) == 0
    printed = capsys.readouterr().out
    assert str(output_dir()) not in printed and "records:" in printed
    exports = output_dir() / "exports"
    assert len(list(exports.glob(f"annotations-*-g{load_labels()['guideline_version']}.json"))) == 1
    records = json.loads((exports / "annotations-latest.json").read_text())
    assert len(records) == len(moments) and "rejected" in {r["review_decision"] for r in records}
    for p in exports.iterdir():
        _assert_clean(p.read_text(), extra=[str(output_dir())])

    assert cli.main(["export", "--verified-only"]) == 0
    verified = json.loads((exports / "annotations-latest.json").read_text())
    assert verified and all(r["review_decision"] != "rejected" for r in verified)


def test_export_without_official_moments_exits_3(capsys):
    from tpa_index import cli

    assert cli.main(["export"]) == 3
    assert "no official moments" in capsys.readouterr().err


def test_observation_types_in_export(labels):
    """US4 AS4 + Clarification 8: a staff-added type saved under 0.6.0 exports as stored, even once it is removed."""
    import copy

    newer = copy.deepcopy(labels)
    newer["guideline_version"] = "0.6.0"
    newer["observation_types"].append({"name": "interview", "description": "a youth speaking to camera"})
    m1, m2 = moment("m1"), moment("m2", start=10, end=18)
    rs = rows(newer, (m1, "interview", review.OBSERVATION, "add", None, "talks about the song", 1),
              (m1, "interview", review.OBSERVATION, "add", None, None, 2),
              (m2, "dance", P, "accept", None, None, 3))
    got = by_id(build_records([m1, m2], rs, labels), [m1, m2])  # shipped labels: interview is not in config
    assert got["m1"]["contextual_observation"] == "interview: talks about the song; interview"
    assert got["m1"]["review_date"] == "2026-10-24" and got["m1"]["review_decision"] == "pending"
    assert got["m2"]["guideline_version"] == "0.6.0"  # the deciding review's version
