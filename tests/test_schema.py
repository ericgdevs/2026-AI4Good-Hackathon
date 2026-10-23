import re
from datetime import datetime, timezone

import pytest

from tpa_index import schema
from tpa_index.schema import SchemaError, append_reviews, read_reviews, read_table, validate, write_table

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

VALID = {
    "videos": dict(video_id="mock01", filename="mock/mock_live_01.mp4", duration_s=60.0, fps=30.0,
                   orientation="landscape", kind="live", phase="early", cohort=None, split="dev"),
    "segments": dict(segment_id="mock01_0", video_id="mock01", start_s=0.0, end_s=8.0, person_count=None,
                     faces_detected=None, has_speech=None, transcript_chars=None, is_idle=None),
    "predictions": dict(segment_id="mock01_0", label="singing", label_group="activity", score=0.7,
                        model="mock-model", run_id="r1"),
    "moments": dict(moment_id="mock01_0_singing", video_id="mock01", start_s=0.0, end_s=8.0,
                    primary_label="singing", secondary_labels=["music"], indicators=[], confidence=0.7,
                    needs_review=False, run_id="r1"),
    "runs": dict(run_id="r1", stage="segment", model=None, config_hash="abcdef012345", hardware="mock-hardware",
                 n_segments=1, runtime_s=1.5, peak_mem_gb=None, started_at=NOW),
    "reviews": dict(review_id="rv1", target_type="moment", target_id="mock01_0_singing", label="singing",
                    label_role="primary", action="accept", new_label=None, source="ai", mode="review",
                    guideline_version="0.1.0", reviewer="mock-staff-a", note=None, created_at=NOW),
}


def row(table, **changes):
    return {**VALID[table], **changes}


@pytest.mark.parametrize("table", list(VALID))
def test_valid_row_passes(table):
    assert validate(table, [row(table)]).num_rows == 1


def test_schemas_have_no_identity_columns():
    banned = re.compile(r"person_id|track|face_id|embedding|crop")
    assert not [n for s in schema.SCHEMAS.values() for n in s.names if banned.search(n)]


def _bad(table, data, column):
    with pytest.raises(SchemaError) as err:
        validate(table, data)
    assert (err.value.table, err.value.column) == (table, column)


def test_missing_column():
    r = row("segments")
    del r["is_idle"]
    _bad("segments", [r], "is_idle")


@pytest.mark.parametrize("col", ["person_id", "track_id", "embedding"])
def test_extra_identity_column_rejected(col):
    _bad("segments", [row("segments", **{col: 1})], col)


def test_wrong_type():
    _bad("predictions", [row("predictions", score="high")], "score")


def test_null_in_required_column():
    _bad("segments", [row("segments", video_id=None)], "video_id")


def test_bad_split():
    _bad("videos", [row("videos", split="train")], "split")


def test_phase_comes_from_config():
    _bad("videos", [row("videos", phase="finale")], "phase")
    assert validate("videos", [row("videos", phase="finale")], phases=["finale"]).num_rows == 1


def test_score_out_of_range():
    _bad("predictions", [row("predictions", score=1.2)], "score")


def test_end_not_after_start():
    _bad("segments", [row("segments", end_s=0.0)], "end_s")


def test_duplicate_primary_key():
    _bad("segments", [row("segments"), row("segments")], "segment_id")


def test_edit_requires_new_label():
    _bad("reviews", [row("reviews", action="edit")], "new_label")


def test_accept_forbids_new_label():
    _bad("reviews", [row("reviews", new_label="dancing")], "new_label")


def test_write_refuses_overwrite(tmp_path):
    write_table("segments", [row("segments")], "r1")
    with pytest.raises(FileExistsError):
        write_table("segments", [row("segments")], "r1")


def test_invalid_write_leaves_no_file():
    with pytest.raises(SchemaError):
        write_table("segments", [row("segments", end_s=-1.0)], "r2")
    assert read_table("segments").num_rows == 0
    assert not list((schema.output_dir() / "runs").rglob("*.parquet*"))


def test_read_table_concatenates_complete_runs_only():
    for run_id in ("r1", "r2"):
        write_table("segments", [row("segments")], run_id)
        write_table("runs", [row("runs", run_id=run_id)], run_id)
    write_table("segments", [row("segments")], "r3-incomplete")  # no runs.parquet: skipped
    assert read_table("segments").num_rows == 2
    assert read_table("runs").num_rows == 2


def test_reviews_append_only():
    assert append_reviews([row("reviews")]) == 1
    assert append_reviews([row("reviews", review_id="rv2"), {k: v for k, v in row("reviews").items()
                                                              if k not in ("review_id", "created_at")}]) == 2
    out = read_reviews()
    assert out.num_rows == 3
    assert out.schema == schema.SCHEMAS["reviews"]
    assert not [n for n in dir(schema) if re.search(r"update|delete", n, re.I)]


def test_read_reviews_without_db_is_empty():
    assert read_reviews().num_rows == 0
