import re
import time

import pytest

from tpa_index.config import load_labels
from tpa_index.mock import generate
from tpa_index.schema import SCHEMAS, read_reviews, read_table, validate_dataset

TABLES = ("videos", "segments", "predictions", "moments", "runs")


@pytest.fixture
def mock_root(tmp_path):
    start = time.perf_counter()
    generate(tmp_path / "mock-a", seed=0)
    assert time.perf_counter() - start < 60
    return tmp_path / "mock-a"


def test_full_dataset_is_valid(mock_root):
    for table in TABLES:
        assert read_table(table, mock_root).num_rows > 0, table
    validate_dataset(mock_root)
    assert read_table("videos", mock_root).num_rows >= 3
    assert 200 <= read_table("segments", mock_root).num_rows <= 1000


def test_reviews_cover_every_value(mock_root):
    reviews = read_reviews(mock_root)
    assert set(reviews["action"].to_pylist()) == {"accept", "edit", "reject", "defer", "add"}
    assert set(reviews["source"].to_pylist()) == {"ai", "staff"}
    assert set(reviews["mode"].to_pylist()) == {"review"}  # mock data never writes blind ground truth


def test_moments_mix_review_flags(mock_root):
    assert set(read_table("moments", mock_root)["needs_review"].to_pylist()) == {True, False}


def test_no_identity_columns():
    banned = re.compile(r"person|track|face_id|embedding|crop")
    allowed = {"person_count"}  # aggregate count per segment, not an identity
    assert not [n for s in SCHEMAS.values() for n in s.names if banned.search(n) and n not in allowed]


def test_reproducible_by_seed(mock_root, tmp_path):
    generate(tmp_path / "mock-b", seed=0)
    generate(tmp_path / "mock-c", seed=1)
    for table in TABLES:
        assert read_table(table, mock_root).equals(read_table(table, tmp_path / "mock-b")), table
    assert read_reviews(mock_root).equals(read_reviews(tmp_path / "mock-b"))
    assert not read_table("segments", mock_root).equals(read_table("segments", tmp_path / "mock-c"))


def test_rules_rows_for_every_segment(mock_root):
    import pyarrow.compute as pc

    preds = read_table("predictions", mock_root)
    rules = preds.filter(pc.equal(preds["model"], "rules"))
    assert set(rules["label"].to_pylist()) == {"uncertain_mixed"}
    assert sorted(rules["segment_id"].to_pylist()) == sorted(read_table("segments", mock_root)["segment_id"].to_pylist())
    scored = set(preds.filter(pc.equal(preds["model"], "mock-model"))["label"].to_pylist())
    assert not {"uncertain_mixed", "needs_review"} & scored


def test_reviews_use_the_shipped_guideline(mock_root):
    assert set(read_reviews(mock_root)["guideline_version"].to_pylist()) == {load_labels()["guideline_version"]}


def test_clip_kind_uses_its_own_window(mock_root):
    import pyarrow.compute as pc

    segs = read_table("segments", mock_root)
    clip = segs.filter(pc.equal(segs["video_id"], "mock03"))  # kind clip: 4 s window, 3 s stride
    starts = sorted(clip["start_s"].to_pylist())
    assert starts[1] - starts[0] == 3 and clip["end_s"][0].as_py() - clip["start_s"][0].as_py() == 4
