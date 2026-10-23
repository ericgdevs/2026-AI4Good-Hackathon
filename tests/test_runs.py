import re
import time

import numpy as np
import pytest

from tpa_index.config import config_hash
from tpa_index.runs import track_run
from tpa_index.schema import read_table, write_table


def test_successful_run_records_cost():
    with track_run("dummy") as ctx:
        big = np.ones(50_000_000)  # ~0.4 GB
        time.sleep(0.1)
        ctx.n_segments = 3
    del big
    (row,) = read_table("runs").to_pylist()
    assert row["run_id"] == ctx.run_id and row["stage"] == "dummy" and row["n_segments"] == 3
    assert row["runtime_s"] >= 0.1
    assert row["peak_mem_gb"] is None or row["peak_mem_gb"] > 0.3
    assert row["hardware"]
    assert re.fullmatch(r"[0-9a-f]{12}", row["config_hash"])


def test_config_hash_tracks_config(labels, pipeline):
    assert config_hash(labels, pipeline) == config_hash(dict(labels), dict(pipeline))
    assert config_hash(labels, pipeline) != config_hash(labels, {**pipeline, "window_s": 10})


def test_failed_run_reraises_and_is_ignored():
    with pytest.raises(ValueError):
        with track_run("dummy") as ctx:
            write_table("segments", [{"segment_id": "t01_0", "video_id": "t01", "start_s": 0.0, "end_s": 8.0,
                                      "person_count": None, "faces_detected": None, "has_speech": None,
                                      "transcript_chars": None, "is_idle": None}], ctx.run_id)
            raise ValueError("boom")
    assert read_table("runs").num_rows == 0
    assert read_table("segments").num_rows == 0  # written but not complete: skipped by readers
