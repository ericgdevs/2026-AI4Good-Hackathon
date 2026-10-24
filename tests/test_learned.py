"""005 learned system: grouped cross-validation without leakage, determinism, clean outputs (mock world only)."""
import numpy as np
import pytest

from tests.insights_fixtures import SECRET, SENTINEL, build_world
from tpa_index.evaluate import block_ids, eval_frame, run_evaluate
from tpa_index.learned import cross_validated, features, fit_final
from tpa_index.view import human_labels, segment_view


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch)
    view = segment_view(w["pipeline"], w["labels"])
    df, _ = eval_frame(view, human_labels(w["labels"], w["pipeline"]))
    return w, view, df


def test_no_block_in_train_and_test(world):
    w, _, df = world
    learned, folds = cross_validated(df, w["labels"], w["pipeline"])
    groups = block_ids(df, w["pipeline"]["insights"]["bootstrap"]["block"])
    assert len(folds) >= 2
    tested = np.concatenate([f["test_rows"] for f in folds])
    assert sorted(tested) == list(range(len(df)))  # every row predicted exactly once
    for f in folds:
        assert not f["train_blocks"] & f["test_blocks"]
        assert set(groups[f["test_rows"]]) == f["test_blocks"]
    assert len(learned) == len(df) and set(learned["segment_id"]) == set(df["segment_id"])


def test_features_have_missing_flags(world):
    w, view, _ = world
    X, cols = features(view, w["labels"], w["pipeline"])
    assert X.shape == (len(view), len(cols)) and not np.isnan(X).any()
    assert {"audio_missing", "people_missing", "expr_missing", "ast:singing_voice"} <= set(cols)
    idle = view.index[view["segment_id"] == "e01_60000"][0]  # no faces -> expression missing
    assert X[idle, cols.index("expr_missing")] == 1.0
    assert not any("filename" in c or "transcript" == c for c in cols)


def test_learned_deterministic(world):
    w, view, df = world
    a, _ = cross_validated(df, w["labels"], w["pipeline"])
    b, _ = cross_validated(df, w["labels"], w["pipeline"])
    assert list(a["model_primary"]) == list(b["model_primary"])
    assert list(a["model_strength"]) == list(b["model_strength"])
    fa, fb = fit_final(df, view, w["labels"], w["pipeline"]), fit_final(df, view, w["labels"], w["pipeline"])
    assert fa.drop(columns=["_model"]).equals(fb.drop(columns=["_model"]))


def test_learned_predictions_cover_all_segments_without_text(world, tmp_path):
    from tpa_index.report import write_result

    w, view, _ = world
    payload = run_evaluate(w["pipeline"], w["labels"])
    preds = payload["learned_predictions"]
    assert sorted(r["segment_id"] for r in preds) == sorted(view["segment_id"])  # every official segment, once
    assert {r["tag"] for r in preds} == {"model estimate (learned)"}
    out = write_result("evaluate", payload)
    text = (out / "learned_predictions.csv").read_text()
    assert SECRET not in text and SENTINEL not in text
    assert text.splitlines()[0] == "segment_id,video_id,primary,primary_prob,activities,categories,actors,tag"
    learned_row = next(h for h in payload["headline"] if h["system"].startswith("learned"))
    assert learned_row["n_segments"] == payload["eval_set"]["labeled_eval_segments"]
