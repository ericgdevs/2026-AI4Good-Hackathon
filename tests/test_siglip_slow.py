"""Real SigLIP2-base on a synthetic clip. Skipped unless the weights are in the local cache."""
import pytest

pytestmark = pytest.mark.slow
MODEL = "google/siglip2-base-patch16-224"


def _cached():
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(MODEL, "config.json"), str)


@pytest.mark.skipif(not _cached(), reason="siglip2-base weights not in the local Hugging Face cache")
def test_real_model_scores_a_synthetic_clip(clip_videos, pipeline, labels, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from tpa_index.config import model_labels
    from tpa_index.embed import SiglipEmbedder
    from tpa_index.score import predict_video

    embedder = SiglipEmbedder(MODEL, "cpu")
    out = predict_video(clip_videos["t01"], pipeline, labels, embedder, "predict-slow")
    rows = [p for p in out["predictions"] if p["model"] == embedder.name]
    groups = set(pipeline["visual_groups"]) | {pipeline["calibrate"]["null_group"]}
    assert len(rows) == len(out["segments"]) * len(model_labels(labels, groups))
    assert all(0 < r["score"] < 1 for r in rows)
    again = predict_video(clip_videos["t01"], pipeline, labels, embedder, "predict-slow")
    again_rows = [p for p in again["predictions"] if p["model"] == embedder.name]
    assert max(abs(a["score"] - b["score"]) for a, b in zip(rows, again_rows)) < 1e-3
