from tpa_index.score import uncertain_mixed

ACT = ["singing", "dancing", "acting"]
T = {"singing": 0.5, "dancing": 0.5, "acting": 0.5, "music": 0.5, "solo": 0.5}


def test_clear_winner():
    scores = {"singing": 0.9, "dancing": 0.2, "acting": 0.1, "music": 0.7, "solo": 0.6}
    assert not uncertain_mixed({k: scores[k] for k in ACT}, T, 0.1)


def test_nothing_clears():
    scores = {"singing": 0.3, "dancing": 0.2, "acting": 0.1, "music": 0.7, "solo": 0.1}
    assert uncertain_mixed({k: scores[k] for k in ACT}, T, 0.1)


def test_close_top_two_is_uncertain():
    assert uncertain_mixed({"singing": 0.80, "dancing": 0.75, "acting": 0.1}, T, 0.1)
    assert not uncertain_mixed({"singing": 0.80, "dancing": 0.65, "acting": 0.1}, T, 0.1)


def test_runner_up_below_threshold_still_counts_for_closeness():
    assert uncertain_mixed({"singing": 0.55, "dancing": 0.49, "acting": 0.1}, T, 0.1)


def test_per_label_threshold_is_used():
    scores = {"singing": 0.40, "dancing": 0.20, "acting": 0.05}
    low = {**T, "dancing": 0.15}  # singing below its 0.5; dancing clears its 0.15
    # dancing wins on its own threshold, but uncleared singing scores higher: too close to call
    assert uncertain_mixed(scores, low, 0.1)


def test_ensembling_and_window_embeddings_are_unit_length():
    import numpy as np

    from tests.fakes import FakeEmbedder
    from tpa_index.score import label_embeddings, prompt_embedding, window_embedding

    fake = FakeEmbedder()
    labs = [{"name": "singing", "prompts": ["A", "b", "c"]}, {"name": "x", "source": "rules"}]
    emb = label_embeddings(fake, labs)
    assert set(emb) == {"singing"}  # rules labels are never embedded
    assert abs(np.linalg.norm(emb["singing"]) - 1) < 1e-6
    assert np.allclose(emb["singing"], prompt_embedding(fake, ["a", "b", "c"]), atol=1e-6)  # lowercased
    shuffled = label_embeddings(fake, [{"name": "singing", "prompts": ["c", "A", "b"]}])
    assert np.allclose(emb["singing"], shuffled["singing"], atol=1e-6)  # prompt order does not matter
    w = window_embedding(fake.embed_images(np.zeros((4, 8, 8, 3), np.uint8)))
    assert abs(np.linalg.norm(w) - 1) < 1e-6


def test_sigmoid_scores_range_and_ordering():
    import numpy as np

    from tests.fakes import FakeEmbedder
    from tpa_index.score import sigmoid_scores

    fake = FakeEmbedder()
    w = fake.embed_images(np.full((1, 8, 8, 3), 120, np.uint8))
    match, other = fake.base, fake.embed_texts(["unrelated"])[0]
    s = sigmoid_scores(w, np.stack([match, other]), fake.logit_scale, fake.logit_bias)
    assert s.dtype == np.float32 and ((s > 0) & (s < 1)).all()
    assert s[0, 0] > 0.9 > s[0, 1]


def test_visual_change():
    import numpy as np

    from tpa_index.score import visual_change

    same = np.full((4, 160, 160, 3), 90, np.uint8)
    assert visual_change(same) == 0.0
    flip = np.stack([np.zeros((128, 128, 3), np.uint8), np.full((128, 128, 3), 255, np.uint8)])
    assert abs(visual_change(flip) - 1.0) < 1e-6
    assert visual_change(same[:1]) is None


def test_is_idle_needs_both_signals():
    from tpa_index.score import is_idle

    cfg = {"score_threshold": 0.1, "change_threshold": 0.02}
    assert is_idle(0.5, 0.0, cfg)
    assert not is_idle(0.05, 0.0, cfg)  # prompt does not match
    assert not is_idle(0.5, 0.3, cfg)  # picture changes
    assert not is_idle(0.05, 0.3, cfg)
    assert not is_idle(0.9, None, cfg)  # single frame: change unknown
