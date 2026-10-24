from tpa_index.score import uncertain_mixed

ACT = ["singing", "dance", "acting_theater"]
T = {"singing": 0.5, "dance": 0.5, "acting_theater": 0.5, "music": 0.5, "actor_participant": 0.5}


def test_clear_winner():
    scores = {"singing": 0.9, "dance": 0.2, "acting_theater": 0.1, "music": 0.7, "actor_participant": 0.6}
    assert not uncertain_mixed({k: scores[k] for k in ACT}, T, 0.8)


def test_nothing_clears():
    scores = {"singing": 0.3, "dance": 0.2, "acting_theater": 0.1, "music": 0.7, "actor_participant": 0.1}
    assert uncertain_mixed({k: scores[k] for k in ACT}, T, 0.8)


def test_close_top_two_is_uncertain():
    assert uncertain_mixed({"singing": 0.80, "dance": 0.70, "acting_theater": 0.1}, T, 0.8)  # 1.4 >= 0.8 * 1.6
    assert not uncertain_mixed({"singing": 0.80, "dance": 0.55, "acting_theater": 0.1}, T, 0.8)  # 1.1 < 1.28


def test_runner_up_below_threshold_does_not_count():
    assert not uncertain_mixed({"singing": 0.55, "dance": 0.49, "acting_theater": 0.1}, T, 0.8)


# real SigLIP scale: raw scores and thresholds are tiny and differ by orders of magnitude across labels
TINY = {"singing": 0.001, "dance": 0.3, "acting_theater": 0.01}


def test_tiny_scores_clear_dominant_label_is_not_uncertain():
    # raw gaps are all < 0.1, but singing is 5x its threshold vs dance 1.1x
    assert not uncertain_mixed({"singing": 0.005, "dance": 0.33, "acting_theater": 0.002}, TINY, 0.8)


def test_tiny_scores_within_ratio_are_uncertain():
    # strengths 5.0 and 4.5: within 0.8 of each other
    assert uncertain_mixed({"singing": 0.005, "dance": 0.002, "acting_theater": 0.045}, TINY, 0.8)


def test_primary_is_highest_strength_not_raw_score():
    from tpa_index.score import primary_label

    scores = {"singing": 0.005, "dance": 0.33, "acting_theater": 0.002}  # dance has the higher raw score
    assert primary_label(scores, TINY) == "singing"
    assert primary_label({"singing": 0.0005, "dance": 0.1}, TINY) is None


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
