from tpa_index.segment import windows


def test_sixty_seconds_default():
    w = windows(60, 8, 2)
    assert [s for s, _ in w] == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
    assert w[-1][1] == 60


def test_short_trailing_window_is_folded():
    w = windows(51, 8, 2)  # window at 48 would be 3 s < 4 s
    assert w[-1] == (42, 51)


def test_trailing_window_of_half_length_is_kept():
    assert windows(52, 8, 2)[-1] == (48, 52)


def test_video_shorter_than_window():
    assert windows(5, 8, 2) == [(0, 5)]


def test_custom_stride():
    assert [s for s, _ in windows(20, 4, 1)][:3] == [0, 3, 6]
