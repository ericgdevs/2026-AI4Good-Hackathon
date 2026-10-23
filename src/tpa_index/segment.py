"""Split a video into fixed windows and sample frames and audio per window."""


def windows(duration_s: float, window_s: float, overlap_s: float) -> list[tuple[float, float]]:
    stride = window_s - overlap_s
    out: list[tuple[float, float]] = []
    i = 0
    while (start := i * stride) < duration_s:
        end = min(start + window_s, duration_s)
        if out and end - start < window_s / 2:
            out[-1] = (out[-1][0], duration_s)  # fold a short trailing window into the previous one
            break
        out.append((start, end))
        if end >= duration_s:
            break
        i += 1
    return out
