"""Straight-line ball filling inside live halves, plus the gap clues that go with it.

Frames are bridged only inside one span: a block (live half) or one of the spans between blocks.
A ball missing at the edge of its span holds the nearest seen value instead of extrapolating.
"""
import numpy as np

from mtah.schema import FPS

NO_SEEN_SECONDS = 60.0
PITCH_CENTRE_XY = (52.5, 34.0)


def segments(n_frames, blocks):
    """Cover [0, n_frames) with the blocks plus the spans between them, in time order."""
    if n_frames < 1:
        raise ValueError("need at least one frame")
    spans, cursor = [], 0
    for start, stop in sorted((int(a), int(b)) for a, b in blocks):
        if not 0 <= start < stop <= n_frames:
            raise ValueError("bad block {}".format((start, stop)))
        if start < cursor:
            raise ValueError("blocks overlap at frame {}".format(start))
        if start > cursor:
            spans.append((cursor, start))
        spans.append((start, stop))
        cursor = stop
    if cursor < n_frames:
        spans.append((cursor, n_frames))
    return spans


def _as_ball(ball_xy):
    ball = np.asarray(ball_xy, dtype=np.float64)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("ball_xy must have shape [T, 2], got {}".format(ball.shape))
    return ball


def _neighbour_indices(seen):
    """Index of the last seen frame at or before each frame (-1 if none), and of the first at or after (len if none)."""
    m = len(seen)
    idx = np.arange(m)
    prev = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, m)[::-1])[::-1]
    return prev, nxt


def _interpolate(xy, prev, nxt):
    """Linear blend between the bracketing seen frames; holds the nearest one at the edges."""
    m = len(xy)
    idx = np.arange(m)
    left = np.where(prev >= 0, prev, nxt)
    right = np.where(nxt < m, nxt, prev)
    span = right - left
    weight = np.where(span > 0, (idx - left) / np.maximum(span, 1), 0.0)
    return xy[left] * (1.0 - weight)[:, None] + xy[right] * weight[:, None]


def _fill_segment(xy):
    """Filled xy, seen flag, gap length, since-seen and until-seen seconds for one span."""
    m = len(xy)
    seen = np.isfinite(xy).all(axis=1)
    if not seen.any():
        filled = np.tile(np.asarray(PITCH_CENTRE_XY, dtype=np.float64), (m, 1))
        gap = np.full(m, m / FPS)
        no_seen = np.full(m, NO_SEEN_SECONDS)
        return filled, seen, gap, no_seen, no_seen
    prev, nxt = _neighbour_indices(seen)
    idx = np.arange(m)
    gap = np.where(seen, 0, nxt - prev - 1) / FPS
    since = np.where(prev >= 0, (idx - prev) / FPS, NO_SEEN_SECONDS)
    until = np.where(nxt < m, (nxt - idx) / FPS, NO_SEEN_SECONDS)
    return _interpolate(xy, prev, nxt), seen, gap, since, until


def fill_linear(ball_xy, blocks):
    """Fill missing ball frames by straight-line interpolation inside each span.

    Returns (filled float32[T,2], seen bool[T], gap_len_s float32[T],
    since_seen_s float32[T], until_seen_s float32[T]). Gap length is the length of the missing run
    the frame sits in (0 where seen); since and until are seconds to the bracketing seen frames,
    NO_SEEN_SECONDS where there is none in the same span.
    """
    ball = _as_ball(ball_xy)
    n = len(ball)
    filled = np.empty((n, 2))
    seen = np.empty(n, dtype=bool)
    gap, since, until = np.empty(n), np.empty(n), np.empty(n)
    for start, stop in segments(n, blocks):
        parts = _fill_segment(ball[start:stop])
        for target, part in zip((filled, seen, gap, since, until), parts):
            target[start:stop] = part
    return (filled.astype(np.float32), seen, gap.astype(np.float32),
            since.astype(np.float32), until.astype(np.float32))
