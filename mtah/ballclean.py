"""Remove physically impossible ball detections from automatic tracking.

Automatic tracking sometimes reports the ball for a few frames at an unrelated place and then
jumps back. Each live half is judged on its seen ball frames only. A jump faster than max_speed
starts a suspect run. The run closes at the next too-fast jump back to a frame that is within
reach of the last trusted frame. A run that closes within max_burst_frames is false; a run that
never closes in time is a real relocation and is kept. Nothing is mutated.
"""
import numpy as np

from mtah.schema import FPS

MAX_BALL_SPEED_MPS = 45.0
MAX_BURST_FRAMES = 25


def _check_inputs(ball_xy, blocks, max_speed, max_burst_frames):
    ball = np.asarray(ball_xy)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("ball_xy must have shape [T, 2], got {}".format(ball.shape))
    if not max_speed > 0:
        raise ValueError("max_speed must be positive, got {}".format(max_speed))
    if max_burst_frames < 1:
        raise ValueError("max_burst_frames must be at least 1, got {}".format(max_burst_frames))
    for start, stop in blocks:
        if not 0 <= start < stop <= len(ball):
            raise ValueError("bad block {} for {} frames".format((start, stop), len(ball)))


def _seen_frames(ball, start, stop):
    """Absolute frame indices and float64 positions of the frames in [start, stop) with a ball."""
    window = np.asarray(ball[start:stop], dtype=np.float64)
    seen = np.flatnonzero(np.isfinite(window).all(axis=1))
    return seen + start, window[seen]


def _fast_jumps(idx, pos, max_speed):
    """bool[n-1]: True where the jump from seen frame k to seen frame k+1 exceeds max_speed."""
    distance = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    elapsed = np.diff(idx) / FPS
    return distance > max_speed * elapsed


def _closing_frame(idx, pos, fast, last_good, suspect, max_speed, max_burst):
    """Seen index where a suspect run closes, or None when it does not close within max_burst.

    The run closes at a frame reached by a too-fast jump and within reach of the last trusted frame.
    """
    last_candidate = min(suspect + max_burst, len(idx) - 1)
    candidates = np.arange(suspect + 1, last_candidate + 1)
    elapsed = (idx[candidates] - idx[last_good]) / FPS
    reach = np.linalg.norm(pos[candidates] - pos[last_good], axis=1)
    closing = (reach <= max_speed * elapsed) & fast[candidates - 1]
    hits = np.flatnonzero(closing)
    return int(candidates[hits[0]]) if hits.size else None


def _false_flags(idx, pos, max_speed, max_burst):
    """bool per seen frame: True on the frames of suspect runs that close in time."""
    flags = np.zeros(len(idx), dtype=bool)
    if len(idx) < 2:
        return flags
    fast = _fast_jumps(idx, pos, max_speed)
    jumps = np.flatnonzero(fast)
    last_good = 0
    cursor = 0
    while cursor < len(jumps):
        suspect = int(jumps[cursor]) + 1
        closing = _closing_frame(idx, pos, fast, last_good, suspect, max_speed, max_burst)
        if closing is None:
            last_good = suspect
            cursor += 1
        else:
            flags[suspect:closing] = True
            last_good = closing
            cursor = int(np.searchsorted(jumps, closing))
    return flags


def implausible_mask(ball_xy, blocks, max_speed=MAX_BALL_SPEED_MPS,
                     max_burst_frames=MAX_BURST_FRAMES):
    """bool[T]: True on ball frames judged false detections. Input is not mutated."""
    _check_inputs(ball_xy, blocks, max_speed, max_burst_frames)
    ball = np.asarray(ball_xy)
    mask = np.zeros(len(ball), dtype=bool)
    for start, stop in blocks:
        idx, pos = _seen_frames(ball, start, stop)
        flags = _false_flags(idx, pos, max_speed, max_burst_frames)
        mask[idx[flags]] = True
    return mask


def drop_implausible(match, max_speed=MAX_BALL_SPEED_MPS, max_burst_frames=MAX_BURST_FRAMES):
    """A copy of the match with the implausible ball frames set to NaN. Players are untouched."""
    mask = implausible_mask(match.ball_xy, match.blocks, max_speed, max_burst_frames)
    ball = np.array(match.ball_xy, dtype=np.float32, copy=True)
    ball[mask] = np.nan
    return match.with_ball(ball)
