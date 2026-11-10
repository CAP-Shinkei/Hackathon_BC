"""Pitch-aware rules applied to a Match or to label codes before scoring."""
import numpy as np

from mtah.schema import D


def drop_far_outside(match, margin_m=10.0):
    """A copy of the match where ball positions further than margin_m outside the pitch are NaN."""
    if margin_m < 0:
        raise ValueError("margin_m must be non-negative, got {}".format(margin_m))
    length, width = match.pitch
    ball = np.array(match.ball_xy, dtype=np.float32, copy=True)
    x_pos, y_pos = ball[:, 0], ball[:, 1]
    outside = ((x_pos < -margin_m) | (x_pos > length + margin_m)
               | (y_pos < -margin_m) | (y_pos > width + margin_m))
    ball[outside] = np.nan
    return match.with_ball(ball)


def dead_before(codes, start, stop):
    """A copy of the label codes with frames [start, stop) set to D (dead time)."""
    values = np.array(codes, dtype=np.int8, copy=True)
    if not 0 <= start <= stop <= len(values):
        raise ValueError("bad span ({}, {}) for {} frames".format(start, stop, len(values)))
    values[start:stop] = D
    return values
