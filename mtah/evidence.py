"""Per-frame evidence computed from tracking arrays.

Tracking arrays, whatever the source:
    ball    (n, 2) normalised x, y, NaN where the ball is not tracked
    players (n, k, 2) normalised x, y per track slot, NaN where absent
    teams   (n, k) HOME=0 / AWAY=1 per slot and frame, -1 where unused
"""

import numpy as np

from mtah.labels import D

NOBODY = -1

# FIFA's stadium recommendations give a 105 x 68 m pitch inside a 125 x 85 m grass area, i.e.
# about 10 m of run-off beyond the lines; a ball farther out is in the stands or is not a ball.
FAR_OUTSIDE_M = 10.0


def drop_far_outside(ball, pitch, margin_m=FAR_OUTSIDE_M):
    """Copy of ball with NaN where it lies more than margin_m beyond the pitch lines."""
    scale = np.asarray(pitch, dtype=float)
    m = ball * scale
    out_x = np.maximum(0, np.maximum(-m[:, 0], m[:, 0] - scale[0]))
    out_y = np.maximum(0, np.maximum(-m[:, 1], m[:, 1] - scale[1]))
    out = ball.copy()
    out[np.hypot(out_x, out_y) > margin_m] = np.nan
    return out


def nearest_team(ball, players, teams, pitch):
    """Team of the tracked player closest to the ball, NOBODY where the ball or every player is
    missing. Distances in metres: the pitch is not square, so raw 0-1 coordinates would favour the
    short axis. Same rule as the official baseline (starter/baseline.py)."""
    scale = np.asarray(pitch, dtype=float)
    out = np.full(len(ball), NOBODY, dtype=np.int8)
    for lo in range(0, len(ball), 20000):
        hi = lo + 20000
        d = np.linalg.norm((players[lo:hi] - ball[lo:hi, None, :]) * scale, axis=2)
        d[np.isnan(d) | (teams[lo:hi] < 0)] = np.inf
        best = np.argmin(d, axis=1)
        ok = np.isfinite(d[np.arange(len(d)), best])
        team = teams[lo:hi][np.arange(len(d)), best]
        out[lo:hi][ok] = team[ok]
    return out


def as_raw_labels(near):
    """Nearest-team codes as labels: the team, or D where there is no answer (baseline_raw)."""
    return np.where(near == NOBODY, D, near).astype(np.int8)
