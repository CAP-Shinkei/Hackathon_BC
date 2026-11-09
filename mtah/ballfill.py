"""Ball gap filling: estimate where the ball is on frames where the tracking lost it.

Two layers:
  1. Interpolation: a straight line between the last and the next frame where the ball was seen,
     within the same half. No training.
  2. Learned correction: a gradient-boosted model predicts how far the true ball is from the
     interpolated point, from the gap's shape and the players around that point. It is trained on
     external games damaged by the gap simulator, against their clean ball positions, on live
     frames only (only there does a true position exist in every dataset).

Positions are handled in metres internally; inputs and outputs are normalised 0-1 like everywhere else.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

FPS = 25
K_NEAR = 4  # players described around the interpolated point


def gap_geometry(ball, blocks):
    """Per frame: index of the last and next frame with a ball inside the same block (-1 if none),
    as two int arrays."""
    n = len(ball)
    seen = ~np.isnan(ball).any(axis=1)
    last = np.full(n, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    for a, b in blocks:
        idx = np.where(seen[a:b], np.arange(a, b), -1)
        last[a:b] = np.maximum.accumulate(idx)
        rev = np.where(seen[a:b], np.arange(a, b), b + 1)[::-1]
        nx = np.minimum.accumulate(rev)[::-1]
        nxt[a:b] = np.where(nx > b, -1, nx)
    return last, nxt


def interpolate(ball, blocks):
    """(filled ball, gap features dict). Seen frames are unchanged."""
    last, nxt = gap_geometry(ball, blocks)
    t = np.arange(len(ball))
    out = ball.copy()
    miss = np.isnan(ball).any(axis=1)
    both = miss & (last >= 0) & (nxt >= 0)
    only_last = miss & (last >= 0) & (nxt < 0)
    only_next = miss & (last < 0) & (nxt >= 0)
    frac = np.zeros(len(ball))
    span = np.maximum(nxt - last, 1)
    frac[both] = (t[both] - last[both]) / span[both]
    out[both] = ball[last[both]] + frac[both, None] * (ball[nxt[both]] - ball[last[both]])
    out[only_last] = ball[last[only_last]]
    out[only_next] = ball[nxt[only_next]]
    since = np.where(last >= 0, t - last, 10 * FPS * 60) / FPS
    until = np.where(nxt >= 0, nxt - t, 10 * FPS * 60) / FPS
    since[~miss] = 0
    until[~miss] = 0
    return out, {"since_s": since, "until_s": until, "frac": frac, "missing": miss}


def _context(filled, gap, players, pitch, last_ball, next_ball):
    """Feature matrix for the correction model, metres, around the interpolated point."""
    scale = np.asarray(pitch, dtype=float)
    pm = filled * scale
    rel = players * scale - pm[:, None, :]
    d = np.linalg.norm(rel, axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    order = np.argsort(d, axis=1)[:, :K_NEAR]
    rows = np.arange(len(filled))[:, None]
    near_rel = rel[rows, order]  # (n, K, 2)
    near_rel[~np.isfinite(d[rows, order])] = np.nan
    move = (next_ball - last_ball) * scale
    cols = [pm, near_rel.reshape(len(filled), -1), move,
            np.column_stack([gap["since_s"], gap["until_s"], gap["frac"]])]
    return np.column_stack(cols).astype(np.float32)


def _endpoints(ball, blocks):
    last, nxt = gap_geometry(ball, blocks)
    lb = np.where((last >= 0)[:, None], ball[np.maximum(last, 0)], np.nan)
    nb = np.where((nxt >= 0)[:, None], ball[np.maximum(nxt, 0)], np.nan)
    return lb, nb


def train(items, seed=0, subsample=3):
    """items: [(damaged ball, players, pitch, blocks, clean ball, live mask)].
    Returns (model_x, model_y) predicting the offset from the interpolated point, in metres."""
    Xs, ys = [], []
    for ball, players, pitch, blocks, clean, live in items:
        filled, gap = interpolate(ball, blocks)
        lb, nb = _endpoints(ball, blocks)
        use = gap["missing"] & live & ~np.isnan(clean).any(axis=1) & ~np.isnan(filled).any(axis=1)
        use = np.flatnonzero(use)[::subsample]
        Xs.append(_context(filled[use], {k: v[use] for k, v in gap.items()}, players[use], pitch,
                           lb[use], nb[use]))
        ys.append((clean[use] - filled[use]) * np.asarray(pitch))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    models = []
    for j in range(2):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                          random_state=seed)
        m.fit(X, y[:, j])
        models.append(m)
    return models


def fill(ball, players, pitch, blocks, models=None):
    """(filled ball, gap features). Without models: interpolation only."""
    filled, gap = interpolate(ball, blocks)
    if models is None:
        return filled, gap
    lb, nb = _endpoints(ball, blocks)
    idx = np.flatnonzero(gap["missing"] & ~np.isnan(filled).any(axis=1))
    if len(idx):
        X = _context(filled[idx], {k: v[idx] for k, v in gap.items()}, players[idx], pitch,
                     lb[idx], nb[idx])
        off = np.column_stack([m.predict(X) for m in models]) / np.asarray(pitch)
        filled[idx] = filled[idx] + off
    return filled, gap
