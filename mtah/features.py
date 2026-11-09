"""Per-frame features for the v2 evidence model, computed from tracking arrays only.

Every feature is written from the home team's point of view and has a mirror under a team swap
(see swap_teams), so the classifier can be trained team-symmetric by adding swapped copies.
Nothing here assumes a track slot is a person: players enter only through per-team sets.
"""

import numpy as np

from mtah.evidence import NOBODY

FPS = 25
WINDOWS = (13, 50, 125)  # centred half-widths: ~0.5 s, 2 s, 5 s


def _rolling_mean(x, half):
    """Centred mean over +/- half frames, ignoring NaN; NaN where the window has no value."""
    ok = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(x), i + half + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _speed(xy, scale):
    """Per-object speed in m/s from positions (n, ..., 2); NaN where either frame is missing."""
    v = np.full(xy.shape[:-1], np.nan)
    v[1:] = np.linalg.norm(np.diff(xy * scale, axis=0), axis=-1) * FPS
    return v


def compute(ball, players, teams, pitch, near):
    """(n, F) float32 feature matrix and the feature names."""
    scale = np.asarray(pitch, dtype=float)
    n = len(ball)
    seen = ~np.isnan(ball).any(axis=1)
    bm = ball * scale
    feats, names = [], []

    def add(name, x):
        names.append(name)
        feats.append(np.asarray(x, dtype=np.float32))

    # Distance from the ball to the nearest player of each team.
    d = np.linalg.norm(players * scale - bm[:, None, :], axis=2)
    d_home = np.nanmin(np.where(teams == 0, d, np.inf), axis=1)
    d_away = np.nanmin(np.where(teams == 1, d, np.inf), axis=1)
    d_home[~np.isfinite(d_home)] = np.nan
    d_away[~np.isfinite(d_away)] = np.nan
    add("ball_seen", seen)
    add("d_home", d_home)
    add("d_away", d_away)
    add("d_diff", d_home - d_away)

    # Where the ball is: distance outside the pitch (0 inside), distance to the nearest line.
    out_x = np.maximum(0, np.maximum(-bm[:, 0], bm[:, 0] - scale[0]))
    out_y = np.maximum(0, np.maximum(-bm[:, 1], bm[:, 1] - scale[1]))
    add("ball_outside_m", np.where(seen, np.hypot(out_x, out_y), np.nan))
    to_line = np.minimum(np.minimum(bm[:, 0], scale[0] - bm[:, 0]),
                         np.minimum(bm[:, 1], scale[1] - bm[:, 1]))
    add("ball_to_line_m", np.where(seen, to_line, np.nan))

    # Ball speed, clipped: false detections jump, and a jump is not a speed.
    bs = _speed(ball, scale)
    add("ball_speed", np.minimum(bs, 40.0))

    # Collective movement: stoppages make everyone walk.
    ps = _speed(players, scale)
    ps[ps > 12.0] = np.nan  # identity swaps between slots look like teleports
    for t, tag in ((0, "home"), (1, "away")):
        add("speed_" + tag, np.nanmean(np.where(teams == t, ps, np.nan), axis=1))
    add("n_players", (~np.isnan(players[:, :, 0])).sum(axis=1))

    # Team shape: spread along the pitch length.
    px = players[:, :, 0] * scale[0]
    for t, tag in ((0, "home"), (1, "away")):
        xt = np.where(teams == t, px, np.nan)
        add("spread_" + tag, np.nanstd(xt, axis=1))

    # Nearest-team evidence over time.
    nh = np.where(near == NOBODY, np.nan, (near == 0).astype(float))
    for w in WINDOWS:
        add("near_home_share_{}".format(w), _rolling_mean(nh, w))
        add("ball_seen_share_{}".format(w), _rolling_mean(seen.astype(float), w))
    all_speed = np.nanmean(ps, axis=1)
    for w in WINDOWS:
        add("speed_all_{}".format(w), _rolling_mean(all_speed, w))

    X = np.column_stack(feats).astype(np.float32)
    return X, names


def compute_filled(filled, gap, players, teams, pitch, near_filled):
    """Extra features from a gap-filled ball (mtah.ballfill): the same distance and nearest-team
    clues as compute(), on the filled position, plus the shape of the gap the frame sits in."""
    scale = np.asarray(pitch, dtype=float)
    d = np.linalg.norm(players * scale - (filled * scale)[:, None, :], axis=2)
    d_home = np.nanmin(np.where(teams == 0, d, np.inf), axis=1)
    d_away = np.nanmin(np.where(teams == 1, d, np.inf), axis=1)
    d_home[~np.isfinite(d_home)] = np.nan
    d_away[~np.isfinite(d_away)] = np.nan
    feats = [d_home, d_away, d_home - d_away,
             np.minimum(gap["since_s"], 120.0), np.minimum(gap["until_s"], 120.0),
             np.minimum(gap["since_s"] + gap["until_s"], 240.0)]
    names = ["f_d_home", "f_d_away", "f_d_diff", "gap_since_s", "gap_until_s", "gap_len_s"]
    nh = np.where(near_filled == NOBODY, np.nan, (near_filled == 0).astype(float))
    for w in WINDOWS:
        feats.append(_rolling_mean(nh, w))
        names.append("f_near_home_share_{}".format(w))
    return np.column_stack(feats).astype(np.float32), names


def swap_teams(X, names):
    """The same frames seen with home and away exchanged."""
    Y = X.copy()
    idx = {k: i for i, k in enumerate(names)}
    for a, b in (("d_home", "d_away"), ("speed_home", "speed_away"),
                 ("spread_home", "spread_away"), ("f_d_home", "f_d_away")):
        if a in idx:
            Y[:, idx[a]], Y[:, idx[b]] = X[:, idx[b]], X[:, idx[a]]
    for k, i in idx.items():
        if k in ("d_diff", "f_d_diff"):
            Y[:, i] = -X[:, i]
        elif "near_home_share" in k:
            Y[:, i] = 1.0 - X[:, i]
    return Y
