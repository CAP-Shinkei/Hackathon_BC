"""Richer possession clues layered on mtah.features.

featurise() returns every column of mtah.features.featurise() first, unchanged, then the v2 clues:
team shape dynamics, direction-free territory, where the ball was and will be, stillness, long
past/future context and ball-to-player proximity. Team columns are home_/away_ pairs computed by one
formula per side; every other new column is symmetric in the teams, so mtah.features.swap_index()
also describes a team swap for the v2 names. Every rolling window stays inside one span from
mtah.ballfill.segments, so nothing leaks across halftime. Team formulas read only slot values, never
slot order.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mtah import features as base
from mtah.ballfill import fill_linear, segments
from mtah.schema import FPS

HOME, AWAY = base.HOME, base.AWAY
MISSING_DIST_M = base.MISSING_DIST_M
FAST_SPEED_MS = 5.0
STILL_SPEED_MS = 1.0
PLAYER_SPEED_HALF_FRAMES = 2  # 5-frame centred smoothing of player velocities, as in mtah.features
SHAPE_WINDOWS_S = (1.0, 3.0)
STILL_WINDOWS_S = (2.0, 6.0)
BALL_STD_WINDOW_S = 2.0
LONG_WINDOWS_S = (6.0, 12.0)
NEAR_BALL_WINDOWS_S = (1.0,)
EXTENT_QUANTILES = (10.0, 90.0)
EDGE_RANKS = 3


def featurise(match):
    """Return (X float32[T, F], names) for a Match: mtah.features columns first, then the v2 columns."""
    base_X, base_names = base.featurise(match)
    base_index = {name: i for i, name in enumerate(base_names)}
    filled, seen = fill_linear(match.ball_xy, match.blocks)[:2]
    pieces = []
    for start, stop in segments(match.n_frames, match.blocks):
        span = slice(start, stop)
        base_span = {name: base_X[span, i] for name, i in base_index.items()}
        cols = _segment_columns(match, span, filled[span], seen[span], base_span)
        pieces.append((span, cols))
    new_names = list(pieces[0][1])
    names = base_names + new_names
    _check_unique(names)
    X = np.empty((match.n_frames, len(names)), dtype=np.float32)
    X[:, :len(base_names)] = base_X
    for span, cols in pieces:
        X[span, len(base_names):] = np.column_stack([cols[name] for name in new_names])
    return X, names


def _check_unique(names):
    if len(set(names)) != len(names):
        raise ValueError("duplicate feature names in features_v2")


def _segment_columns(match, span, filled, seen, base_span):
    raw = np.asarray(match.ball_xy[span], dtype=np.float64)
    filled64 = filled.astype(np.float64)
    last_xy, next_xy, bracket_s = _seen_anchors(raw, filled64, seen)
    cols = _anchor_columns(last_xy, next_xy, bracket_s)
    home_cols, home_speed = _team_columns(match.home_xy[span], last_xy, next_xy, match.pitch)
    away_cols, away_speed = _team_columns(match.away_xy[span], last_xy, next_xy, match.pitch)
    cols.update(_prefixed(HOME, home_cols))
    cols.update(_prefixed(AWAY, away_cols))
    cols.update(_stillness_columns(home_speed, away_speed))
    cols.update(_ball_motion_columns(filled64))
    cols.update(_long_context_columns(base_span))
    cols.update(_near_ball_columns(base_span))
    return cols


def _prefixed(prefix, cols):
    return {prefix + name: values for name, values in cols.items()}


# --- rolling helpers (time is axis 0; NaN is ignored; windows never cross the span edge) ---

def _window_bounds(n, lo, hi):
    if lo > hi:
        raise ValueError("window offsets must satisfy lo <= hi, got {} > {}".format(lo, hi))
    idx = np.arange(n)
    return np.clip(idx + lo, 0, n), np.clip(idx + hi + 1, 0, n)


def _window_mean(values, lo, hi, fallback):
    """Mean over frame offsets lo..hi (inclusive); fallback where no finite value is in the window."""
    series = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(series)
    zeros = np.zeros((1,) + series.shape[1:])
    sums = np.concatenate([zeros, np.cumsum(np.where(finite, series, 0.0), axis=0)])
    counts = np.concatenate([zeros, np.cumsum(finite, axis=0, dtype=np.float64)])
    start, stop = _window_bounds(len(series), lo, hi)
    count = counts[stop] - counts[start]
    mean = (sums[stop] - sums[start]) / np.maximum(count, 1.0)
    return np.where(count > 0, mean, fallback)


def _window_min(values, lo, hi, fallback):
    """Minimum over frame offsets lo..hi (inclusive); fallback where no finite value is in the window."""
    series = np.asarray(values, dtype=np.float64)
    margin = max(abs(lo), abs(hi))
    padded = np.concatenate([
        np.full(margin, np.inf),
        np.where(np.isfinite(series), series, np.inf),
        np.full(margin, np.inf),
    ])
    windows = sliding_window_view(padded, hi - lo + 1)[margin + lo: margin + lo + len(series)]
    low = windows.min(axis=1)
    return np.where(np.isfinite(low), low, fallback)


def _frames(seconds):
    return int(round(seconds * FPS))


def _rate(series):
    """Frame difference times FPS; first frame NaN."""
    rate = np.full(series.shape, np.nan)
    rate[1:] = (series[1:] - series[:-1]) * FPS
    return rate


def _smoothed_rate(series, seconds):
    """Rate of change averaged over a centred window of the given length in seconds."""
    half = _frames(seconds) // 2
    return _window_mean(_rate(series), -half, half, np.nan)


def _or_zero(values):
    return np.where(np.isfinite(values), values, 0.0)


def _masked_mean(values, axis):
    """Mean over finite entries along axis; NaN where there are none."""
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _finite_or_zero_sorted(values):
    return np.where(np.isfinite(values), values, 0.0)


def _sorted_visible(values):
    """Row-sorted values with missing entries pushed to the end, and the count of finite entries per row."""
    finite = np.isfinite(values)
    return np.sort(np.where(finite, values, np.inf), axis=1), finite.sum(axis=1)


def _row_percentile(sorted_values, count, quantile):
    """Linear-interpolated percentile of the first count entries of each sorted row; NaN where count is 0."""
    last = np.maximum(count - 1, 0) * (quantile / 100.0)
    low = np.floor(last).astype(np.intp)
    high = np.ceil(last).astype(np.intp)
    safe = _finite_or_zero_sorted(sorted_values)
    low_value = np.take_along_axis(safe, low[:, None], axis=1)[:, 0]
    high_value = np.take_along_axis(safe, high[:, None], axis=1)[:, 0]
    value = low_value + (last - low) * (high_value - low_value)
    return np.where(count > 0, value, np.nan)


def _edge_mean(sorted_values, count, from_front):
    """Mean of the EDGE_RANKS smallest (or largest) finite entries of each sorted row; NaN where none."""
    n_slots = sorted_values.shape[1]
    ranks = np.arange(EDGE_RANKS)
    if from_front:
        pos = count[:, None] - 1 - ranks[None, :]
    else:
        pos = np.broadcast_to(ranks[None, :], (len(count), EDGE_RANKS))
    valid = (pos >= 0) & (pos < count[:, None])
    picked = np.take_along_axis(_finite_or_zero_sorted(sorted_values),
                                np.clip(pos, 0, n_slots - 1), axis=1)
    total = np.where(valid, picked, 0.0).sum(axis=1)
    used = valid.sum(axis=1)
    return np.where(used > 0, total / np.maximum(used, 1), np.nan)


# --- ball clues ---

def _seen_neighbours(seen):
    """Index of the last seen frame at or before each frame (-1 if none) and of the first at or after (len if none)."""
    m = len(seen)
    idx = np.arange(m)
    prev = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, m)[::-1])[::-1]
    return prev, nxt


def _seen_anchors(raw, filled, seen):
    """Last and next seen ball positions per frame (the filled value when there is none), and the bracket length in s."""
    m = len(seen)
    prev, nxt = _seen_neighbours(seen)
    has_prev, has_next = prev >= 0, nxt < m
    last_xy = np.where(has_prev[:, None], raw[np.clip(prev, 0, m - 1)], filled)
    next_xy = np.where(has_next[:, None], raw[np.clip(nxt, 0, m - 1)], filled)
    bracket_s = np.where(has_prev & has_next, (nxt - prev) / FPS, 0.0)
    return last_xy, next_xy, bracket_s


def _anchor_columns(last_xy, next_xy, bracket_s):
    gap = next_xy - last_xy
    distance = np.hypot(gap[:, 0], gap[:, 1])
    safe_bracket = np.where(bracket_s > 0, bracket_s, 1.0)
    return {
        "ball_last_seen_x": last_xy[:, 0],
        "ball_last_seen_y": last_xy[:, 1],
        "ball_next_seen_x": next_xy[:, 0],
        "ball_next_seen_y": next_xy[:, 1],
        "ball_seen_gap_dist": distance,
        "ball_seen_gap_speed": np.where(bracket_s > 0, distance / safe_bracket, 0.0),
    }


def _ball_motion_columns(filled):
    half = _frames(BALL_STD_WINDOW_S) // 2
    mean = _window_mean(filled, -half, half, filled)
    mean_sq = _window_mean(np.square(filled), -half, half, np.square(filled))
    variance = np.maximum(mean_sq - np.square(mean), 0.0).sum(axis=1)
    return {"ball_std_centred_2s": np.sqrt(variance)}


# --- team clues ---

def _visible_xy(slots):
    xy = np.asarray(slots, dtype=np.float64)
    visible = np.isfinite(xy).all(axis=2)[..., None]
    return np.where(visible, xy, np.nan)


def _player_speeds(xy):
    velocity = _window_mean(_rate(xy), -PLAYER_SPEED_HALF_FRAMES, PLAYER_SPEED_HALF_FRAMES, np.nan)
    return np.hypot(velocity[..., 0], velocity[..., 1])


def _team_columns(slots, last_xy, next_xy, pitch):
    """Unprefixed team columns and the per-player speed array [T, slots] (NaN when invisible)."""
    xy = _visible_xy(slots)
    if xy.shape[1] == 0:
        raise ValueError("team has no player slots")
    speed = _player_speeds(xy)
    centroid = _masked_mean(xy, axis=1)
    cols = _extent_columns(xy, centroid, pitch)
    cols.update(_motion_columns(xy, centroid, speed))
    cols.update(_ball_proximity_columns(xy, last_xy, next_xy))
    return cols, speed


def _extent_columns(xy, centroid, pitch):
    x, y = xy[..., 0], xy[..., 1]
    centre_x = pitch[0] / 2.0
    centroid_x = np.where(np.isfinite(centroid[:, 0]), centroid[:, 0], centre_x)
    sorted_x, count = _sorted_visible(x)
    sorted_y, _ = _sorted_visible(y)
    low, high = EXTENT_QUANTILES
    length = _row_percentile(sorted_x, count, high) - _row_percentile(sorted_x, count, low)
    width = _row_percentile(sorted_y, count, high) - _row_percentile(sorted_y, count, low)
    rear = _edge_mean(sorted_x, count, from_front=False)
    front = _edge_mean(sorted_x, count, from_front=True)
    return {
        "centroid_dx": centroid_x - centre_x,
        "rear3_x": np.where(np.isfinite(rear), rear, centre_x),
        "front3_x": np.where(np.isfinite(front), front, centre_x),
        "length_p10_90": _or_zero(length),
        "width_p10_90": _or_zero(width),
    }


def _motion_columns(xy, centroid, speed):
    centroid_x = centroid[:, 0]
    offsets = np.hypot(xy[..., 0] - centroid[:, None, 0], xy[..., 1] - centroid[:, None, 1])
    spread = _masked_mean(offsets, axis=1)
    cols = {}
    for seconds in SHAPE_WINDOWS_S:
        cols["centroid_vx_{:g}s".format(seconds)] = _or_zero(_smoothed_rate(centroid_x, seconds))
        cols["spread_rate_{:g}s".format(seconds)] = _or_zero(_smoothed_rate(spread, seconds))
    fast = np.where(np.isfinite(speed), speed, 0.0) > FAST_SPEED_MS
    cols["n_fast"] = fast.sum(axis=1).astype(np.float64)
    return cols


def _nearest_distance(xy, point):
    dist = np.hypot(xy[..., 0] - point[:, None, 0], xy[..., 1] - point[:, None, 1])
    nearest = np.where(np.isfinite(dist), dist, np.inf).min(axis=1)
    return np.where(np.isfinite(nearest), nearest, MISSING_DIST_M)


def _ball_proximity_columns(xy, last_xy, next_xy):
    return {
        "dist_last_seen_ball": _nearest_distance(xy, last_xy),
        "dist_next_seen_ball": _nearest_distance(xy, next_xy),
    }


# --- both teams, and long context ---

def _stillness_columns(home_speed, away_speed):
    speeds = np.concatenate([home_speed, away_speed], axis=1)
    finite = np.isfinite(speeds)
    count = finite.sum(axis=1)
    slow = (np.where(finite, speeds, np.inf) < STILL_SPEED_MS).sum(axis=1)
    share = np.where(count > 0, slow / np.maximum(count, 1), 0.0)
    mean_speed = _or_zero(_masked_mean(speeds, axis=1))
    cols = {}
    for seconds in STILL_WINDOWS_S:
        half = _frames(seconds) // 2
        cols["still_share_centred_{:g}s".format(seconds)] = _window_mean(share, -half, half, share)
        cols["mean_speed_all_centred_{:g}s".format(seconds)] = _window_mean(
            mean_speed, -half, half, mean_speed)
    return cols


def _long_series(base_span):
    home_dist = base_span[HOME + "dist_1st"].astype(np.float64)
    away_dist = base_span[AWAY + "dist_1st"].astype(np.float64)
    return {
        HOME + "dist_1st": home_dist,
        AWAY + "dist_1st": away_dist,
        "ball_seen": base_span["ball_seen"].astype(np.float64),
        "ball_speed": base_span["ball_speed"].astype(np.float64),
        HOME + "nearer_share": (home_dist < away_dist).astype(np.float64),
        AWAY + "nearer_share": (away_dist < home_dist).astype(np.float64),
    }


def _long_context_columns(base_span):
    cols = {}
    for name, values in _long_series(base_span).items():
        for seconds in LONG_WINDOWS_S:
            half_window = _frames(seconds)
            cols["{}_long_past_{:g}s".format(name, seconds)] = _window_mean(
                values, -half_window, -1, values)
            cols["{}_long_future_{:g}s".format(name, seconds)] = _window_mean(
                values, 1, half_window, values)
    return cols


def _near_ball_columns(base_span):
    cols = {}
    for team in (HOME, AWAY):
        dist = base_span[team + "dist_1st"].astype(np.float64)
        for seconds in NEAR_BALL_WINDOWS_S:
            window = _frames(seconds)
            cols[team + "min_dist_past_{:g}s".format(seconds)] = _window_min(dist, -window, -1, dist)
            cols[team + "min_dist_future_{:g}s".format(seconds)] = _window_min(dist, 1, window, dist)
    return cols
