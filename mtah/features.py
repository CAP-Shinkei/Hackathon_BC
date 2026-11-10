"""Per-frame possession clues built from anonymous tracking.

Every team column is a symmetric function of that team's slots, so slot order never matters.
Columns prefixed home_ and away_ come from the same formula applied to each side; that is what
lets swap_index() describe a team swap exactly. Everything is computed per span (see
mtah.ballfill.segments), so nothing leaks across halftime.
"""
import numpy as np

from mtah.ballfill import fill_linear, segments
from mtah.schema import FPS

HOME, AWAY = "home_", "away_"
MISSING_DIST_M = 60.0
NEAR_RADIUS_M = 3.0
WIDE_RADIUS_M = 10.0
RANKS = 3
SMOOTH_HALF = 2  # 5-frame centred smoothing of velocities
CENTRED_S = (0.5, 2.0, 5.0)
PAST_S = (1.0, 3.0)
FUTURE_S = (1.0, 3.0)
KEY_SERIES = ("home_dist_1st", "away_dist_1st", "home_n_within_3m",
              "away_n_within_3m", "ball_speed", "ball_seen")


def featurise(match):
    """Return (X float32[T, F], names) for a Match. Labels are never read."""
    ball = fill_linear(match.ball_xy, match.blocks)
    pieces = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for start, stop in segments(match.n_frames, match.blocks):
            span = slice(start, stop)
            pieces.append((span, _segment_columns(match, span, ball)))
    names = list(pieces[0][1])
    X = np.empty((match.n_frames, len(names)), dtype=np.float32)
    for span, cols in pieces:
        X[span] = np.column_stack([cols[name] for name in names])
    return X, names


def swap_index(names):
    """Column permutation so that featurise(swap_teams(m))[0] == featurise(m)[0][:, swap_index(names)]."""
    position = {name: i for i, name in enumerate(names)}
    partners = []
    for name in names:
        partner = _swap_name(name)
        if partner not in position:
            raise ValueError("no team counterpart for column {}".format(name))
        partners.append(position[partner])
    return np.array(partners, dtype=np.intp)


def _swap_name(name):
    if name.startswith(HOME):
        return AWAY + name[len(HOME):]
    if name.startswith(AWAY):
        return HOME + name[len(AWAY):]
    return name


def _segment_columns(match, span, ball):
    filled, seen, gap, since, until = (arr[span] for arr in ball)
    xy_ball = filled.astype(np.float64)
    cols = _ball_columns(xy_ball, seen, gap, since, until, match.pitch)
    cols.update(_prefixed(HOME, _team_columns(match.home_xy[span], xy_ball, match.pitch)))
    cols.update(_prefixed(AWAY, _team_columns(match.away_xy[span], xy_ball, match.pitch)))
    cols.update(_context_columns(cols))
    return cols


def _prefixed(prefix, cols):
    return {prefix + name: values for name, values in cols.items()}


def _window_mean(values, lo, hi, fallback):
    """Mean over frame offsets lo..hi along axis 0, ignoring NaN; fallback where nothing is finite."""
    finite = np.isfinite(values)
    zeros = np.zeros((1,) + values.shape[1:])
    sums = np.concatenate([zeros, np.cumsum(np.where(finite, values, 0.0), axis=0)])
    counts = np.concatenate([zeros, np.cumsum(finite, axis=0, dtype=np.float64)])
    n = len(values)
    idx = np.arange(n)
    start = np.clip(idx + lo, 0, n)
    stop = np.clip(idx + hi + 1, 0, n)
    count = counts[stop] - counts[start]
    has = count > 0
    mean = (sums[stop] - sums[start]) / np.where(has, count, 1.0)
    return np.where(has, mean, fallback)


def _or_zero(values):
    return np.where(np.isfinite(values), values, 0.0)


def _masked_mean(values, axis):
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean, count


def _masked_std(values, axis):
    mean, _ = _masked_mean(values, axis)
    deviation = values - np.expand_dims(mean, axis)
    variance, _ = _masked_mean(np.square(deviation), axis)
    return np.sqrt(variance)


def _rate(positions):
    """Frame difference times FPS; first frame NaN."""
    rate = np.full(positions.shape, np.nan)
    rate[1:] = (positions[1:] - positions[:-1]) * FPS
    return rate


def _smoothed_velocity(positions):
    return _window_mean(_rate(positions), -SMOOTH_HALF, SMOOTH_HALF, np.nan)


def _player_speed(xy):
    velocity = _smoothed_velocity(xy)
    return np.hypot(velocity[..., 0], velocity[..., 1])


def _ball_columns(ball, seen, gap, since, until, pitch):
    velocity = _smoothed_velocity(ball)
    speed = _or_zero(np.hypot(velocity[:, 0], velocity[:, 1]))
    acceleration = _rate(velocity)
    magnitude = np.hypot(acceleration[:, 0], acceleration[:, 1])
    accel = _or_zero(_window_mean(magnitude, -SMOOTH_HALF, SMOOTH_HALF, np.nan))
    x, y = ball[:, 0], ball[:, 1]
    edge = np.minimum(np.minimum(x, pitch[0] - x), np.minimum(y, pitch[1] - y))
    return {
        "ball_seen": seen.astype(np.float64),
        "ball_gap_len_s": gap.astype(np.float64),
        "ball_since_seen_s": since.astype(np.float64),
        "ball_until_seen_s": until.astype(np.float64),
        "ball_x": x,
        "ball_y": y,
        "ball_edge_dist": edge,
        "ball_speed": speed,
        "ball_accel": accel,
    }


def _team_columns(slots, ball, pitch):
    slots = np.asarray(slots, dtype=np.float64)
    visible = np.isfinite(slots).all(axis=2)[..., None]
    xy = np.where(visible, slots, np.nan)
    dist = np.hypot(xy[..., 0] - ball[:, None, 0], xy[..., 1] - ball[:, None, 1])
    speed = _player_speed(xy)
    cols = _distance_columns(dist)
    cols["nearest_speed"] = _nearest_value(dist, speed)
    cols["mean_speed"] = _or_zero(_masked_mean(speed, axis=1)[0])
    cols["n_visible"] = np.isfinite(xy[..., 0]).sum(axis=1).astype(np.float64)
    cols.update(_shape_columns(xy, pitch))
    return cols


def _distance_columns(dist):
    n, n_slots = dist.shape
    ranked = np.full((n, RANKS), MISSING_DIST_M)
    kept = min(n_slots, RANKS)
    ranked[:, :kept] = np.sort(dist, axis=1)[:, :kept]
    ranked = np.where(np.isfinite(ranked), ranked, MISSING_DIST_M)
    return {
        "dist_1st": ranked[:, 0],
        "dist_2nd": ranked[:, 1],
        "dist_3rd": ranked[:, 2],
        "n_within_3m": np.sum(dist <= NEAR_RADIUS_M, axis=1).astype(np.float64),
        "n_within_10m": np.sum(dist <= WIDE_RADIUS_M, axis=1).astype(np.float64),
    }


def _nearest_value(dist, values):
    n, n_slots = dist.shape
    if n_slots == 0:
        return np.zeros(n)
    usable = np.isfinite(dist)
    nearest = np.argmin(np.where(usable, dist, np.inf), axis=1)
    picked = values[np.arange(n), nearest]
    return _or_zero(np.where(usable.any(axis=1), picked, np.nan))


def _shape_columns(xy, pitch):
    centroid_raw = _masked_mean(xy, axis=1)[0]
    centre = np.array([pitch[0] / 2.0, pitch[1] / 2.0])
    centroid = np.where(np.isfinite(centroid_raw), centroid_raw, centre)
    velocity = _or_zero(_smoothed_velocity(centroid_raw))
    offsets = xy - centroid_raw[:, None, :]
    spread = _or_zero(_masked_mean(np.hypot(offsets[..., 0], offsets[..., 1]), axis=1)[0])
    return {
        "centroid_x": centroid[:, 0],
        "centroid_y": centroid[:, 1],
        "centroid_vx": velocity[:, 0],
        "centroid_vy": velocity[:, 1],
        "spread": spread,
        "width": _or_zero(_masked_std(xy[..., 1], axis=1)),
        "depth": _or_zero(_masked_std(xy[..., 0], axis=1)),
    }


def _frames(seconds):
    return int(round(seconds * FPS))


def _rolling_means(name, series, centred_s):
    out = {}
    for sec in centred_s:
        w = _frames(sec)
        out["{}_centred_{:g}s".format(name, sec)] = _window_mean(series, -w, w, series)
    for sec in PAST_S:
        w = _frames(sec)
        out["{}_past_{:g}s".format(name, sec)] = _window_mean(series, -w, -1, series)
    for sec in FUTURE_S:
        w = _frames(sec)
        out["{}_future_{:g}s".format(name, sec)] = _window_mean(series, 1, w, series)
    return out


def _context_columns(cols):
    out = {}
    for key in KEY_SERIES:
        out.update(_rolling_means(key, cols[key], CENTRED_S))
    for team, other in ((HOME, AWAY), (AWAY, HOME)):
        nearer = (cols[team + "dist_1st"] < cols[other + "dist_1st"]).astype(np.float64)
        out.update(_rolling_means(team + "nearer_share", nearer, ()))
    return out
