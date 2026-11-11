"""Per-frame features for the v2 evidence model, computed from tracking arrays only.

Every feature is written from the home team's point of view and has a mirror under a team swap
(see swap_teams), so the classifier can be trained team-symmetric by adding swapped copies.
Nothing here assumes a track slot is a person: players enter only through per-team sets.
"""

import numpy as np
import pandas as pd

from mtah.evidence import NOBODY

FPS = 25
WINDOWS = (13, 50, 125)  # centred half-widths: ~0.5 s, 2 s, 5 s
# 6 s and 12 s: half and one mean controlled possession on the external labels (11-13 s under
# the 2 s rule).
LONG = (150, 300)
PASS = 25          # 1 s: the flight of a 15-20 m pass at 15-20 m/s
WALK_MS = 1.0      # below walking pace (about 1.4 m/s): standing or strolling
SPRINT_MS = 5.5    # 19.8 km/h, the usual high-speed-running threshold in match analysis
# Durations are capped where they stop telling anything: possessions longer than 25 s are only
# 10-14% of the external ones.
RUN_CAP_S = 30.0
# 0.4 s: with the gap simulator's 0.3 m position jitter a velocity taken over 0.4 s is good to
# about 1 m/s (0.3 * sqrt(2) / 0.4); frame-to-frame it would be noise (E11).
KIN = 10
# A velocity break a free ball cannot make: friction and drag take 1-3 m/s^2, about 1 m/s in
# 0.4 s, and the jitter above adds about 2.5 m/s to the measured break.
PLAYED_MS = 6.0


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


def _past(x, w, how="mean"):
    """Statistic of x over the w frames before each frame (the frame itself excluded), NaN
    ignored; NaN where the window holds no value."""
    r = pd.Series(x).rolling(w, min_periods=1)
    return getattr(r, how)().shift(1).to_numpy()


def _future(x, w, how="mean"):
    """The same over the w frames after each frame."""
    return _past(np.asarray(x)[::-1], w, how)[::-1]


def _nearest_m(players, teams, point, scale, team):
    """Distance in metres from each frame's point (n, 2, metres) to that team's nearest player."""
    d = np.linalg.norm(players * scale - point[:, None, :], axis=2)
    out = np.nanmin(np.where(teams == team, d, np.inf), axis=1)
    out[~np.isfinite(out)] = np.nan
    return out


def _extent(pos):
    """Per row, the 90th minus the 10th percentile of the values that are not NaN."""
    ordered = np.sort(pos, axis=1)  # NaN sorts last
    count = (~np.isnan(pos)).sum(axis=1)
    edge = []
    for q in (0.1, 0.9):
        r = np.maximum(count - 1, 0) * q
        lo, hi = np.floor(r).astype(int), np.ceil(r).astype(int)
        a = np.take_along_axis(ordered, lo[:, None], axis=1)[:, 0]
        b = np.take_along_axis(ordered, hi[:, None], axis=1)[:, 0]
        edge.append(a + (r - lo) * (b - a))
    return np.where(count > 0, edge[1] - edge[0], np.nan)


def compute_context(ball, players, teams, pitch, near_filled):
    """Longer and one-sided context, after the team's E18 (CAP-Shinkei): what happened before
    and after each frame separately, the players' collective state, and where each team stands
    relative to the last and the next place the ball was seen."""
    scale = np.asarray(pitch, dtype=float)
    seen = ~np.isnan(ball).any(axis=1)
    bm = ball * scale
    feats, names = [], []

    def add(name, x):
        names.append(name)
        feats.append(np.asarray(x, dtype=np.float32))

    nh = np.where(near_filled == NOBODY, np.nan, (near_filled == 0).astype(float))
    ps = _speed(players, scale)
    ps[ps > 12.0] = np.nan
    all_speed = np.nanmean(ps, axis=1)
    for w in LONG:
        for side, fn in (("past", _past), ("future", _future)):
            add("f_near_home_share_{}{}".format(side, w), fn(nh, w))
            add("ball_seen_share_{}{}".format(side, w), fn(seen.astype(float), w))
    add("speed_all_past{}".format(LONG[0]), _past(all_speed, LONG[0]))
    add("speed_all_future{}".format(LONG[0]), _future(all_speed, LONG[0]))

    # Closest approach of each team to the ball just before and just after.
    for t, tag in ((0, "home"), (1, "away")):
        d = _nearest_m(players, teams, bm, scale, t)
        add("dmin_past_" + tag, _past(d, PASS, "min"))
        add("dmin_future_" + tag, _future(d, PASS, "min"))

    # Everyone standing or walking: a stoppage even when the ball is not seen.
    tracked = ~np.isnan(ps)
    with np.errstate(invalid="ignore", divide="ignore"):
        still = (ps < WALK_MS).sum(axis=1) / tracked.sum(axis=1)
    add("still_share_{}".format(WINDOWS[1]), _rolling_mean(still, WINDOWS[1]))
    add("still_share_{}".format(LONG[0]), _rolling_mean(still, LONG[0] // 2))

    # Team shape: runners, length, width, and whether the block is opening or closing.
    for t, tag in ((0, "home"), (1, "away")):
        mine = teams == t
        add("fast_n_" + tag, (np.where(mine, ps, 0.0) > SPRINT_MS).sum(axis=1))
        for axis, what in ((0, "length"), (1, "width")):
            add("{}_{}".format(what, tag),
                _extent(np.where(mine, players[:, :, axis] * scale[axis], np.nan)))
        spread = np.nanstd(np.where(mine, players[:, :, 0] * scale[0], np.nan), axis=1)
        rate = np.full(len(spread), np.nan)
        rate[PASS:-PASS] = (spread[2 * PASS:] - spread[:-2 * PASS]) / (2 * PASS / FPS)
        add("spread_rate_" + tag, rate)

    # Each team's nearest player to where the ball was last seen and where it is next seen.
    idx = np.arange(len(ball))
    last = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, len(ball))[::-1])[::-1]
    last_xy = np.where((last >= 0)[:, None], bm[np.maximum(last, 0)], np.nan)
    next_xy = np.where((nxt < len(ball))[:, None], bm[np.minimum(nxt, len(ball) - 1)], np.nan)
    for t, tag in ((0, "home"), (1, "away")):
        add("d_last_" + tag, _nearest_m(players, teams, last_xy, scale, t))
        add("d_next_" + tag, _nearest_m(players, teams, next_xy, scale, t))
    return np.column_stack(feats).astype(np.float32), names


def _runs_of(code):
    """Per frame: start index, stop index (exclusive) and ordinal of the constant run it is in."""
    br = np.flatnonzero(code[1:] != code[:-1]) + 1
    starts = np.concatenate(([0], br))
    stops = np.concatenate((br, [len(code)]))
    which = np.repeat(np.arange(len(starts)), stops - starts)
    return starts, stops, which


def compute_turnover(ball, players, teams, pitch, near, near_filled):
    """Clues aimed at the moment possession changes hands: how long the same team has been (and
    stays) nearest to the ball, how the ball's velocity breaks when somebody plays it and who is
    there when it does, and the nearest-team share just before against just after."""
    scale = np.asarray(pitch, dtype=float)
    bm = ball * scale
    n = len(ball)
    feats, names = [], []

    def add(name, x):
        names.append(name)
        feats.append(np.asarray(x, dtype=np.float32))

    # Runs of one team being nearest to the (filled) ball, after the 0.5 s vote that removes
    # frame-to-frame flicker between two players standing at the ball.
    nh = np.where(near_filled == NOBODY, np.nan, (near_filled == 0).astype(float))
    share = _rolling_mean(nh, WINDOWS[0])
    team = np.where(np.isnan(share) | (share == 0.5), 0, np.where(share > 0.5, 1, -1))
    starts, stops, which = _runs_of(team)
    length = (stops - starts) / FPS
    idx = np.arange(n)
    add("s_run_team", team)
    add("run_since_s", np.minimum((idx - starts[which]) / FPS, RUN_CAP_S))
    add("run_until_s", np.minimum((stops[which] - 1 - idx) / FPS, RUN_CAP_S))
    prev_len = np.concatenate(([np.nan], length[:-1]))
    next_len = np.concatenate((length[1:], [np.nan]))
    add("run_prev_s", np.minimum(prev_len[which], RUN_CAP_S))
    add("run_next_s", np.minimum(next_len[which], RUN_CAP_S))

    # Ball velocity over KIN frames before and after each frame, and the size of the break
    # between the two: a rolling or flying ball changes velocity slowly, a played ball abruptly.
    before = np.full((n, 2), np.nan)
    after = np.full((n, 2), np.nan)
    before[KIN:] = (bm[KIN:] - bm[:-KIN]) / (KIN / FPS)
    after[:-KIN] = (bm[KIN:] - bm[:-KIN]) / (KIN / FPS)
    kink = np.linalg.norm(after - before, axis=1)
    add("kink", kink)
    add("ball_speed_s", 0.5 * np.linalg.norm(after + before, axis=1))
    for t, tag in ((0, "home"), (1, "away")):
        mine = np.where(near == t, kink, np.nan)
        add("kink_past_" + tag, _past(mine, WINDOWS[1], "max"))
        add("kink_future_" + tag, _future(mine, WINDOWS[1], "max"))
        played = np.nan_to_num(mine) > PLAYED_MS
        since = idx - np.maximum.accumulate(np.where(played, idx, -10 ** 9))
        until = np.minimum.accumulate(np.where(played, idx, 10 ** 9)[::-1])[::-1] - idx
        add("played_since_" + tag, np.minimum(since / FPS, RUN_CAP_S))
        add("played_until_" + tag, np.minimum(until / FPS, RUN_CAP_S))
        d = _nearest_m(players, teams, bm, scale, t)
        closing = np.full(n, np.nan)
        closing[KIN:-KIN] = (d[:-2 * KIN] - d[2 * KIN:]) / (2 * KIN / FPS)
        add("closing_" + tag, closing)

    for w in (PASS, 3 * PASS):
        add("f_near_home_share_past{}".format(w), _past(nh, w))
        add("f_near_home_share_future{}".format(w), _future(nh, w))
    return np.column_stack(feats).astype(np.float32), names


def swap_teams(X, names):
    """The same frames seen with home and away exchanged."""
    Y = X.copy()
    idx = {k: i for i, k in enumerate(names)}
    for k, i in idx.items():
        if k.endswith("_home"):
            j = idx[k[:-len("home")] + "away"]
            Y[:, i], Y[:, j] = X[:, j], X[:, i]
        elif k in ("d_diff", "f_d_diff") or k.startswith("s_"):
            Y[:, i] = -X[:, i]
        elif "near_home_share" in k:
            Y[:, i] = 1.0 - X[:, i]
    return Y
