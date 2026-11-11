"""Gap simulator: damage clean external tracking so it looks like raw automatic tracking.

The Metrica sample tracking is cleaned: the ball is present on ~99% of live frames and nearly
absent while the ball is dead. Raw automatic tracking, like the competition's, loses the ball
during live play and often still sees it while play is stopped. A model tuned on the clean data
would learn "no ball = dead ball", which is false on raw data.

Every parameter here is set from physical reasoning and swept over a range. None is fitted to
the competition match (its tracking, ground truth or scores).

Mechanisms, each with a parameter in Params:
  live-ball gaps   Gaps start more often where the ball is hard to see: near several players
                   (occlusion) or moving fast (in the air, or blurred). Gap lengths are lognormal.
                   The total share of live frames without ball is a swept target.
  dead-ball sight  While play is stopped, the ball is often placed for the restart and visible.
                   For each dead stretch, the ball is shown at the restart position for a random
                   tail of the stretch.
  position noise   Gaussian jitter on every visible ball position.
  player dropouts  Each player track loses frames in short gaps.
  false detections Off by default. Short bursts where the detector reports a ball that is not
                   there (a head, a boot, a sign), anywhere in the camera's view: uniformly over
                   the pitch plus a margin beyond its lines.
  centre-spot blind Off by default. The ball is not seen within a radius of the centre spot,
                   where the painted spot looks like a ball to the detector and the ball sits
                   still before a kick-off.
"""

from dataclasses import dataclass

import numpy as np

FPS = 25
H, A, D = 0, 1, 2


@dataclass(frozen=True)
class Params:
    live_missing: float = 0.35      # share of live frames without ball (swept)
    gap_median_s: float = 1.0       # median live-ball gap
    gap_sigma: float = 1.0          # lognormal shape of gap lengths
    crowd_radius_m: float = 2.0     # players this close to the ball can hide it
    crowd_weight: float = 1.0       # extra gap hazard per nearby player
    fast_speed_ms: float = 15.0     # ball faster than this: likely in the air or blurred
    fast_weight: float = 4.0        # extra gap hazard while fast
    dead_visible: float = 0.5       # mean share of each dead stretch with the ball visible
    noise_m: float = 0.3            # ball position jitter
    player_missing: float = 0.05    # share of player-frames dropped
    player_gap_s: float = 1.0       # mean player gap
    false_per_min: float = 0.0      # false ball detection bursts per minute (swept)
    false_len_s: float = 0.2        # mean burst length
    false_margin_m: float = 35.0    # how far beyond the lines the camera's view reaches
    centre_blind_m: float = 0.0     # ball unseen this close to the centre spot (0 = off)


# Training profiles beyond the plain gap rates (E25), as Params fields: E13's artefacts (false
# detections, centre-spot blind spot), and stoppages left dark with few live gaps, at the
# figures measured on SkillCorner's real tracking (E11: 7% of live frames without ball, ball
# seen on 21% of dead frames).
TRAIN_EXTRA = (dict(live_missing=0.35, false_per_min=5.0, centre_blind_m=3.0),
               dict(live_missing=0.07, dead_visible=0.2))


def _lognormal_lengths(rng, n, median_s, sigma):
    return np.maximum(1, np.round(rng.lognormal(np.log(median_s * FPS), sigma, n))).astype(int)


def _fill_gaps(mask, eligible, weights, target, rng, median_s, sigma):
    """Set mask True on gaps started at weighted random eligible frames until the share of
    eligible frames covered reaches target."""
    n_elig = eligible.sum()
    if n_elig == 0 or target <= 0:
        return mask
    p = np.where(eligible, weights, 0.0)
    p = p / p.sum()
    mean_len = median_s * FPS * np.exp(sigma ** 2 / 2)
    for _ in range(50):
        have = (mask & eligible).sum() / n_elig
        if have >= target:
            break
        k = max(1, int((target - have) * n_elig / mean_len))
        starts = rng.choice(len(mask), size=k, p=p)
        lengths = _lognormal_lengths(rng, k, median_s, sigma)
        for s, L in zip(starts, lengths):
            mask[s:s + L] = True
    return mask


def damage(ball, players, teams, gt, pitch, params=Params(), seed=0):
    """(ball, players) damaged copies. gt is the answer key (codes H/A/D/X), used only to know
    where the ball is live or dead, which the simulator needs to decide how to hide it."""
    rng = np.random.default_rng(seed)
    scale = np.asarray(pitch, dtype=float)
    n = len(ball)
    live = np.isin(gt, (H, A))
    dead = gt == D

    # Hazard of losing the ball on each frame.
    bm = ball * scale
    speed = np.zeros(n)
    speed[1:] = np.linalg.norm(np.diff(bm, axis=0), axis=1) * FPS
    speed = np.nan_to_num(speed)
    dist = np.linalg.norm(players * scale - bm[:, None, :], axis=2)
    crowd = np.nansum(dist < params.crowd_radius_m, axis=1)
    weights = (1.0 + params.crowd_weight * crowd
               + params.fast_weight * (speed > params.fast_speed_ms))

    hide = np.isnan(ball).any(axis=1) & live  # gaps already in the clean data stay
    hide = _fill_gaps(hide, live, weights, params.live_missing, rng, params.gap_median_s,
                      params.gap_sigma)

    out_ball = ball + rng.normal(0, params.noise_m, ball.shape) / scale
    out_ball[hide] = np.nan

    # Dead stretches: hidden, except a random tail where the ball sits at the restart spot.
    out_ball[dead] = np.nan
    edges = np.flatnonzero(np.diff(np.concatenate(([0], dead.view(np.int8), [0]))))
    for a, b in zip(edges[0::2], edges[1::2]):
        restart = ball[b] if b < n and not np.isnan(ball[b]).any() else None
        if restart is None:
            continue
        tail = int(round((b - a) * min(1.0, rng.exponential(params.dead_visible))))
        if tail:
            out_ball[b - tail:b] = restart + rng.normal(0, params.noise_m, (tail, 2)) / scale

    # Player dropouts, independent per slot.
    out_players = players.copy()
    if params.player_missing > 0:
        k = players.shape[1]
        per_slot = int(params.player_missing * n / (params.player_gap_s * FPS))
        for j in range(k):
            starts = rng.integers(0, n, per_slot)
            lengths = np.maximum(1, rng.geometric(1.0 / (params.player_gap_s * FPS), per_slot))
            for s, L in zip(starts, lengths):
                out_players[s:s + L, j] = np.nan

    # Drawn after everything above, so the default settings reproduce earlier damage exactly.
    if params.centre_blind_m > 0:
        r = np.linalg.norm((out_ball - 0.5) * scale, axis=1)
        out_ball[r < params.centre_blind_m] = np.nan
    if params.false_per_min > 0:
        k = rng.poisson(params.false_per_min * n / FPS / 60)
        starts = rng.integers(0, n, k)
        lengths = np.maximum(1, rng.geometric(1.0 / (params.false_len_s * FPS), k))
        lo = -params.false_margin_m / scale
        hi = 1 + params.false_margin_m / scale
        spots = rng.uniform(lo, hi, (k, 2))
        graded = gt != 3
        for s, L, xy in zip(starts, lengths, spots):
            if graded[s]:
                out_ball[s:s + L] = xy
    return out_ball, out_players


def describe(ball_damaged, gt):
    """Aggregate gap statistics of a damaged ball track."""
    seen = ~np.isnan(ball_damaged).any(axis=1)
    graded = gt != 3
    live = np.isin(gt, (H, A))
    miss = ~seen & graded
    edges = np.flatnonzero(np.diff(np.concatenate(([0], miss.view(np.int8), [0]))))
    lengths = (edges[1::2] - edges[0::2]) / FPS
    return {
        "missing_graded": round(float(miss.sum() / graded.sum()), 3),
        "missing_live": round(float((~seen & live).sum() / live.sum()), 3),
        "seen_dead": round(float((seen & (gt == D)).sum() / max(1, (gt == D).sum())), 3),
        "gap_median_s": round(float(np.median(lengths)), 2) if len(lengths) else 0.0,
        "gaps_over_10s": int((lengths > 10).sum()),
    }
