"""Simulate the defects of broadcast-style ball tracking on clean external matches.

A Regime says how much of the ball track is lost (hidden), where the loss happens (random
moments, crowded or fast-ball moments, or a mix), and which other defects are added (jitter,
false detections, player drops). External matches never lose the ball, so models are trained and
tested on the damaged copies made here. Nothing is mutated: every function returns new arrays.
"""
from dataclasses import dataclass, replace

import numpy as np

from mtah.schema import D, FPS

CROWD_RADIUS_M = 5.0
SPEED_SCALE_MPS = 10.0
MIN_BURST_FRAMES = 3
MAX_BURST_FRAMES = 15
MIN_DROP_S = 0.5
MAX_DROP_S = 3.0
DAMAGE_STREAM = 1
MECHANISMS = ("random", "crowd", "mixed")


def _check_share(regime_name, field, value):
    if not 0.0 <= value <= 1.0:
        raise ValueError("{}: {} must be in [0, 1], got {}".format(regime_name, field, value))


@dataclass(frozen=True)
class Regime:
    name: str
    rate: float  # target share of block frames with the ball hidden
    mechanism: str  # "random", "crowd" or "mixed"
    dead_hidden: float  # probability a dead-ball stretch (label D run) is hidden entirely
    jitter_m: float = 0.0  # gaussian sd added to visible ball positions, metres
    false_rate: float = 0.0  # share of block frames replaced by false ball detections
    median_gap_s: float = 1.0  # median length of a hidden gap, seconds
    sigma: float = 1.0  # lognormal sigma of gap lengths
    player_drop: float = 0.0  # share of player-frames set to NaN, in runs of 0.5-3 s

    def __post_init__(self):
        if self.mechanism not in MECHANISMS:
            raise ValueError("{}: unknown mechanism {!r}".format(self.name, self.mechanism))
        for field in ("rate", "dead_hidden", "false_rate", "player_drop"):
            _check_share(self.name, field, getattr(self, field))
        if self.median_gap_s <= 0 or self.sigma < 0 or self.jitter_m < 0:
            raise ValueError("{}: median_gap_s must be > 0, sigma and jitter_m >= 0".format(self.name))


_MIXED41 = Regime("mixed41", 0.41, "mixed", dead_hidden=0.2)

REGIMES = {
    "random35": Regime("random35", 0.35, "random", dead_hidden=0.0),
    "random50": Regime("random50", 0.50, "random", dead_hidden=0.0),
    "crowd41": Regime("crowd41", 0.41, "crowd", dead_hidden=0.0),
    "mixed41": _MIXED41,
    "mixed41_art": replace(_MIXED41, name="mixed41_art", jitter_m=0.3,
                           false_rate=0.01, player_drop=0.05),
}


def _block_info(match):
    """(in_block bool[T], block_end int[T]): the end of the live half containing each frame."""
    in_block = np.zeros(match.n_frames, dtype=bool)
    block_end = np.zeros(match.n_frames, dtype=np.int64)
    for start, stop in match.blocks:
        in_block[start:stop] = True
        block_end[start:stop] = stop
    return in_block, block_end


def _sampler(weights):
    """Cumulative distribution over frames, normalised to end at 1."""
    cdf = np.cumsum(weights)
    return cdf / cdf[-1]


def _draw(cdf, rng):
    """One frame index drawn from a sampler built by _sampler."""
    index = np.searchsorted(cdf, rng.random(), side="right")
    return int(min(index, len(cdf) - 1))


def _ball_speed(ball):
    """Ball speed in m/s per frame; 0 where the ball or its previous position is missing."""
    step = np.diff(ball, axis=0, prepend=ball[:1]) * FPS
    return np.nan_to_num(np.linalg.norm(step, axis=1))


def _crowd_weights(match):
    """Per-frame weight (1 + players within 5 m of the ball) * (1 + ball speed / 10)."""
    ball = match.ball_xy.astype(np.float64)
    players = np.concatenate([match.home_xy, match.away_xy], axis=1).astype(np.float64)
    distance = np.linalg.norm(players - ball[:, None, :], axis=2)
    crowd = np.sum(distance <= CROWD_RADIUS_M, axis=1)  # NaN distances never count
    return (1.0 + crowd) * (1.0 + _ball_speed(ball) / SPEED_SCALE_MPS)


def _runs(mask):
    """(start, stop) pairs of the True runs in a boolean array."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _dead_ball_hidden(match, in_block, dead_hidden, rng):
    """Hide whole dead-ball stretches (label D inside the halves) with probability dead_hidden."""
    hidden = np.zeros(match.n_frames, dtype=bool)
    dead = (match.labels == D) & in_block
    for start, stop in _runs(dead):
        if rng.random() < dead_hidden:
            hidden[start:stop] = True
    return hidden


def _gap_length(regime, rng):
    """Lognormal gap length in frames, median median_gap_s seconds."""
    median_frames = regime.median_gap_s * FPS
    return max(1, int(round(median_frames * np.exp(regime.sigma * rng.standard_normal()))))


def _pick_mechanism(mechanism, counts, rng):
    """For mixed, the mechanism that has hidden fewer frames so far (coin flip on a tie)."""
    if mechanism != "mixed":
        return mechanism
    if counts["random"] != counts["crowd"]:
        return "random" if counts["random"] < counts["crowd"] else "crowd"
    return "random" if rng.random() < 0.5 else "crowd"


def _hide_span(hidden, start, stop, budget):
    """Hide the not-yet-hidden frames of [start, stop), stopping after `budget` new frames.

    Returns the number of frames newly hidden. Mutates `hidden`, which must be a local array.
    """
    free = ~hidden[start:stop]
    cumulative = np.cumsum(free)
    if cumulative[-1] > budget:
        stop = start + int(np.searchsorted(cumulative, budget)) + 1
        free = free[: stop - start]
    hidden[start:stop] = True
    return int(np.count_nonzero(free))


def _hide_gaps(hidden, regime, match, in_block, block_end, target, rng):
    """Add gaps until the hidden share inside the halves reaches target (in frames)."""
    samplers = {"random": _sampler(in_block.astype(np.float64))}
    if regime.mechanism != "random":
        samplers["crowd"] = _sampler(in_block * _crowd_weights(match))
    counts = {"random": 0, "crowd": 0}
    current = int(np.count_nonzero(hidden & in_block))
    while current < target:
        name = _pick_mechanism(regime.mechanism, counts, rng)
        start = _draw(samplers[name], rng)
        stop = min(start + _gap_length(regime, rng), int(block_end[start]))
        newly = _hide_span(hidden, start, stop, int(np.ceil(target - current)))
        counts[name] += newly
        current += newly


def hidden_mask(match, regime, seed):
    """Boolean [T] mask of the frames where the ball is hidden. Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    in_block, block_end = _block_info(match)
    target = regime.rate * np.count_nonzero(in_block)
    hidden = np.zeros(match.n_frames, dtype=bool)
    if target <= 0:
        return hidden
    if match.labels is not None:
        hidden |= _dead_ball_hidden(match, in_block, regime.dead_hidden, rng)
    _hide_gaps(hidden, regime, match, in_block, block_end, target, rng)
    return hidden


def _hide(ball_xy, hidden):
    """A float64 copy of the ball track with NaN on hidden frames."""
    ball = np.array(ball_xy, dtype=np.float64)
    ball[hidden] = np.nan
    return ball


def _jitter(ball, sd, rng):
    """A new track with gaussian noise on the visible positions (NaN stays NaN)."""
    if sd == 0.0:
        return ball
    return ball + rng.normal(0.0, sd, size=ball.shape)


def _false_bursts(ball, in_block, block_end, regime, pitch, rng):
    """Replace visible block frames with false detections in bursts of 3-15 frames.

    Each burst sits at one random pitch position, so every false detection stays inside the pitch.
    """
    out = ball.copy()
    available = in_block & ~np.isnan(out[:, 0])
    target = regime.false_rate * np.count_nonzero(in_block)
    placed = 0
    while placed < target and available.any():
        start = _draw(_sampler(available.astype(np.float64)), rng)
        length = int(rng.integers(MIN_BURST_FRAMES, MAX_BURST_FRAMES + 1))
        stop = min(start + length, int(block_end[start]))
        span = available[start:stop].copy()
        out[start:stop][span] = (rng.uniform(0.0, pitch[0]), rng.uniform(0.0, pitch[1]))
        available[start:stop] = False
        placed += int(np.count_nonzero(span))
    return out


def _drop_players(home_xy, away_xy, share, rng):
    """Home and away arrays with runs of 0.5-3 s set to NaN until `share` of visible player-frames."""
    players = np.concatenate([home_xy, away_xy], axis=1)
    n_frames, n_slots = players.shape[:2]
    target = share * np.count_nonzero(~np.isnan(players[:, :, 0]))
    min_len = round(MIN_DROP_S * FPS)
    max_len = round(MAX_DROP_S * FPS)
    dropped = 0
    while dropped < target:
        start = int(rng.integers(0, n_frames))
        stop = min(start + int(rng.integers(min_len, max_len + 1)), n_frames)
        slot = int(rng.integers(0, n_slots))
        dropped += int(np.count_nonzero(~np.isnan(players[start:stop, slot, 0])))
        players[start:stop, slot] = np.nan
    n_home = home_xy.shape[1]
    return players[:, :n_home], players[:, n_home:]


def damage(match, regime, seed):
    """A damaged copy of `match`: hidden ball frames, then jitter, false detections, player drops.

    The input is never mutated and labels are carried over unchanged.
    """
    hidden = hidden_mask(match, regime, seed)
    rng = np.random.default_rng([seed, DAMAGE_STREAM])
    in_block, block_end = _block_info(match)
    ball = _hide(match.ball_xy, hidden)
    ball = _jitter(ball, regime.jitter_m, rng)
    ball = _false_bursts(ball, in_block, block_end, regime, match.pitch, rng)
    home, away = _drop_players(match.home_xy, match.away_xy, regime.player_drop, rng)
    return replace(match, home_xy=home, away_xy=away, ball_xy=ball.astype(np.float32))
