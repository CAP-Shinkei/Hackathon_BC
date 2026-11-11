"""Sanity checks on a loaded match and on a predicted label sequence.

The submission is produced once, by a person, on data nobody inspected beforehand. A tracking file
read with the wrong scale, the wrong team mapping or the wrong frame alignment still yields a CSV
that passes format validation. These checks turn such a silent failure into a line on the screen.

Every figure is a count or a share over the graded blocks; nothing frame-level is reported. The
limits are generic facts about football and tracking, not values fitted to any game. A warning
never stops the run: it asks the person to look before submitting.
"""
import numpy as np

from mtah.schema import A, D, FPS, H

SECONDS_PER_MINUTE = 60
PITCH_MARGIN_M = 5.0  # players and the ball are legitimately a little outside the lines

# Limits
MAX_BALL_MISSING_SHARE = 0.8
MIN_PLAYERS_MEAN = 6.0
MAX_OFF_PITCH_SHARE = 0.2
MAX_DEAD_SHARE = 0.8
MIN_TEAM_SHARE_OF_LIVE = 0.1
MIN_POSSESSIONS_PER_MINUTE = 1.0
MAX_POSSESSIONS_PER_MINUTE = 8.0


def _graded(blocks, n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in blocks:
        mask[start:stop] = True
    if not mask.any():
        raise ValueError("no graded frames: the match has no blocks")
    return mask


def _off_pitch_share(xy, pitch):
    """Share of the present positions lying beyond the pitch plus its margin."""
    present = ~np.isnan(xy).any(axis=-1)
    if not present.any():
        return 0.0
    points = xy[present]
    limits = np.asarray(pitch, dtype=np.float64) + PITCH_MARGIN_M
    outside = (points < -PITCH_MARGIN_M).any(axis=1) | (points > limits).any(axis=1)
    return float(outside.mean())


def _players_mean(team_xy):
    """Mean number of tracked players per frame."""
    if team_xy.shape[1] == 0:
        return 0.0
    return float((~np.isnan(team_xy).any(axis=2)).sum(axis=1).mean())


def input_summary(match):
    """Aggregates of the tracking inside the graded blocks."""
    graded = _graded(match.blocks, match.n_frames)
    home, away, ball = match.home_xy[graded], match.away_xy[graded], match.ball_xy[graded]
    players = np.concatenate([home, away], axis=1)
    return {
        "graded_frames": int(graded.sum()),
        "ball_missing_share": float(np.isnan(ball).any(axis=1).mean()),
        "home_players_mean": _players_mean(home),
        "away_players_mean": _players_mean(away),
        "ball_off_pitch_share": _off_pitch_share(ball, match.pitch),
        "players_off_pitch_share": _off_pitch_share(players, match.pitch),
    }


def input_warnings(summary):
    """Plain sentences for every implausible input aggregate; empty when all look sane."""
    found = []
    if summary["ball_missing_share"] > MAX_BALL_MISSING_SHARE:
        found.append("the ball is missing on {:.0%} of graded frames: check the tracking file "
                     "and the frame alignment".format(summary["ball_missing_share"]))
    for team in ("home", "away"):
        mean = summary[team + "_players_mean"]
        if mean < MIN_PLAYERS_MEAN:
            found.append("only {:.1f} {} players tracked per frame on average: check the team "
                         "mapping in the metadata".format(mean, team))
    if summary["ball_off_pitch_share"] > MAX_OFF_PITCH_SHARE:
        found.append("{:.0%} of ball positions fall outside the pitch: check the coordinate "
                     "scale".format(summary["ball_off_pitch_share"]))
    if summary["players_off_pitch_share"] > MAX_OFF_PITCH_SHARE:
        found.append("{:.0%} of player positions fall outside the pitch: check the coordinate "
                     "scale".format(summary["players_off_pitch_share"]))
    return found


def output_summary(codes, blocks, possessions):
    """Aggregates of the predicted labels inside the graded blocks."""
    values = np.asarray(codes)
    graded = values[_graded(blocks, len(values))]
    live = int(np.count_nonzero(graded != D))
    home = int(np.count_nonzero(graded == H))
    minutes = len(graded) / (FPS * SECONDS_PER_MINUTE)
    return {
        "dead_share": float(np.count_nonzero(graded == D) / len(graded)),
        "home_share_of_live": home / live if live else None,
        "away_share_of_live": int(np.count_nonzero(graded == A)) / live if live else None,
        "possessions_per_minute": possessions / minutes,
    }


def output_warnings(summary):
    """Plain sentences for every implausible output aggregate; empty when all look sane."""
    found = []
    if summary["dead_share"] > MAX_DEAD_SHARE:
        found.append("{:.0%} of graded frames are dead ball: the model saw almost no live "
                     "play".format(summary["dead_share"]))
    shares = (summary["home_share_of_live"], summary["away_share_of_live"])
    if None not in shares and min(shares) < MIN_TEAM_SHARE_OF_LIVE:
        found.append("one team holds {:.0%} of live play: check the team mapping".format(
            min(shares)))
    rate = summary["possessions_per_minute"]
    if not MIN_POSSESSIONS_PER_MINUTE <= rate <= MAX_POSSESSIONS_PER_MINUTE:
        found.append("{:.1f} possessions per minute, outside the usual {:.0f}-{:.0f}: "
                     "possessions are being merged or split".format(
                         rate, MIN_POSSESSIONS_PER_MINUTE, MAX_POSSESSIONS_PER_MINUTE))
    return found
