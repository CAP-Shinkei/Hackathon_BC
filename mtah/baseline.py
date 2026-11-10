"""The official baseline (nearest player to the ball, then a majority vote) on a Match.

The nearest-player step is redone on arrays because the official one reads the raw file; the
smoothing is the official function itself, so the reference number is the organisers' algorithm.
"""
import os
import sys

import numpy as np

from mtah.schema import A, D, H

OFFICIAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "official")
for folder in (OFFICIAL, os.path.join(OFFICIAL, "starter")):
    if folder not in sys.path:
        sys.path.insert(0, folder)

import baseline as official_baseline  # noqa: E402

FAR = np.float32(1e6)
LETTER_TO_CODE = {"H": H, "A": A, "D": D}


def _nearest(team_xy, ball_xy):
    distance = np.linalg.norm(team_xy - ball_xy[:, None, :], axis=2)
    return np.where(np.isnan(distance), FAR, distance).min(axis=1)


def nearest_team(match):
    """Official codes per frame: 0 home, 1 away, -1 where the ball is missing."""
    home, away = _nearest(match.home_xy, match.ball_xy), _nearest(match.away_xy, match.ball_xy)
    codes = np.where(home <= away, official_baseline.HOME, official_baseline.AWAY)
    unseen = np.isnan(match.ball_xy).any(axis=1) | ((home >= FAR) & (away >= FAR))
    return np.where(unseen, official_baseline.NOBODY, codes).astype(np.int8)


def predict(match, window):
    letters = official_baseline.smooth(nearest_team(match), window)
    return np.vectorize(LETTER_TO_CODE.get, otypes=[np.int8])(letters)
