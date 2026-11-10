"""The one in-memory shape every loader produces and every later stage consumes.

Coordinates are metres: x in [0, pitch length], y in [0, pitch width]. A missing player or ball
is NaN. Everything runs at 25 fps, the rate of the competition tracking. Label codes follow the
official scorer's order so arrays can be handed to it unchanged.
"""
from dataclasses import dataclass, replace

import numpy as np

FPS = 25
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])


@dataclass(frozen=True)
class Match:
    game_id: str
    home_xy: np.ndarray  # float32 [T, n_home_slots, 2]
    away_xy: np.ndarray  # float32 [T, n_away_slots, 2]
    ball_xy: np.ndarray  # float32 [T, 2]
    blocks: tuple  # ((start, stop), ...) half-open frame spans of live halves
    pitch: tuple = (105.0, 68.0)
    labels: np.ndarray = None  # int8 [T] of H/A/D/X codes, or None when unknown

    def __post_init__(self):
        n = len(self.ball_xy)
        if self.home_xy.shape[0] != n or self.away_xy.shape[0] != n:
            raise ValueError("{}: frame counts differ".format(self.game_id))
        if self.home_xy.shape[2:] != (2,) or self.away_xy.shape[2:] != (2,) \
                or self.ball_xy.shape != (n, 2):
            raise ValueError("{}: bad coordinate shapes".format(self.game_id))
        if self.labels is not None and len(self.labels) != n:
            raise ValueError("{}: labels length differs".format(self.game_id))
        for start, stop in self.blocks:
            if not 0 <= start < stop <= n:
                raise ValueError("{}: bad block {}".format(self.game_id, (start, stop)))

    @property
    def n_frames(self):
        return len(self.ball_xy)

    def with_ball(self, ball_xy):
        """A copy with a different ball track (gap simulation, filling)."""
        return replace(self, ball_xy=np.asarray(ball_xy, dtype=np.float32))

    def with_labels(self, labels):
        return replace(self, labels=np.asarray(labels, dtype=np.int8))


def swap_teams(match):
    """The same match seen from the other side: home<->away, H<->A. For symmetric training."""
    labels = match.labels
    if labels is not None:
        labels = np.where(labels == H, A, np.where(labels == A, H, labels)).astype(np.int8)
    return replace(match, home_xy=match.away_xy, away_xy=match.home_xy, labels=labels)
