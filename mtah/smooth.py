"""Smoothing of per-frame evidence into label sequences."""

import numpy as np

from mtah.evidence import NOBODY
from mtah.labels import A, D, H


def majority_vote(near, window):
    """The official baseline's smoothing (starter/baseline.py): a centred majority vote over
    +/- window frames of the nearest-team codes; D where nobody in the window has the ball; ties
    to H. window 0 returns the raw labels."""
    if window <= 0:
        return np.where(near == NOBODY, D, near).astype(np.int8)
    home = np.concatenate(([0], np.cumsum(near == H)))
    away = np.concatenate(([0], np.cumsum(near == A)))
    i = np.arange(len(near))
    lo = np.maximum(0, i - window)
    hi = np.minimum(len(near), i + window + 1)
    nh, na = home[hi] - home[lo], away[hi] - away[lo]
    out = np.full(len(near), D, dtype=np.int8)
    seen = (nh + na) > 0
    out[seen] = np.where(nh[seen] >= na[seen], H, A)
    return out
