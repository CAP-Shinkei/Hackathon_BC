"""Kick-off detection: the frame the ball leaves the centre spot.

Used for the second half of the competition match, which is graded from before its kick-off;
every graded frame before the kick-off is dead ball (FAQ.md). The search window comes from the
Smart Tagging "Kick Off (Start)" phase, which brackets the restart by a few seconds.
"""

import numpy as np

CENTRE_RADIUS_M = 1.5    # the ball is placed on the spot; allows for tracking jitter
DEPART_RADIUS_M = 3.0    # clearly played: farther than any jitter from the spot
DEPART_WITHIN = 50       # frames (2 s) after the last at-centre frame


def detect(ball, pitch, lo, hi):
    """Last frame in [lo, hi) where the ball is on the centre spot and is then seen more than
    DEPART_RADIUS_M away within DEPART_WITHIN frames. None if no such frame."""
    lo = max(0, lo)
    scale = np.asarray(pitch, dtype=float)
    seg = (ball[lo:hi + DEPART_WITHIN] - 0.5) * scale
    r = np.linalg.norm(seg, axis=1)  # NaN where the ball is missing
    at_centre = np.flatnonzero(r[:hi - lo] <= CENTRE_RADIUS_M)
    for i in at_centre:
        ahead = r[i + 1:i + 1 + DEPART_WITHIN]
        if np.any(ahead > DEPART_RADIUS_M):
            # the latest at-centre frame before the departure
            first_far = i + 1 + int(np.flatnonzero(ahead > DEPART_RADIUS_M)[0])
            last = at_centre[at_centre < first_far].max()
            return lo + int(last)
    return None


def tagged_windows(instances):
    """{half: (start, end)} of Smart Tagging 'Kick Off (Start)' phases, video frames."""
    out = {}
    for inst in instances:
        if inst["code"] == "SET PIECES" and inst["tags"].get("type") == "kick off (start)":
            out[inst["tags"].get("half")] = (inst["start"], inst["end"])
    return out
