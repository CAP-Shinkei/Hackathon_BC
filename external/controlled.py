"""Turn touch-level external possession labels into "controlled" possession labels.

The challenge counts a change of possession only when the other team gains *controlled*
possession; touches, deflections and won aerial duels without control do not count, and while
the ball is contested or loose possession stays with the last team (CHALLENGE.md, "Possession
rules"). Its first suggested proxy is "a player keeps the ball for a few seconds".

External labels (DFL per-frame BallPossession, Metrica event-derived keys) switch on every touch.
This applies the proxy: inside each live stretch, a team spell shorter than min_s seconds goes
back to the team that had the ball before it. The first spell after a restart is kept whatever its
length, because a restart always has a clear possessor (CHALLENGE.md, "Starting point").
min_s is read from the challenge text ("a few seconds"); it is not fitted to any ground truth.
"""

import numpy as np

H, A, D, X = 0, 1, 2, 3
FPS = 25


def controlled(gt, min_s):
    out = np.asarray(gt).copy()
    if min_s <= 0:
        return out
    min_frames = int(round(min_s * FPS))
    br = np.flatnonzero(out[1:] != out[:-1]) + 1
    starts = np.concatenate(([0], br))
    stops = np.concatenate((br, [len(out)]))
    prev_team = None
    for a, b in zip(starts, stops):
        c = out[a]
        if c not in (H, A):
            prev_team = None  # a stoppage or a half boundary: the next spell is a restart
            continue
        if prev_team is not None and c != prev_team and b - a < min_frames:
            out[a:b] = prev_team
        else:
            prev_team = c
    return out
