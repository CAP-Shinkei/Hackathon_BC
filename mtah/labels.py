"""Controlled possession: the challenge counts a change only once the new team controls the ball.

External answer keys flip on every touch. A spell shorter than `min_s` is therefore not a
possession: it goes back to the team that had the ball before it, or to the team that has it
right after when it follows a stoppage (a restart is always somebody's).
"""
import numpy as np

from mtah.schema import A, D, FPS, H


def _runs(codes):
    edges = np.flatnonzero(np.diff(codes)) + 1
    return np.concatenate(([0], edges)), np.concatenate((edges, [len(codes)]))


def _controlled_block(codes, min_frames):
    out = codes.copy()
    changed = True
    while changed:
        changed = False
        starts, stops = _runs(out)
        for i, (a, b) in enumerate(zip(starts, stops)):
            if out[a] not in (H, A) or b - a >= min_frames:
                continue
            before = out[a - 1] if i > 0 else D
            after = out[b] if b < len(out) else D
            target = before if before in (H, A) else after
            if target in (H, A) and target != out[a]:
                out[a:b] = target
                changed = True
                break
    return out


def controlled(labels, blocks, min_s):
    """A copy of `labels` where no H/A spell inside a block is shorter than `min_s` seconds
    unless it stands alone between two stoppages."""
    min_frames = int(round(min_s * FPS))
    out = np.asarray(labels, dtype=np.int8).copy()
    if min_frames <= 1:
        return out
    for start, stop in blocks:
        out[start:stop] = _controlled_block(out[start:stop], min_frames)
    return out
