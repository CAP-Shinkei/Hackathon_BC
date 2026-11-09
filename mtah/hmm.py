"""Three-state hidden Markov model over H / A / D, decoded with Viterbi.

Adapted from Álvaro's v0.2 (handoff_alvaro/mtah-possession/src/layers/{emissions,decoder}.py).
His constants were set on the competition match; here every probability is counted on external
answer keys, and both tables are made team-symmetric so nothing about "home" or a particular club
is learned from the external games.

Observation per frame: the nearest-team code, H / A / NOBODY (ball or players not tracked).
"""

import numba
import numpy as np

from mtah.evidence import NOBODY
from mtah.labels import A, D, H, X

OBS_NONE = 2  # column index of NOBODY in the emission table


def obs_index(near):
    return np.where(near == NOBODY, OBS_NONE, near).astype(np.int8)


def count_tables(pairs, pseudo=1.0):
    """Team-symmetric log emission (3 states x 3 obs) and log transition (3x3) tables counted
    from [(gt_codes, near_codes)] on external games. Frames where gt is X are skipped, and no
    transition is counted across an X frame (so halves never connect)."""
    emis = np.full((3, 3), pseudo)
    trans = np.full((3, 3), pseudo)
    for gt, near in pairs:
        obs = obs_index(near)
        g = gt != X
        np.add.at(emis, (gt[g], obs[g]), 1)
        both = g[1:] & g[:-1]
        np.add.at(trans, (gt[:-1][both], gt[1:][both]), 1)
    # Swap H<->A in both state and observation, then average: a team-neutral model.
    perm = np.array([A, H, D])
    emis = emis + emis[perm][:, perm]
    trans = trans + trans[perm][:, perm]
    emis /= emis.sum(axis=1, keepdims=True)
    trans /= trans.sum(axis=1, keepdims=True)
    return np.log(emis), np.log(trans)


@numba.njit(cache=True)
def _viterbi(emis, log_t, log_p0):
    n, k = emis.shape
    delta = np.empty(k)
    psi = np.empty((n, k), dtype=np.int8)
    for s in range(k):
        delta[s] = log_p0[s] + emis[0, s]
    new = np.empty(k)
    for t in range(1, n):
        for s in range(k):
            best, arg = -np.inf, 0
            for r in range(k):
                v = delta[r] + log_t[r, s]
                if v > best:
                    best, arg = v, r
            new[s] = best + emis[t, s]
            psi[t, s] = arg
        delta[:] = new
    path = np.empty(n, dtype=np.int8)
    path[n - 1] = np.argmax(delta)
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def decode(near, blocks, log_emis, log_trans, temperature=1.0, n_frames=None):
    """Most likely H/A/D path inside each graded block, X elsewhere.

    temperature scales the emission log-probabilities. Consecutive observations are strongly
    correlated (the nearest player flickers in bursts), so treating them as independent makes the
    evidence far too confident; a temperature below 1 corrects for that. It is fitted on external
    games, never on the competition match.
    """
    n_frames = len(near) if n_frames is None else n_frames
    out = np.full(n_frames, X, dtype=np.int8)
    obs = obs_index(near)
    log_p0 = np.log(np.array([0.45, 0.45, 0.10]))  # each block starts at a kick-off
    for a, b in blocks:
        e = temperature * log_emis[:, obs[a:b]].T
        out[a:b] = _viterbi(np.ascontiguousarray(e), log_trans, log_p0)
    return out


@numba.njit(cache=True)
def _merge_runs(vals, lens, min_frames):
    """Run-length merge: repeatedly give the shortest run (first on ties) the value of its longer
    neighbour (the left one on ties) and join equal neighbours, until every run is long enough."""
    n = len(vals)
    while n > 1:
        i = 0
        for j in range(1, n):
            if lens[j] < lens[i]:
                i = j
        if lens[i] >= min_frames:
            break
        left = lens[i - 1] if i > 0 else -1
        right = lens[i + 1] if i + 1 < n else -1
        vals[i] = vals[i - 1] if (left >= right and i > 0) else vals[i + 1]
        m = 0
        for j in range(n):
            if m > 0 and vals[j] == vals[m - 1]:
                lens[m - 1] += lens[j]
            else:
                vals[m] = vals[j]
                lens[m] = lens[j]
                m += 1
        n = m
    return vals[:n], lens[:n]


def merge_short(path, blocks, min_frames):
    """Inside each block, absorb spells shorter than min_frames into the longer neighbour,
    shortest spells first (Álvaro's merge rule, made order-independent)."""
    out = path.copy()
    if min_frames <= 1:
        return out
    for a, b in blocks:
        seg = out[a:b]
        br = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], br))
        stops = np.concatenate((br, [len(seg)]))
        vals, lens = _merge_runs(seg[starts].copy(), (stops - starts).astype(np.int64),
                                 min_frames)
        out[a:b] = np.repeat(vals, lens)
    return out
