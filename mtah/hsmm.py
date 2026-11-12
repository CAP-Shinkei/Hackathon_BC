"""Duration-aware decoding (hidden semi-Markov model) over H / A / D.

Adapted from Álvaro's v0.3 (handoff_alvaro/mtah-possession/src/layers/hsmm.py). A plain HMM
implies geometric spell lengths: very short spells are always the most likely, which is why v1/v2
need a merge step afterwards. An HSMM scores each spell by an explicit duration distribution.
Álvaro's gamma shapes were set by hand; here the duration distributions and the spell-to-spell
transitions are counted from external answer keys, team-symmetric.
"""

import numba
import numpy as np

from mtah.labels import A, D, H, X, runs

FPS = 25
DMAX = 3000      # 120 s; longer spells are decoded as several consecutive ones
SMOOTH = 12      # frames, Gaussian smoothing of the duration histograms
FLOOR = 1e-7


def count_durations(labels_list):
    """(log_dur (3, DMAX+1), log_trans (3, 3)) from external labels. Spells cut at DMAX."""
    hist = np.zeros((3, DMAX + 1))
    trans = np.ones((3, 3))
    for lab in labels_list:
        prev = None
        for c, a, b in runs(lab):
            if c == X:
                prev = None
                continue
            L = b - a
            while L > DMAX:
                hist[c, DMAX] += 1
                L -= DMAX
            hist[c, L] += 1
            if prev is not None:
                trans[prev, c] += 1
            prev = c
    hist[H] = hist[A] = 0.5 * (hist[H] + hist[A])
    perm = np.array([A, H, D])
    trans = trans + trans[perm][:, perm]
    k = np.exp(-0.5 * (np.arange(-4 * SMOOTH, 4 * SMOOTH + 1) / SMOOTH) ** 2)
    out = np.empty_like(hist)
    for s in range(3):
        sm = np.convolve(hist[s], k / k.sum(), mode="same")
        sm[0] = 0
        out[s] = sm / sm.sum() + FLOOR
        out[s] /= out[s].sum()
    trans /= trans.sum(axis=1, keepdims=True)
    return np.log(out), np.log(trans)


@numba.njit(cache=True)
def _decode(emis, log_dur, log_trans, log_p0, trans_adjust):
    # trans_adjust[t]: extra log-odds for a team-to-team switch whose new segment starts at
    # frame t (mtah.acquire's episode evidence). All zeros = the plain HSMM, bit-identical.
    n, k = emis.shape
    dmax = log_dur.shape[1] - 1
    cum = np.zeros((n + 1, k))
    for t in range(n):
        for s in range(k):
            cum[t + 1, s] = cum[t, s] + emis[t, s]
    # best[t, s]: best score of a segmentation of [0, t) whose last segment has state s;
    # enter[t, s]: best score of entering state s at t from another segment (and from whom).
    best = np.full((n + 1, k), -np.inf)
    enter = np.full((n + 1, k), -np.inf)
    enter_from = np.full((n + 1, k), -1, dtype=np.int8)
    back_d = np.zeros((n + 1, k), dtype=np.int32)
    back_s = np.full((n + 1, k), -1, dtype=np.int8)
    for s in range(k):
        enter[0, s] = log_p0[s]
    for t in range(1, n + 1):
        for s in range(k):
            top = -np.inf
            arg_d = 1
            for d in range(1, min(dmax, t) + 1):
                start = t - d
                v = enter[start, s] + cum[t, s] - cum[start, s] + log_dur[s, d]
                if v > top:
                    top = v
                    arg_d = d
            best[t, s] = top
            back_d[t, s] = arg_d
            back_s[t, s] = enter_from[t - arg_d, s]
        for s in range(k):
            for r in range(k):
                w = best[t, r] + log_trans[r, s]
                if r != s and r != D and s != D:
                    w += trans_adjust[t]
                if w > enter[t, s]:
                    enter[t, s] = w
                    enter_from[t, s] = r
    path = np.empty(n, dtype=np.int8)
    s = 0
    for r in range(1, k):
        if best[n, r] > best[n, s]:
            s = r
    t = n
    while t > 0:
        d = back_d[t, s]
        for i in range(t - d, t):
            path[i] = s
        p = back_s[t, s]
        t -= d
        s = p if p >= 0 else s
    return path


def decode(emis, blocks, log_dur, log_trans, n_frames, trans_adjust=None):
    """trans_adjust: optional (n_frames + 1,) per-boundary log-odds on team-to-team
    switches, indexed by absolute frame (mtah.acquire.transition_adjust). None decodes
    exactly as before."""
    if trans_adjust is None:
        trans_adjust = np.zeros(n_frames + 1)
    out = np.full(n_frames, X, dtype=np.int8)
    log_p0 = np.log(np.array([0.45, 0.45, 0.10]))
    lt = log_trans.copy()
    for a, b in blocks:
        out[a:b] = _decode(np.ascontiguousarray(emis[a:b]), log_dur, lt, log_p0,
                           np.ascontiguousarray(trans_adjust[a:b + 1]))
    return out
