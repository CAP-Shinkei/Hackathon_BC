"""Explicit-duration (semi-Markov) decoding of per-frame possession probabilities.

Per block: a coarse Viterbi over step-frame units with duration and transition priors, then
boundary refinement at full frame rate, then removal of short H/A blips. Label codes follow
mtah.schema (H=0, A=1, D=2).
"""
import math

import numpy as np

from mtah.schema import A, D, FPS, H

STATES = (H, A, D)
N_STATES = len(STATES)
PROBA_FLOOR = 1e-6
SMOOTH_KERNEL = np.array([0.25, 0.5, 0.25])


def _check_timing(temperature, step):
    if temperature <= 0:
        raise ValueError("temperature must be positive, got {}".format(temperature))
    if int(step) != step or step < 1:
        raise ValueError("step must be a positive integer, got {}".format(step))


def _check_pairs(labels_list, blocks_list):
    if len(labels_list) != len(blocks_list):
        raise ValueError("labels_list and blocks_list must have the same length")


def _n_duration_bins(max_s, step):
    _check_timing(1.0, step)
    if max_s <= 0:
        raise ValueError("max_s must be positive, got {}".format(max_s))
    return int(max_s * FPS / step)


def _log_proba(proba):
    values = np.asarray(proba, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_STATES:
        raise ValueError("proba must have shape [T, 3], got {}".format(values.shape))
    if len(values) == 0:
        raise ValueError("proba must contain at least one frame")
    if not np.isfinite(values).all():
        raise ValueError("proba holds {} non-finite values".format(
            int((~np.isfinite(values)).sum())))
    return np.log(np.clip(values, PROBA_FLOOR, None))


def _segments(codes):
    """Starts, lengths and labels of the runs of equal values in a 1-D array."""
    codes = np.asarray(codes)
    if codes.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), codes[:0]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(codes)) + 1))
    lengths = np.diff(np.concatenate((starts, [codes.size])))
    return starts, lengths, codes[starts]


def _check_block(block, n_frames):
    start, stop = block
    if not 0 <= start < stop <= n_frames:
        raise ValueError("bad block {} for {} frames".format(block, n_frames))


def _duration_counts(labels_list, blocks_list, max_s, step):
    _check_pairs(labels_list, blocks_list)
    d_max = _n_duration_bins(max_s, step)
    counts = np.zeros((N_STATES, d_max))
    for labels, blocks in zip(labels_list, blocks_list, strict=True):
        labels = np.asarray(labels)
        for start, stop in blocks:
            _, lengths, runs = _segments(labels[start:stop])
            keep = np.isin(runs, STATES)
            units = np.ceil(lengths[keep] / step).astype(int)
            bins = np.minimum(units, d_max) - 1
            np.add.at(counts, (runs[keep].astype(int), bins), 1)
    return counts


def _smooth_rows(values):
    """Light moving average along each row, edge-padded so the row length is kept."""
    half = len(SMOOTH_KERNEL) // 2
    return np.stack([
        np.convolve(np.pad(row, half, mode="edge"), SMOOTH_KERNEL, mode="valid")
        for row in values
    ])


def count_durations(labels_list, blocks_list, max_s=90.0, step=5):
    """Log-pmf [3, Dmax] of segment durations per state, in units of `step` frames."""
    counts = _duration_counts(labels_list, blocks_list, max_s, step)
    smoothed = _smooth_rows(counts + 1.0)
    return np.log(smoothed / smoothed.sum(axis=1, keepdims=True))


def _transition_counts(labels_list, blocks_list):
    _check_pairs(labels_list, blocks_list)
    counts = np.zeros((N_STATES, N_STATES))
    for labels, blocks in zip(labels_list, blocks_list, strict=True):
        labels = np.asarray(labels)
        for start, stop in blocks:
            _, _, runs = _segments(labels[start:stop])
            if runs.size < 2:
                continue
            before, after = runs[:-1], runs[1:]
            valid = np.isin(before, STATES) & np.isin(after, STATES) & (before != after)
            np.add.at(counts, (before[valid].astype(int), after[valid].astype(int)), 1)
    return counts


def count_transitions(labels_list, blocks_list):
    """Log transition probabilities between different consecutive segments; diagonal -inf."""
    smoothed = _transition_counts(labels_list, blocks_list) + 1.0
    np.fill_diagonal(smoothed, 0.0)
    with np.errstate(divide="ignore"):
        log_trans = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
    np.fill_diagonal(log_trans, -np.inf)
    return log_trans


def _check_priors(dur_logpmf, trans_log):
    dur = np.asarray(dur_logpmf, dtype=np.float64)
    trans = np.asarray(trans_log, dtype=np.float64)
    if dur.ndim != 2 or dur.shape[0] != N_STATES:
        raise ValueError("dur_logpmf must have shape [3, Dmax], got {}".format(dur.shape))
    if trans.shape != (N_STATES, N_STATES):
        raise ValueError("trans_log must have shape [3, 3], got {}".format(trans.shape))
    return dur, trans


def _unit_scores(logp, step, temperature):
    """Sum of log-probabilities over each step-frame unit, scaled by 1/temperature.

    A sum (not a mean) keeps the emission on the same per-frame evidence scale as the duration
    and transition priors; a mean would weigh each unit by 1/step and let the priors merge
    genuine segments.
    """
    n_units = math.ceil(len(logp) / step)
    padded = np.pad(logp, ((0, n_units * step - len(logp)), (0, 0)), mode="edge")
    return padded.reshape(n_units, step, N_STATES).sum(axis=1) / temperature


def _candidates(t, cum, dur, pre_value):
    """Scores of ending a segment at unit t, for every duration d and state (rows = d)."""
    durations = np.arange(1, min(dur.shape[1], t) + 1)
    starts = t - durations
    scores = (cum[t] - cum[starts]) + dur[:, durations - 1].T + pre_value[starts]
    return scores, starts


def _best_predecessor(value, cont, trans):
    """Best way into a new segment of each state: switch from another state, or continue."""
    switch = value[:, None] + trans
    from_state = np.argmax(switch, axis=0)
    switch_best = switch[from_state, np.arange(N_STATES)]
    continues = cont > switch_best
    score = np.maximum(switch_best, cont)
    state = np.where(continues, np.arange(N_STATES), from_state)
    return score, state, continues


def _backtrack(n_units, value, best_start, pre_state, pre_cont, d_max):
    t = n_units
    state = int(np.argmax(value[n_units]))
    continued = False
    labels = np.empty(n_units, dtype=np.int8)
    while True:
        start = t - d_max if continued else int(best_start[t, state])
        labels[start:t] = STATES[state]
        if start == 0:
            return labels
        state, continued, t = int(pre_state[start, state]), bool(pre_cont[start, state]), start


def hsmm_decode(proba, dur_logpmf, trans_log, temperature=1.0, step=5):
    """Explicit-duration Viterbi on step-frame units; returns int8 labels of length T."""
    logp = _log_proba(proba)
    dur, trans = _check_priors(dur_logpmf, trans_log)
    _check_timing(temperature, step)
    n_frames = len(logp)
    units = _unit_scores(logp, step, temperature)
    n_units = len(units)
    d_max = dur.shape[1]
    cum = np.vstack([np.zeros((1, N_STATES)), np.cumsum(units, axis=0)])

    value = np.full((n_units + 1, N_STATES), -np.inf)
    best_start = np.zeros((n_units + 1, N_STATES), dtype=int)
    cont = np.full((n_units + 1, N_STATES), -np.inf)
    pre_value = np.zeros((n_units + 1, N_STATES))
    pre_state = np.zeros((n_units + 1, N_STATES), dtype=int)
    pre_cont = np.zeros((n_units + 1, N_STATES), dtype=bool)

    for t in range(1, n_units + 1):
        scores, starts = _candidates(t, cum, dur, pre_value)
        choice = np.argmax(scores, axis=0)
        value[t] = scores[choice, np.arange(N_STATES)]
        best_start[t] = starts[choice]
        if t >= d_max:
            cont[t] = scores[d_max - 1]
        pre_value[t], pre_state[t], pre_cont[t] = _best_predecessor(value[t], cont[t], trans)

    unit_labels = _backtrack(n_units, value, best_start, pre_state, pre_cont, d_max)
    return np.repeat(unit_labels, step)[:n_frames]


def _best_cut(left_start, cut, right_end, left_label, right_label, cum, radius):
    """Position in [cut-radius, cut+radius] maximising the two adjacent segments' log-prob."""
    low = max(left_start + 1, cut - radius)
    high = min(right_end - 1, cut + radius)
    positions = np.arange(low, high + 1)
    gain = cum[positions, left_label] - cum[positions, right_label]
    return int(positions[np.argmax(gain)])


def refine_boundaries(codes, proba, radius=5):
    """Move each boundary by up to `radius` frames to the best full-rate position."""
    logp = _log_proba(proba)
    codes = np.asarray(codes)
    if len(codes) != len(logp):
        raise ValueError("codes and proba differ in length")
    starts, _, runs = _segments(codes)
    bounds = np.concatenate((starts, [len(codes)])).astype(int)
    cum = np.vstack([np.zeros((1, N_STATES)), np.cumsum(logp, axis=0)])
    refined = bounds.copy()
    for j in range(1, len(starts)):
        refined[j] = _best_cut(refined[j - 1], bounds[j], bounds[j + 1],
                               int(runs[j - 1]), int(runs[j]), cum, radius)
    return np.repeat(runs.astype(np.int8), np.diff(refined))


def _merge_block(block_codes, min_frames):
    _, lengths, runs = _segments(block_codes)
    labels = runs.astype(np.int8)
    for j in range(len(labels)):
        if labels[j] == D or lengths[j] >= min_frames:
            continue
        if j > 0:
            labels[j] = labels[j - 1]
        elif len(labels) > 1:
            labels[j] = runs[j + 1]
    return np.repeat(labels, lengths)


def merge_short(codes, min_frames, blocks):
    """Give H/A segments shorter than min_frames the label of the previous segment, per block."""
    codes = np.asarray(codes)
    out = codes.astype(np.int8, copy=True)
    for start, stop in blocks:
        _check_block((start, stop), len(codes))
        out[start:stop] = _merge_block(codes[start:stop], min_frames)
    return out


def _decode_block(block_proba, dur, trans, temperature, step, min_frames):
    coarse = hsmm_decode(block_proba, dur, trans, temperature=temperature, step=step)
    refined = refine_boundaries(coarse, block_proba)
    return _merge_block(refined, min_frames)


def decode_blocks(proba, blocks, dur_logpmf, trans_log, temperature=1.0, step=5, min_frames=0):
    """Decode each live block separately; frames outside every block are D."""
    values = np.asarray(proba)
    dur, trans = _check_priors(dur_logpmf, trans_log)
    out = np.full(len(values), D, dtype=np.int8)
    for start, stop in blocks:
        _check_block((start, stop), len(values))
        out[start:stop] = _decode_block(values[start:stop], dur, trans,
                                        temperature, step, min_frames)
    return out
