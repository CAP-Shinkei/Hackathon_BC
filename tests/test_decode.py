"""Tests for explicit-duration decoding, boundary refinement, segment cleanup and rule helpers.

All data is synthetic: label runs are generated here, and per-frame probabilities are noisy
versions of those runs.
"""
import time

import numpy as np
import pytest

from mtah.decode import (
    _duration_counts,
    count_durations,
    count_transitions,
    decode_blocks,
    hsmm_decode,
    merge_short,
    refine_boundaries,
)
from mtah.rules import dead_before, drop_far_outside
from mtah.schema import A, D, H, Match
from mtah.scoring import score

STEP = 5
FPS = 25
DEFAULT_MAX_S = 90.0
RECOVERY_MIN_SPELL_F1 = 0.9
BOUNDARY_TOLERANCE_FRAMES = 5
# Above ~0.25 flip noise the maximum-likelihood cut can drift beyond 5 frames, which refinement
# (limited to +-radius) cannot correct; 0.2 keeps the 5-frame tolerance a fair test.
RECOVERY_FLIP_RATE = 0.2


def labels_from_segments(segments):
    return np.concatenate([np.full(length, label, dtype=np.int8) for label, length in segments])


def random_segments(seed, n_segments, low=60, high=300):
    rng = np.random.default_rng(seed)
    segments = []
    previous = None
    for _ in range(n_segments):
        choices = [state for state in (H, A, D) if state != previous]
        label = int(rng.choice(choices))
        segments.append((label, int(rng.integers(low, high + 1))))
        previous = label
    return segments


def noisy_proba(labels, seed, flip_rate=0.45):
    """True label gets 0.6 on unflipped frames; flipped frames put 0.6 on a wrong label."""
    rng = np.random.default_rng(seed)
    n_frames = len(labels)
    truth = labels.astype(int)
    flipped = rng.random(n_frames) < flip_rate
    wrong = (truth + 1 + rng.integers(0, 2, n_frames)) % 3
    chosen = np.where(flipped, wrong, truth)
    proba = np.full((n_frames, 3), 0.2)
    proba[np.arange(n_frames), chosen] = 0.6
    return proba


def change_points(codes):
    return np.flatnonzero(np.diff(codes) != 0) + 1


def uniform_priors(d_max):
    dur = np.full((3, d_max), -np.log(d_max))
    trans = np.full((3, 3), np.log(0.5))
    np.fill_diagonal(trans, -np.inf)
    return dur, trans


def make_match(ball_xy):
    n_frames = len(ball_xy)
    players = np.zeros((n_frames, 1, 2), dtype=np.float32)
    return Match(
        game_id="synthetic",
        home_xy=players,
        away_xy=players.copy(),
        ball_xy=np.asarray(ball_xy, dtype=np.float32),
        blocks=((0, n_frames),),
    )


def test_duration_counts_index_runs_in_step_units():
    # Arrange: H run of 10 frames (2 units), H run of 5 frames (1 unit), A run of 5 frames.
    labels = labels_from_segments([(H, 10), (A, 5), (H, 5)])

    # Act
    counts = _duration_counts([labels], [((0, len(labels)),)], DEFAULT_MAX_S, STEP)

    # Assert
    assert counts[H, 1] == 1
    assert counts[H, 0] == 1
    assert counts[A, 0] == 1
    assert counts[D].sum() == 0


def test_duration_counts_clip_long_runs_into_last_bin():
    # Arrange: max_s=10 gives 50 bins; a 500-frame run is 100 units and must land in bin 49.
    labels = labels_from_segments([(H, 500)])

    # Act
    counts = _duration_counts([labels], [((0, 500),)], 10.0, STEP)

    # Assert
    assert counts.shape == (3, 50)
    assert counts[H, 49] == 1


def test_duration_counts_ignore_frames_outside_blocks():
    # Arrange: the run spans frames 0..20 but only frames 0..10 are inside the block.
    labels = labels_from_segments([(A, 20)])

    # Act
    counts = _duration_counts([labels], [((0, 10),)], DEFAULT_MAX_S, STEP)

    # Assert
    assert counts[A, 1] == 1
    assert counts.sum() == 1


def test_count_durations_returns_normalised_log_pmf_per_state():
    # Arrange
    labels = labels_from_segments([(H, 10), (A, 5), (D, 30)])

    # Act
    log_pmf = count_durations([labels], [((0, len(labels)),)], DEFAULT_MAX_S, STEP)

    # Assert
    assert log_pmf.shape == (3, 450)
    assert np.all(np.isfinite(log_pmf))
    np.testing.assert_allclose(np.exp(log_pmf).sum(axis=1), 1.0)


def test_count_transitions_counts_only_changes_between_segments():
    # Arrange: segments H, A, D, H -> transitions H->A, A->D, D->H once each.
    labels = labels_from_segments([(H, 2), (A, 2), (D, 1), (H, 1)])

    # Act
    log_trans = count_transitions([labels], [((0, len(labels)),)])

    # Assert: row H has HA=2, HD=1 after add-one smoothing.
    np.testing.assert_allclose(np.exp(log_trans[H, A]), 2 / 3)
    np.testing.assert_allclose(np.exp(log_trans[H, D]), 1 / 3)
    np.testing.assert_allclose(np.exp(log_trans[A, D]), 2 / 3)
    np.testing.assert_allclose(np.exp(log_trans[D, H]), 2 / 3)
    assert np.isneginf(log_trans[H, H])
    assert np.isneginf(log_trans[A, A])


def test_hsmm_refine_recovers_segments_and_beats_per_frame_argmax():
    # Arrange: priors from an independent training sequence, noisy probabilities for the truth.
    train = labels_from_segments(random_segments(seed=10, n_segments=40))
    blocks = ((0, len(train)),)
    dur = count_durations([train], [blocks], DEFAULT_MAX_S, STEP)
    trans = count_transitions([train], [blocks])
    truth = labels_from_segments(random_segments(seed=2, n_segments=12))
    proba = noisy_proba(truth, seed=3, flip_rate=RECOVERY_FLIP_RATE)

    # Act
    coarse = hsmm_decode(proba, dur, trans, step=STEP)
    refined = refine_boundaries(coarse, proba, radius=BOUNDARY_TOLERANCE_FRAMES)

    # Assert
    true_cuts = change_points(truth)
    pred_cuts = change_points(refined)
    assert len(pred_cuts) == len(true_cuts)
    for cut in true_cuts:
        assert np.min(np.abs(pred_cuts - cut)) <= BOUNDARY_TOLERANCE_FRAMES
    decoded_f1 = score(truth, refined)["spell_f1"]
    argmax_f1 = score(truth, proba.argmax(axis=1))["spell_f1"]
    assert decoded_f1 > RECOVERY_MIN_SPELL_F1
    assert argmax_f1 < decoded_f1 - 0.3


def test_hsmm_decodes_run_longer_than_max_s_by_chaining_last_bin():
    # Arrange: 3000 frames (120 s) of H with max_s=10 means each segment is at most 250 frames,
    # so the run must be chained through same-state continuations from the last bin.
    run_frames = 3000
    truth = labels_from_segments([(H, run_frames), (A, 200)])
    proba = noisy_proba(truth, seed=4, flip_rate=0.0)
    dur, trans = uniform_priors(d_max=50)

    # Act
    decoded = hsmm_decode(proba, dur, trans, step=STEP)

    # Assert
    assert len(decoded) == len(truth)
    assert np.all(decoded[:run_frames] == H)


def test_hsmm_rejects_probabilities_without_three_columns():
    # Arrange
    dur, trans = uniform_priors(d_max=10)
    bad_proba = np.full((20, 2), 0.5)

    # Act / Assert
    with pytest.raises(ValueError):
        hsmm_decode(bad_proba, dur, trans, step=STEP)


def test_hsmm_decode_handles_80000_frames_under_a_minute():
    # Arrange
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), size=80_000)
    dur, trans = uniform_priors(d_max=450)

    # Act
    started = time.perf_counter()
    decoded = hsmm_decode(proba, dur, trans, step=STEP)
    elapsed = time.perf_counter() - started

    # Assert
    assert len(decoded) == 80_000
    assert elapsed < 60.0


def test_merge_short_removes_five_frame_blip_between_long_segments():
    # Arrange
    codes = labels_from_segments([(H, 100), (A, 5), (H, 100)])

    # Act
    merged = merge_short(codes, min_frames=10, blocks=((0, len(codes)),))

    # Assert
    assert np.all(merged == H)


def test_merge_short_keeps_segments_at_or_above_threshold():
    # Arrange
    codes = labels_from_segments([(H, 100), (A, 50), (H, 100)])

    # Act
    merged = merge_short(codes, min_frames=50, blocks=((0, len(codes)),))

    # Assert
    np.testing.assert_array_equal(merged, codes)


def test_merge_short_never_crosses_block_boundaries():
    # Arrange: block two opens with a 3-frame A blip followed by D. The blip has no previous
    # segment inside its block, so it must take D (the next segment), never H from block one.
    codes = labels_from_segments([(H, 50), (A, 3), (D, 50)])
    blocks = ((0, 50), (50, 103))

    # Act
    merged = merge_short(codes, min_frames=10, blocks=blocks)

    # Assert
    assert np.all(merged[0:50] == H)
    assert np.all(merged[50:103] == D)


def test_decode_blocks_keeps_labels_inside_blocks_and_dead_time_as_d():
    # Arrange: block one is clearly H, block two clearly A, with a dead gap between.
    block_one = labels_from_segments([(H, 300)])
    gap = labels_from_segments([(D, 100)])
    block_two = labels_from_segments([(A, 300)])
    truth = np.concatenate([block_one, gap, block_two])
    proba = noisy_proba(truth, seed=5, flip_rate=0.0)
    blocks = ((0, 300), (400, 700))
    dur, trans = uniform_priors(d_max=450)

    # Act
    decoded = decode_blocks(proba, blocks, dur, trans, step=STEP, min_frames=0)

    # Assert
    assert len(decoded) == 700
    assert np.all(decoded[0:300] == H)
    assert np.all(decoded[300:400] == D)
    assert np.all(decoded[400:700] == A)


def test_decode_blocks_produces_no_segment_spanning_a_block_gap():
    # Arrange: identical state on both sides of the gap; the gap must still be D.
    truth = labels_from_segments([(H, 200)])
    proba = noisy_proba(truth, seed=6, flip_rate=0.0)
    blocks = ((0, 100), (150, 200))
    dur, trans = uniform_priors(d_max=450)

    # Act
    decoded = decode_blocks(proba, blocks, dur, trans, step=STEP)

    # Assert
    assert np.all(decoded[100:150] == D)
    assert np.all(decoded[0:100] == H)
    assert np.all(decoded[150:200] == H)


def test_drop_far_outside_nans_only_ball_beyond_margin_and_keeps_input():
    # Arrange: pitch is 105 x 68; margin 10 m, so limits are x in [-10, 115], y in [-10, 78].
    ball = np.array([[0.0, 0.0], [-11.0, 30.0], [-10.0, 30.0], [115.0, 30.0],
                     [115.5, 30.0], [50.0, 78.5], [50.0, 77.0]], dtype=np.float32)
    match = make_match(ball)
    original = match.ball_xy.copy()

    # Act
    result = drop_far_outside(match, margin_m=10.0)

    # Assert
    expected_missing = np.array([False, True, False, False, True, True, False])
    np.testing.assert_array_equal(np.isnan(result.ball_xy[:, 0]), expected_missing)
    np.testing.assert_array_equal(np.isnan(result.ball_xy[:, 1]), expected_missing)
    np.testing.assert_array_equal(match.ball_xy, original)
    assert result is not match


def test_dead_before_sets_span_to_d_without_mutating_input():
    # Arrange
    codes = labels_from_segments([(H, 10), (A, 10)])
    original = codes.copy()

    # Act
    result = dead_before(codes, start=5, stop=15)

    # Assert
    np.testing.assert_array_equal(codes, original)
    assert np.all(result[5:15] == D)
    assert np.all(result[:5] == H)
    assert np.all(result[15:] == A)


def test_dead_before_rejects_span_outside_the_sequence():
    # Arrange
    codes = labels_from_segments([(H, 10)])

    # Act / Assert
    with pytest.raises(ValueError):
        dead_before(codes, start=5, stop=50)


def test_decode_refuses_non_finite_probabilities():
    proba = np.tile(np.array([0.1, 0.8, 0.1]), (10, 1))
    proba[4] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        hsmm_decode(proba, np.zeros((3, 4)), np.zeros((3, 3)))
