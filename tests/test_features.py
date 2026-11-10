import dataclasses
import time

import numpy as np

from mtah.ballfill import fill_linear
from mtah.features import featurise, swap_index
from mtah.schema import FPS, Match, swap_teams

N_SLOTS = 11
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
BALL_MISSING_SHARE = 0.41
PLAYER_MISSING_SHARE = 0.3
MISSING_DIST_M = 60.0


def _walk(rng, n_frames, n_tracks):
    steps = rng.normal(0.0, 0.2, size=(n_frames, n_tracks, 2))
    start = rng.uniform([10.0, 10.0], [95.0, 58.0], size=(n_tracks, 2))
    path = start + np.cumsum(steps, axis=0)
    return np.clip(path, [0.0, 0.0], [PITCH_LENGTH, PITCH_WIDTH])


def _synthetic_match(n_frames=2000, seed=0):
    rng = np.random.default_rng(seed)
    home = _walk(rng, n_frames, N_SLOTS)
    away = _walk(rng, n_frames, N_SLOTS)
    ball = _walk(rng, n_frames, 1)[:, 0]
    home[rng.random((n_frames, N_SLOTS)) < PLAYER_MISSING_SHARE] = np.nan
    away[rng.random((n_frames, N_SLOTS)) < PLAYER_MISSING_SHARE] = np.nan
    ball[rng.random(n_frames) < BALL_MISSING_SHARE] = np.nan
    half = n_frames // 2
    return Match(
        "synthetic",
        home.astype(np.float32),
        away.astype(np.float32),
        ball.astype(np.float32),
        ((0, half), (half, n_frames)),
    )


def test_fill_linear_interpolates_gap_exactly():
    # Arrange
    ball = np.full((5, 2), np.nan)
    ball[0] = (0.0, 0.0)
    ball[4] = (4.0, 8.0)

    # Act
    filled, seen, gap, since, until = fill_linear(ball, ((0, 5),))

    # Assert
    assert np.allclose(filled[1:4], [[1, 2], [2, 4], [3, 6]])
    assert seen.tolist() == [True, False, False, False, True]
    assert gap[0] == 0.0 and np.allclose(gap[1:4], 3 / FPS)
    assert np.allclose(since[1:4], np.array([1, 2, 3]) / FPS)
    assert np.allclose(until[1:4], np.array([3, 2, 1]) / FPS)


def test_fill_linear_never_bridges_a_block_boundary():
    # Arrange: frame 2 is the last seen frame of block 1, frame 4 the first of block 2
    ball = np.full((6, 2), np.nan)
    ball[2] = (2.0, 2.0)
    ball[4] = (40.0, 40.0)

    # Act
    filled, _, _, _, _ = fill_linear(ball, ((0, 3), (3, 6)))

    # Assert: interpolating across the boundary would give (21, 21) at frame 3
    assert np.allclose(filled[3], (40.0, 40.0))
    assert np.allclose(filled[0], (2.0, 2.0))


def test_fill_linear_output_is_finite_and_input_untouched():
    # Arrange
    ball = np.full((8, 2), np.nan)
    ball[3] = (10.0, 20.0)
    original = ball.copy()

    # Act
    filled, _, _, _, _ = fill_linear(ball, ((0, 8),))

    # Assert
    assert np.isfinite(filled).all()
    assert np.array_equal(ball, original, equal_nan=True)


def test_fill_linear_rejects_wrong_shape_and_overlapping_blocks():
    # Arrange
    good = np.zeros((6, 2))

    # Act / Assert
    try:
        fill_linear(np.zeros((6, 3)), ((0, 6),))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong shape")
    try:
        fill_linear(good, ((0, 4), (3, 6)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for overlapping blocks")


def test_featurise_is_finite_with_unique_names_and_matching_width():
    # Arrange
    match = _synthetic_match()

    # Act
    X, names = featurise(match)

    # Assert
    assert X.shape == (match.n_frames, len(names))
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    assert len(set(names)) == len(names)


def test_swap_identity_holds_exactly_on_synthetic_match():
    # Arrange
    match = _synthetic_match()
    X, names = featurise(match)

    # Act
    X_swapped, names_swapped = featurise(swap_teams(match))

    # Assert
    assert names_swapped == names
    assert np.allclose(X_swapped, X[:, swap_index(names)])


def test_permuting_players_inside_a_team_leaves_features_unchanged():
    # Arrange
    match = _synthetic_match()
    rng = np.random.default_rng(1)
    home_perm = rng.permutation(N_SLOTS)
    away_perm = rng.permutation(N_SLOTS)
    shuffled = dataclasses.replace(
        match,
        home_xy=match.home_xy[:, home_perm],
        away_xy=match.away_xy[:, away_perm],
    )

    # Act
    X, _ = featurise(match)
    X_shuffled, _ = featurise(shuffled)

    # Assert
    assert np.allclose(X, X_shuffled, equal_nan=False)


def test_team_without_visible_players_reports_large_distance():
    # Arrange
    match = _synthetic_match(n_frames=500)
    match = dataclasses.replace(match, away_xy=np.full_like(match.away_xy, np.nan))

    # Act
    X, names = featurise(match)

    # Assert
    column = X[:, names.index("away_dist_1st")]
    assert np.allclose(column, MISSING_DIST_M)
    assert np.all(X[:, names.index("away_n_visible")] == 0)


def test_changing_one_half_leaves_the_other_half_features_unchanged():
    # Arrange
    match = _synthetic_match()
    half = match.n_frames // 2
    rng = np.random.default_rng(2)
    altered_ball = match.ball_xy.copy()
    altered_ball[half:] = rng.uniform(0.0, 50.0, size=altered_ball[half:].shape)
    altered = dataclasses.replace(match, ball_xy=altered_ball)

    # Act
    X, _ = featurise(match)
    X_altered, _ = featurise(altered)

    # Assert
    assert np.array_equal(X[:half], X_altered[:half])


def test_featurise_runs_under_five_seconds_for_20000_frames():
    # Arrange
    match = _synthetic_match(n_frames=20000)

    # Act
    started = time.perf_counter()
    featurise(match)
    elapsed = time.perf_counter() - started

    # Assert
    assert elapsed < 5.0
