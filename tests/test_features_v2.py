import dataclasses
import time

import numpy as np
import pytest

from mtah import features
from mtah.features_v2 import featurise
from mtah.features import swap_index
from mtah.schema import FPS, Match, swap_teams

N_SLOTS = 11
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
BALL_MISSING_SHARE = 0.41
PLAYER_MISSING_SHARE = 0.3
SPEED_FRAMES = 30_000
SPEED_BUDGET_S = 4.0


def _walk(rng, n_frames, n_tracks):
    steps = rng.normal(0.0, 0.2, size=(n_frames, n_tracks, 2))
    start = rng.uniform([10.0, 10.0], [95.0, 58.0], size=(n_tracks, 2))
    path = start + np.cumsum(steps, axis=0)
    return np.clip(path, [0.0, 0.0], [PITCH_LENGTH, PITCH_WIDTH])


def _synthetic_match(n_frames=900, seed=0, two_blocks=True):
    rng = np.random.default_rng(seed)
    home = _walk(rng, n_frames, N_SLOTS)
    away = _walk(rng, n_frames, N_SLOTS)
    ball = _walk(rng, n_frames, 1)[:, 0]
    home[rng.random((n_frames, N_SLOTS)) < PLAYER_MISSING_SHARE] = np.nan
    away[rng.random((n_frames, N_SLOTS)) < PLAYER_MISSING_SHARE] = np.nan
    ball[rng.random(n_frames) < BALL_MISSING_SHARE] = np.nan
    half = n_frames // 2
    blocks = ((0, half), (half, n_frames)) if two_blocks else ((0, n_frames),)
    return Match(
        "synthetic",
        home.astype(np.float32),
        away.astype(np.float32),
        ball.astype(np.float32),
        blocks,
    )


def _static_match(home_xy, away_xy, ball_xy, n_frames):
    """Players and ball held still at the given positions, one block covering every frame."""
    home = np.tile(np.asarray(home_xy, dtype=np.float32), (n_frames, 1, 1))
    away = np.tile(np.asarray(away_xy, dtype=np.float32), (n_frames, 1, 1))
    ball = np.tile(np.asarray(ball_xy, dtype=np.float32), (n_frames, 1))
    return Match("static", home, away, ball, ((0, n_frames),))


def _column(X, names, name):
    return X[:, names.index(name)]


def test_featurise_starts_with_every_base_column_unchanged():
    # Arrange
    match = _synthetic_match()
    base_X, base_names = features.featurise(match)

    # Act
    X, names = featurise(match)

    # Assert
    assert names[:len(base_names)] == base_names
    assert np.array_equal(X[:, :len(base_names)], base_X)


def test_output_is_finite_float32_with_unique_names_and_matching_width():
    # Arrange
    match = _synthetic_match()

    # Act
    X, names = featurise(match)

    # Assert
    assert X.shape == (match.n_frames, len(names))
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    assert len(set(names)) == len(names)
    assert len(names) > len(features.featurise(match)[1])


def test_swap_identity_holds_on_match_with_missing_players_and_ball_gaps():
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
    shuffled = dataclasses.replace(
        match,
        home_xy=match.home_xy[:, rng.permutation(N_SLOTS)],
        away_xy=match.away_xy[:, rng.permutation(N_SLOTS)],
    )

    # Act
    X, _ = featurise(match)
    X_shuffled, _ = featurise(shuffled)

    # Assert
    assert np.allclose(X, X_shuffled)


def test_changing_the_second_block_leaves_first_block_rows_unchanged():
    # Arrange
    match = _synthetic_match()
    half = match.n_frames // 2
    rng = np.random.default_rng(2)
    altered_ball = match.ball_xy.copy()
    altered_ball[half:] = rng.uniform(0.0, 50.0, size=altered_ball[half:].shape)
    altered_home = match.home_xy.copy()
    altered_home[half:] = 0.0
    altered = dataclasses.replace(match, ball_xy=altered_ball, home_xy=altered_home)

    # Act
    X, _ = featurise(match)
    X_altered, _ = featurise(altered)

    # Assert
    assert np.array_equal(X[:half], X_altered[:half])


def test_ball_anchors_hold_last_and_next_seen_positions_with_implied_speed():
    # Arrange: seen at frames 0 and 4 only, so frames 1-3 sit inside one gap of 4 frames
    n_frames = 6
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball[0] = (10.0, 20.0)
    ball[4] = (14.0, 23.0)
    home = np.full((n_frames, 2, 2), 30.0, dtype=np.float32)
    away = np.full((n_frames, 2, 2), 70.0, dtype=np.float32)
    match = Match("anchors", home, away, ball, ((0, n_frames),))

    # Act
    X, names = featurise(match)

    # Assert
    assert np.allclose(_column(X, names, "ball_last_seen_x")[1:4], 10.0)
    assert np.allclose(_column(X, names, "ball_next_seen_y")[1:4], 23.0)
    distance = 5.0  # 3-4-5 triangle between (10, 20) and (14, 23)
    assert np.allclose(_column(X, names, "ball_seen_gap_dist")[1:4], distance)
    bracket_s = 4 / FPS
    assert np.allclose(_column(X, names, "ball_seen_gap_speed")[1:4], distance / bracket_s)


def test_team_extent_and_edge_ranks_match_hand_computed_layout():
    # Arrange: four home players at x = 10, 20, 30, 40 and no away players
    n_frames = 30
    home_xy = [(10.0, 5.0), (20.0, 15.0), (30.0, 25.0), (40.0, 35.0)]
    away_xy = [(np.nan, np.nan)] * 2
    match = _static_match(home_xy, away_xy, (50.0, 34.0), n_frames)

    # Act
    X, names = featurise(match)

    # Assert
    assert np.allclose(_column(X, names, "home_rear3_x")[5], (10 + 20 + 30) / 3)
    assert np.allclose(_column(X, names, "home_front3_x")[5], (20 + 30 + 40) / 3)
    assert np.allclose(_column(X, names, "home_centroid_dx")[5], 25.0 - 52.5)
    # p10 of [10, 20, 30, 40] is 13 and p90 is 37 under linear interpolation
    assert np.allclose(_column(X, names, "home_length_p10_90")[5], 24.0)
    assert np.allclose(_column(X, names, "away_rear3_x")[5], PITCH_LENGTH / 2)
    assert np.allclose(_column(X, names, "away_length_p10_90")[5], 0.0)


def test_stationary_players_are_all_still_and_fast_count_detects_a_runner():
    # Arrange: home holds still; away player 0 runs along x at 10 m/s, the rest are still
    n_frames = 200
    home_xy = [(20.0, 20.0), (30.0, 30.0)]
    away_xy = [(60.0, 30.0), (70.0, 40.0)]
    match = _static_match(home_xy, away_xy, (50.0, 34.0), n_frames)
    away = match.away_xy.astype(np.float64)
    away[:, 0, 0] = 60.0 + np.arange(n_frames) * (10.0 / FPS)
    match = dataclasses.replace(match, away_xy=away.astype(np.float32))

    # Act
    X, names = featurise(match)

    # Assert
    interior = slice(10, n_frames - 10)
    assert np.allclose(_column(X, names, "still_share_centred_2s")[interior], 0.75)
    assert np.allclose(_column(X, names, "away_n_fast")[interior], 1.0)
    assert np.allclose(_column(X, names, "home_n_fast")[interior], 0.0)


def test_past_long_window_is_the_mean_of_the_six_frames_before_excluding_current():
    # Arrange: a short synthetic match so a brute-force reference is cheap
    match = _synthetic_match(n_frames=600, seed=3, two_blocks=False)
    X, names = featurise(match)
    seen = _column(X, names, "ball_seen")
    window = 6 * FPS

    # Act
    reference = np.array([
        seen[max(0, t - window):t].mean() if t > 0 else seen[0] for t in range(len(seen))
    ])

    # Assert
    column = _column(X, names, "ball_seen_long_past_6s")
    assert np.allclose(column[window + 1:], reference[window + 1:], atol=1e-5)


def test_near_ball_past_minimum_matches_brute_force_within_block():
    # Arrange
    match = _synthetic_match(n_frames=400, seed=4, two_blocks=False)
    X, names = featurise(match)
    dist = _column(X, names, "home_dist_1st").astype(np.float64)
    window = FPS

    # Act
    column = _column(X, names, "home_min_dist_past_1s")
    future = _column(X, names, "home_min_dist_future_1s")

    # Assert
    for t in range(window, len(dist) - window):
        assert np.isclose(column[t], dist[t - window:t].min(), atol=1e-4)
        assert np.isclose(future[t], dist[t + 1:t + window + 1].min(), atol=1e-4)


def test_team_without_slots_is_rejected_with_a_clear_message():
    # Arrange
    match = _synthetic_match(n_frames=100)
    empty = dataclasses.replace(match, home_xy=np.zeros((100, 0, 2), dtype=np.float32))

    # Act / Assert
    with pytest.raises(ValueError, match="no player slots"):
        featurise(empty)


def test_featurise_runs_fast_on_30000_frames():
    # Arrange
    match = _synthetic_match(n_frames=SPEED_FRAMES, seed=5)

    # Act
    started = time.perf_counter()
    featurise(match)
    elapsed = time.perf_counter() - started

    # Assert
    assert elapsed < SPEED_BUDGET_S
