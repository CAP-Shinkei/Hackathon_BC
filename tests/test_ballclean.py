from dataclasses import replace

import numpy as np
import pytest

from mtah.ballclean import MAX_BALL_SPEED_MPS, drop_implausible, implausible_mask
from mtah.gapsim import Regime, damage
from mtah.schema import FPS, Match

PITCH = (105.0, 68.0)
SMOOTH_FRAMES = 500
SMOOTH_PERIOD_S = 4.0
A_POINT = (20.0, 20.0)
B_POINT = (65.0, 20.0)
FAR_POINT = (5.0, 5.0)
N_PLAYER_SLOTS = 4

WALK_FRAMES = 40000
WALK_BLOCKS = ((0, 20000), (20000, 40000))
WALK_STEP_SD_M = 0.3
WALK_PLAYER_SLOTS = 4
DAMAGE_SEED = 21
MIN_FALSE_RECALL = 0.8
MAX_TRUE_LOSS = 0.01


def _smooth_ball(n_frames):
    """Slow, gentle ball motion around the centre spot: peak speed well below any threshold."""
    t = np.arange(n_frames) / FPS
    phase = 2.0 * np.pi * t / SMOOTH_PERIOD_S
    return np.column_stack([52.5 + 5.0 * np.sin(phase), 34.0 + 3.0 * np.cos(phase)]).astype(np.float32)


def _still_ball(n_frames, spot):
    return np.tile(np.asarray(spot, dtype=np.float32), (n_frames, 1))


def _players(n_frames, slots):
    return np.full((n_frames, slots, 2), 30.0, dtype=np.float32)


def _match(ball, blocks):
    n_frames = len(ball)
    return Match("ballclean", _players(n_frames, N_PLAYER_SLOTS), _players(n_frames, N_PLAYER_SLOTS),
                 ball, blocks, PITCH)


def _expected_mask(n_frames, spans):
    expected = np.zeros(n_frames, dtype=bool)
    for start, stop in spans:
        expected[start:stop] = True
    return expected


def test_smooth_trajectory_is_untouched():
    # Arrange
    ball = _smooth_ball(SMOOTH_FRAMES)

    # Act
    mask = implausible_mask(ball, ((0, SMOOTH_FRAMES),))

    # Assert
    assert not mask.any()


def test_five_frame_burst_far_away_is_removed_exactly():
    # Arrange
    ball = _smooth_ball(SMOOTH_FRAMES)
    ball[200:205] = FAR_POINT

    # Act
    mask = implausible_mask(ball, ((0, SMOOTH_FRAMES),))

    # Assert
    assert np.array_equal(mask, _expected_mask(SMOOTH_FRAMES, [(200, 205)]))


def test_single_frame_outlier_is_removed():
    # Arrange
    ball = _smooth_ball(SMOOTH_FRAMES)
    ball[300] = FAR_POINT

    # Act
    mask = implausible_mask(ball, ((0, SMOOTH_FRAMES),))

    # Assert
    assert np.array_equal(mask, _expected_mask(SMOOTH_FRAMES, [(300, 301)]))


def test_genuine_relocation_that_stays_is_kept():
    # Arrange
    ball = np.concatenate([_still_ball(100, A_POINT), _still_ball(100, B_POINT)])

    # Act
    mask = implausible_mask(ball, ((0, 200),))

    # Assert
    assert not mask.any()


def test_jump_across_long_gap_that_is_physically_reachable_is_kept():
    # Arrange: 45 m away, but 51 frames (about 2 s) later, which is 22 m/s, a credible move.
    # A frame-to-frame check would wrongly call this jump impossible.
    ball = np.concatenate([_still_ball(100, A_POINT), np.full((50, 2), np.nan, np.float32),
                           _still_ball(100, B_POINT)])

    # Act
    mask = implausible_mask(ball, ((0, 250),))

    # Assert
    assert not mask.any()


def test_burst_straddling_a_block_boundary_is_not_judged_across_the_boundary():
    # Arrange: frames 98-101 are a far false detection, but the boundary sits at frame 100.
    # Judged across the boundary this would be a 4-frame burst; per block nothing closes in time.
    ball = _still_ball(200, A_POINT)
    ball[98:102] = FAR_POINT
    blocks = ((0, 100), (100, 200))

    # Act
    mask = implausible_mask(ball, blocks)

    # Assert
    assert not mask.any()


def test_implausible_mask_does_not_mutate_input_track():
    # Arrange
    ball = _smooth_ball(SMOOTH_FRAMES)
    ball[200:205] = FAR_POINT
    before = ball.copy()

    # Act
    implausible_mask(ball, ((0, SMOOTH_FRAMES),))

    # Assert
    assert np.array_equal(ball, before)


def test_drop_implausible_sets_only_flagged_ball_frames_to_nan():
    # Arrange
    ball = _smooth_ball(SMOOTH_FRAMES)
    ball[200:205] = FAR_POINT
    match = _match(ball, ((0, SMOOTH_FRAMES),))

    # Act
    cleaned = drop_implausible(match)

    # Assert
    nan_frames = np.isnan(cleaned.ball_xy[:, 0])
    assert np.array_equal(nan_frames, _expected_mask(SMOOTH_FRAMES, [(200, 205)]))
    assert np.array_equal(cleaned.ball_xy[~nan_frames], ball[~nan_frames])


def test_drop_implausible_returns_new_match_and_leaves_input_and_players_untouched():
    # Arrange
    ball = _smooth_ball(SMOOTH_FRAMES)
    ball[300] = FAR_POINT
    match = _match(ball, ((0, SMOOTH_FRAMES),))
    before_ball = match.ball_xy.copy()
    before_home = match.home_xy.copy()

    # Act
    cleaned = drop_implausible(match)

    # Assert
    assert cleaned is not match
    assert np.array_equal(match.ball_xy, before_ball)
    assert np.array_equal(cleaned.home_xy, before_home)
    assert np.array_equal(cleaned.away_xy, match.away_xy)


def test_rejects_ball_track_that_is_not_two_dimensional_positions():
    # Arrange
    bad_ball = np.zeros((10, 3))

    # Act / Assert
    with pytest.raises(ValueError, match="ball_xy"):
        implausible_mask(bad_ball, ((0, 10),))


@pytest.mark.parametrize("bad_speed", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_max_speed(bad_speed):
    # Arrange
    ball = _smooth_ball(SMOOTH_FRAMES)

    # Act / Assert
    with pytest.raises(ValueError, match="max_speed"):
        implausible_mask(ball, ((0, SMOOTH_FRAMES),), max_speed=bad_speed)


def test_rejects_block_outside_the_track():
    # Arrange
    ball = _smooth_ball(SMOOTH_FRAMES)

    # Act / Assert
    with pytest.raises(ValueError, match="block"):
        implausible_mask(ball, ((0, SMOOTH_FRAMES + 1),))


def test_default_speed_limit_is_forty_five_metres_per_second():
    # Arrange / Act / Assert
    assert MAX_BALL_SPEED_MPS == 45.0


def _random_walk(rng, n_frames, n_points, step_sd_m):
    """Clipped random walk of shape [T, n_points, 2] that stays inside the pitch."""
    start = rng.uniform([0.0, 0.0], PITCH, size=(n_points, 2))
    path = start + np.cumsum(rng.normal(0.0, step_sd_m, size=(n_frames, n_points, 2)), axis=0)
    return np.clip(path, [0.0, 0.0], list(PITCH)).astype(np.float32)


def _random_walk_match(seed):
    rng = np.random.default_rng(seed)
    ball = _random_walk(rng, WALK_FRAMES, 1, WALK_STEP_SD_M)[:, 0]
    home = _random_walk(rng, WALK_FRAMES, WALK_PLAYER_SLOTS, 0.1)
    away = _random_walk(rng, WALK_FRAMES, WALK_PLAYER_SLOTS, 0.1)
    return Match("walk", home, away, ball, WALK_BLOCKS, PITCH)


def test_random_walk_damaged_with_false_detections_is_mostly_cleaned():
    # Arrange: the same regime with and without false detections. Both share the seed, so the
    # hidden frames and the jitter are identical; the only ball difference is the false bursts.
    match = _random_walk_match(seed=0)
    noisy = Regime("walk_false", 0.3, "random", dead_hidden=0.0, jitter_m=0.3, false_rate=0.02)
    quiet = replace(noisy, false_rate=0.0)
    damaged = damage(match, noisy, DAMAGE_SEED)
    clean = damage(match, quiet, DAMAGE_SEED)
    visible = ~np.isnan(damaged.ball_xy[:, 0])
    assert np.array_equal(visible, ~np.isnan(clean.ball_xy[:, 0]))
    false_truth = visible & np.any(damaged.ball_xy != clean.ball_xy, axis=1)
    true_visible = visible & ~false_truth

    # Act
    flagged = implausible_mask(damaged.ball_xy, WALK_BLOCKS)

    # Assert
    assert false_truth.sum() > 0
    assert flagged[false_truth].mean() >= MIN_FALSE_RECALL
    assert flagged[true_visible].mean() < MAX_TRUE_LOSS
