import numpy as np
import pytest

from mtah.gapsim import REGIMES, Regime, damage, hidden_mask
from mtah.schema import D, H, Match

N_FRAMES = 20000
PITCH = (105.0, 68.0)
BLOCKS = ((0, 10000), (10000, 20000))
RATE_TOLERANCE = 0.02
N_PLAYERS_PER_TEAM = 11


def _walk(rng, n_frames, n_points, step_sd_m):
    """Clipped random walk of shape [T, n_points, 2] that stays inside the pitch."""
    start = rng.uniform([0.0, 0.0], PITCH, size=(n_points, 2))
    path = start + np.cumsum(rng.normal(0.0, step_sd_m, size=(n_frames, n_points, 2)), axis=0)
    return np.clip(path, [0.0, 0.0], list(PITCH)).astype(np.float32)


def _synthetic_match(seed=0, labels=None):
    rng = np.random.default_rng(seed)
    home = _walk(rng, N_FRAMES, N_PLAYERS_PER_TEAM, 0.1)
    away = _walk(rng, N_FRAMES, N_PLAYERS_PER_TEAM, 0.1)
    ball = _walk(rng, N_FRAMES, 1, 0.5)[:, 0]
    return Match("synthetic", home, away, ball, BLOCKS, PITCH, labels)


def _in_block(n_frames=N_FRAMES):
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in BLOCKS:
        mask[start:stop] = True
    return mask


@pytest.mark.parametrize("regime_name", ["random35", "random50", "crowd41", "mixed41"])
def test_hidden_share_within_tolerance_of_target_rate(regime_name):
    # Arrange
    match = _synthetic_match()
    regime = REGIMES[regime_name]

    # Act
    hidden = hidden_mask(match, regime, seed=3)
    achieved = hidden[_in_block()].mean()

    # Assert
    assert abs(achieved - regime.rate) <= RATE_TOLERANCE


def test_same_seed_gives_identical_hidden_mask():
    # Arrange
    match = _synthetic_match()
    regime = REGIMES["mixed41"]

    # Act
    first = hidden_mask(match, regime, seed=11)
    second = hidden_mask(match, regime, seed=11)

    # Assert
    assert np.array_equal(first, second)


def test_different_seed_gives_different_hidden_mask():
    # Arrange
    match = _synthetic_match()
    regime = REGIMES["random35"]

    # Act
    first = hidden_mask(match, regime, seed=1)
    second = hidden_mask(match, regime, seed=2)

    # Assert
    assert not np.array_equal(first, second)


def test_damage_does_not_mutate_input_match():
    # Arrange
    labels = np.full(N_FRAMES, H, dtype=np.int8)
    match = _synthetic_match(labels=labels)
    before = (match.home_xy.copy(), match.away_xy.copy(), match.ball_xy.copy(), labels.copy())

    # Act
    damage(match, REGIMES["mixed41_art"], seed=5)

    # Assert
    assert np.array_equal(match.home_xy, before[0])
    assert np.array_equal(match.away_xy, before[1])
    assert np.array_equal(match.ball_xy, before[2])
    assert np.array_equal(match.labels, before[3])


def test_damage_keeps_labels_and_hides_ball_exactly_where_mask_says():
    # Arrange
    labels = np.full(N_FRAMES, H, dtype=np.int8)
    match = _synthetic_match(labels=labels)
    regime = REGIMES["random50"]

    # Act
    damaged = damage(match, regime, seed=7)

    # Assert
    assert np.array_equal(damaged.labels, labels)
    assert np.array_equal(np.isnan(damaged.ball_xy[:, 0]), hidden_mask(match, regime, seed=7))


def _crowd_test_match(n_frames):
    """Ball at centre. First half: 22 players around it. Second half: nobody near it."""
    half = n_frames // 2
    ball = np.tile([52.5, 34.0], (n_frames, 1)).astype(np.float32)
    players = np.empty((n_frames, 2 * N_PLAYERS_PER_TEAM, 2), dtype=np.float32)
    offsets = np.random.default_rng(0).uniform(-0.5, 0.5, size=(2 * N_PLAYERS_PER_TEAM, 2))
    players[:half] = ball[:half, None, :] + offsets
    players[half:] = [1.0, 1.0]
    home = players[:, :N_PLAYERS_PER_TEAM]
    away = players[:, N_PLAYERS_PER_TEAM:]
    return Match("crowd", home, away, ball, ((0, n_frames),), PITCH)


def test_crowd_mechanism_hides_crowded_frames_more_often_than_uncrowded():
    # Arrange
    n_frames = 20000
    match = _crowd_test_match(n_frames)
    regime = Regime("crowd_probe", 0.2, "crowd", dead_hidden=0.0)

    # Act
    hidden = hidden_mask(match, regime, seed=4)
    half = n_frames // 2
    crowded_hidden = np.count_nonzero(hidden[:half])
    uncrowded_hidden = np.count_nonzero(hidden[half:])

    # Assert
    assert crowded_hidden > 5 * uncrowded_hidden


def test_dead_ball_stretches_hidden_entirely_when_dead_hidden_is_one():
    # Arrange
    labels = np.full(N_FRAMES, H, dtype=np.int8)
    labels[2000:2500] = D
    labels[12000:12300] = D
    match = _synthetic_match(labels=labels)
    regime = Regime("dead_probe", 0.2, "random", dead_hidden=1.0)

    # Act
    hidden = hidden_mask(match, regime, seed=9)

    # Assert
    assert hidden[labels == D].all()


def test_dead_hidden_is_ignored_when_labels_are_unknown():
    # Arrange
    match = _synthetic_match(labels=None)
    regime = Regime("no_labels", 0.2, "random", dead_hidden=1.0)

    # Act
    hidden = hidden_mask(match, regime, seed=9)

    # Assert
    assert abs(hidden[_in_block()].mean() - regime.rate) <= RATE_TOLERANCE


def test_false_detections_stay_inside_pitch_and_are_applied():
    # Arrange
    match = _synthetic_match()
    clean = damage(match, Regime("clean", 0.2, "random", dead_hidden=0.0), seed=6)
    noisy = damage(match, Regime("false", 0.2, "random", dead_hidden=0.0, false_rate=0.05), seed=6)
    visible = ~np.isnan(noisy.ball_xy[:, 0])

    # Act
    x_in_pitch = (noisy.ball_xy[visible, 0] >= 0) & (noisy.ball_xy[visible, 0] <= PITCH[0])
    y_in_pitch = (noisy.ball_xy[visible, 1] >= 0) & (noisy.ball_xy[visible, 1] <= PITCH[1])
    changed = np.any(noisy.ball_xy[visible] != clean.ball_xy[visible])

    # Assert
    assert x_in_pitch.all() and y_in_pitch.all()
    assert changed


def test_player_drop_share_is_close_to_target():
    # Arrange
    match = _synthetic_match()
    regime = Regime("drops", 0.1, "random", dead_hidden=0.0, player_drop=0.05)

    # Act
    damaged = damage(match, regime, seed=8)
    players = np.concatenate([damaged.home_xy, damaged.away_xy], axis=1)
    dropped_share = np.mean(np.isnan(players[:, :, 0]))

    # Assert
    assert abs(dropped_share - regime.player_drop) <= RATE_TOLERANCE


def test_regime_rejects_unknown_mechanism_and_out_of_range_rate():
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        Regime("bad_mechanism", 0.3, "sometimes", dead_hidden=0.0)
    with pytest.raises(ValueError):
        Regime("bad_rate", 1.5, "random", dead_hidden=0.0)
