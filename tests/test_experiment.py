"""Tests for the leave-one-game-out protocol on synthetic games.

Features come from a tiny stand-in featuriser and every cache lives under tmp_path, so the real
out/ folder is never read or written.
"""
import os

import numpy as np
import pytest

from mtah import experiment, model
from mtah.experiment import (
    Config,
    compare,
    decode_and_score,
    featurise_cached,
    fit_fold,
    fold_proba,
    run,
    summarise,
)
from mtah.schema import A, D, H, Match

N_FRAMES = 400
BLOCKS = ((0, 200), (200, 400))
FEATURE_NAMES = ["ball_x", "ball_y", "home_x", "home_y", "away_x", "away_y"]
SWAP = np.array([0, 1, 4, 5, 2, 3])
FAST_PARAMS = model.Params(max_iter=30, min_samples_leaf=10, stride=1, seed=0)


def _match(game_id, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.array([H, A, D, H, A, D, H, A], dtype=np.int8), N_FRAMES // 8)
    return Match(
        game_id=game_id,
        home_xy=rng.uniform(0, 100, size=(N_FRAMES, 3, 2)).astype(np.float32),
        away_xy=rng.uniform(0, 100, size=(N_FRAMES, 3, 2)).astype(np.float32),
        ball_xy=rng.uniform(0, 60, size=(N_FRAMES, 2)).astype(np.float32),
        blocks=BLOCKS,
        labels=labels,
    )


def _featurise(match):
    columns = np.column_stack([match.ball_xy, match.home_xy[:, 0], match.away_xy[:, 0]])
    return np.nan_to_num(columns).astype(np.float32), list(FEATURE_NAMES)


def _fast_config(**overrides):
    base = dict(name="test", train_regimes=("random35",), eval_regimes=("random35",),
                eval_seeds=(7,), params=FAST_PARAMS)
    return Config(**{**base, **overrides})


def _row(game, spell, macro=0.0, d_f1=0.0):
    return {"game": game, "regime": "random35", "seed": 7,
            "spell_f1": spell, "macro_f1": macro, "d_f1": d_f1}


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = str(tmp_path / "cache")
    monkeypatch.setattr(experiment, "CACHE", root)
    return root


def test_model_key_ignores_decoder_settings():
    # Arrange
    decoder_changed = Config(name="other", step=3, temperature=2.0, min_frames=10)

    # Act / Assert
    assert decoder_changed.model_key() == Config(name="base").model_key()


def test_model_key_changes_with_training_settings_and_only_when_flags_are_on():
    # Arrange
    base_key = Config(name="base").model_key()

    # Act / Assert
    assert Config(name="base", train_seed=2).model_key() != base_key
    assert Config(name="base", real_games=1).model_key() != base_key
    assert Config(name="base", clean_ball=True).model_key() != base_key
    assert Config(name="base", clean_ball=False, real_games=0).model_key() == base_key


def test_featurise_cached_reads_back_from_disk_without_refeaturising(cache_root):
    # Arrange
    calls = []

    def counting(match):
        calls.append(match.game_id)
        return _featurise(match)

    match = _match("game_a")
    cfg = _fast_config()

    # Act
    first_X, first_names = featurise_cached(match, "random35", 7, cfg, counting)
    second_X, second_names = featurise_cached(match, "random35", 7, cfg, counting)

    # Assert
    assert calls == ["game_a"]
    assert np.array_equal(first_X, second_X)
    assert first_names == second_names == FEATURE_NAMES
    assert os.path.exists(os.path.join(cache_root, "features"))


def test_featurise_cached_uses_the_game_as_loaded_when_regime_is_none(cache_root):
    # Arrange
    seen = []

    def capturing(match):
        seen.append(match)
        return _featurise(match)

    match = _match("game_b")

    # Act
    featurise_cached(match, None, 0, _fast_config(), capturing)

    # Assert
    assert np.array_equal(seen[0].ball_xy, match.ball_xy)


def test_featurise_cached_drops_ball_positions_far_outside_the_pitch(cache_root):
    # Arrange
    seen = []

    def capturing(match):
        seen.append(match)
        return _featurise(match)

    match = _match("game_c")
    far_ball = match.ball_xy.copy()
    far_ball[0] = (-50.0, 0.0)
    match = match.with_ball(far_ball)

    # Act
    featurise_cached(match, None, 0, _fast_config(), capturing)

    # Assert
    assert np.isnan(seen[0].ball_xy[0]).all()


def test_featurise_cached_writes_separate_file_for_cleaned_ball(cache_root):
    # Arrange
    cfg = _fast_config(clean_ball=True)

    # Act
    featurise_cached(_match("game_d"), None, 0, cfg, _featurise)

    # Assert
    files = os.listdir(os.path.join(cache_root, "features"))
    assert len(files) == 1
    assert files[0].startswith("game_d_None_0_v1_1_clean_")


def test_featurise_cached_does_not_reuse_features_after_labels_change(cache_root):
    # Arrange
    cfg = _fast_config()
    match = _match("game_e")
    relabelled = match.with_labels(np.roll(match.labels, 1))

    # Act
    featurise_cached(match, "dark50", 1, cfg, _featurise)
    featurise_cached(relabelled, "dark50", 1, cfg, _featurise)

    # Assert
    assert len(os.listdir(os.path.join(cache_root, "features"))) == 2


def test_fit_fold_trains_one_model_per_model_seed_and_returns_swap_index(cache_root):
    # Arrange
    cfg = _fast_config(model_seeds=(0, 1))
    train = [_match("train_a"), _match("train_b", seed=1)]

    # Act
    models, swap = fit_fold(train, cfg, _featurise)

    # Assert
    assert len(models) == 2
    assert np.array_equal(swap, SWAP)


def test_fit_fold_adds_real_games_as_one_extra_training_block(cache_root):
    # Arrange
    cfg = _fast_config()
    train = [_match("train_a")]
    real = (_match("real_a", seed=2),)

    # Act
    models, _ = fit_fold(train, cfg, _featurise, real=real)

    # Assert
    assert len(models) == 1


def test_fold_proba_returns_one_probability_array_per_evaluation_key(cache_root):
    # Arrange
    cfg = _fast_config(eval_seeds=(7, 8))
    held_out = _match("held")

    # Act
    proba = fold_proba(held_out, [_match("train_a", seed=1)], cfg, _featurise)

    # Assert
    assert set(proba) == {("random35", 7), ("random35", 8)}
    assert proba[("random35", 7)].shape == (N_FRAMES, 3)


def test_fold_proba_reads_saved_probabilities_instead_of_refitting(cache_root, monkeypatch):
    # Arrange
    cfg = _fast_config()
    held_out = _match("held")
    train = [_match("train_a", seed=1)]
    first = fold_proba(held_out, train, cfg, _featurise)

    def refuse_to_fit(*args, **kwargs):
        raise AssertionError("fit_fold must not run when the probabilities are cached")

    monkeypatch.setattr(experiment, "fit_fold", refuse_to_fit)

    # Act
    second = fold_proba(held_out, train, cfg, _featurise)

    # Assert
    assert np.array_equal(first[("random35", 7)], second[("random35", 7)])


def test_decode_and_score_returns_official_metrics_for_the_held_out_game(cache_root):
    # Arrange
    held_out = _match("held")
    train = [_match("train_a", seed=1)]
    proba = np.full((N_FRAMES, 3), 1.0 / 3.0)

    # Act
    result = decode_and_score(proba, held_out, train, _fast_config())

    # Assert
    assert 0.0 <= result["spell_f1"] <= 1.0
    assert result["n_gt_spells"] > 0


def test_run_rejects_a_real_game_count_that_differs_from_the_config(cache_root):
    # Arrange
    cfg = _fast_config(real_games=1)

    # Act / Assert
    with pytest.raises(ValueError, match="real games"):
        run([_match("a"), _match("b")], cfg, _featurise, log=lambda line: None)


def test_run_returns_one_row_per_held_out_game_regime_and_seed(cache_root):
    # Arrange
    matches = [_match("g1"), _match("g2", seed=1), _match("g3", seed=2)]
    logs = []

    # Act
    rows = run(matches, _fast_config(), _featurise, log=logs.append)

    # Assert
    assert sorted(r["game"] for r in rows) == ["g1", "g2", "g3"]
    assert all(r["regime"] == "random35" and r["seed"] == 7 for r in rows)
    assert len(logs) == 3
    assert all(line.startswith("test") for line in logs)


def test_run_with_real_games_trains_on_them_and_still_scores_each_held_out_game(cache_root):
    # Arrange
    cfg = _fast_config(real_games=1)
    matches = [_match("g1"), _match("g2", seed=1)]
    real = (_match("real", seed=3),)

    # Act
    rows = run(matches, cfg, _featurise, log=lambda line: None, real=real)

    # Assert
    assert sorted(r["game"] for r in rows) == ["g1", "g2"]


def test_summarise_averages_each_game_before_averaging_games():
    # Arrange
    rows = [_row("g1", 0.2), _row("g1", 0.4), _row("g2", 0.9)]

    # Act
    summary = summarise(rows)

    # Assert
    assert summary["per_game"]["g1"]["spell_f1"] == pytest.approx(0.3)
    assert summary["spell_f1"] == pytest.approx(0.6)


def test_compare_reports_mean_gain_and_how_many_games_improved():
    # Arrange
    base = [_row("g1", 0.5), _row("g2", 0.5)]
    new = [_row("g1", 0.7), _row("g2", 0.4)]

    # Act
    mean_gain, improved, total = compare(base, new)

    # Assert
    assert mean_gain == pytest.approx(0.05)
    assert improved == 1
    assert total == 2
