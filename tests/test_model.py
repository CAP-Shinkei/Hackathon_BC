import numpy as np
import pytest

from mtah.model import Params, fit, load, predict_proba, save, temperature, training_rows
from mtah.schema import A, D, H, X as UNGRADED

SWAP = np.array([3, 4, 5, 0, 1, 2])
FAST = Params(max_iter=40, min_samples_leaf=20, stride=1, seed=7)
HA_D_ORDER = [1, 0, 2]


def _make_match(n_frames, seed, drop_draws=False):
    """Three mirrored features; label follows the first (home-minus-away) feature."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.uniform(-1, 1, size=(3, n_frames))
    features = np.column_stack([a, b, c, -a, -b, -c]).astype(np.float32)
    labels = np.where(a > 0.5, H, np.where(a < -0.5, A, D)).astype(np.int8)
    if drop_draws:
        keep = labels != D
        features, labels = features[keep], labels[keep]
    return features, labels, ((0, len(labels)),)


def _fit_toy(seed=0, drop_draws=False):
    matches = [_make_match(1500, seed + i, drop_draws) for i in range(2)]
    X_list, y_list, blocks_list = zip(*matches)
    return fit(list(X_list), list(y_list), list(blocks_list), SWAP, FAST)


# training_rows

def test_training_rows_excludes_unclear_label_and_frames_outside_blocks():
    # Arrange
    labels = np.array([H, A, UNGRADED, D, H, A, D, UNGRADED, H, A], dtype=np.int8)
    blocks = ((0, 4), (6, 9))

    # Act
    rows = training_rows(labels, blocks, stride=1)

    # Assert
    assert rows.tolist() == [0, 1, 3, 6, 8]


def test_training_rows_takes_every_stride_th_graded_frame():
    labels = np.array([H, A, D, H, A, D, H, A], dtype=np.int8)
    rows = training_rows(labels, ((0, 8),), stride=3)
    assert rows.tolist() == [0, 3, 6]


def test_training_rows_rejects_non_positive_stride():
    labels = np.array([H, A], dtype=np.int8)
    with pytest.raises(ValueError):
        training_rows(labels, ((0, 2),), stride=0)


def test_training_rows_does_not_mutate_labels():
    labels = np.array([H, UNGRADED, D], dtype=np.int8)
    before = labels.copy()
    training_rows(labels, ((0, 3),), stride=1)
    assert np.array_equal(labels, before)


# fit / predict_proba

def test_separable_toy_problem_reaches_high_accuracy():
    # Arrange
    model = _fit_toy()
    X_test, y_test, _ = _make_match(2000, seed=99)

    # Act
    proba = predict_proba(model, X_test, SWAP)
    accuracy = np.mean(np.argmax(proba, axis=1) == y_test)

    # Assert
    assert accuracy > 0.9


def test_probabilities_sum_to_one_with_h_a_d_column_order():
    # Arrange
    model = _fit_toy()
    X_test, y_test, _ = _make_match(500, seed=5)

    # Act
    proba = predict_proba(model, X_test, SWAP)

    # Assert
    assert proba.dtype == np.float32
    assert proba.shape == (500, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    home_rows = X_test[:, 0] > 0.9
    assert np.all(proba[home_rows].argmax(axis=1) == 0)


def test_missing_draw_class_still_gives_three_columns_in_had_order():
    # Arrange
    model = _fit_toy(drop_draws=True)
    X_test, _, _ = _make_match(300, seed=11)

    # Act
    proba = predict_proba(model, X_test, SWAP)

    # Assert
    assert proba.shape == (300, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(proba[:, 2] < 1e-6)
    away_rows = X_test[:, 0] < -0.9
    assert np.all(proba[away_rows].argmax(axis=1) == 1)


def test_predictions_are_exactly_symmetric_under_team_swap():
    # Arrange
    model = _fit_toy()
    X_test, _, _ = _make_match(400, seed=21)

    # Act
    original = predict_proba(model, X_test, SWAP)
    swapped = predict_proba(model, X_test[:, SWAP], SWAP)

    # Assert
    assert np.allclose(swapped, original[:, HA_D_ORDER], atol=1e-6)


def test_two_fits_with_same_inputs_give_identical_probabilities():
    X_test, _, _ = _make_match(200, seed=3)

    first = predict_proba(_fit_toy(seed=4), X_test, SWAP)
    second = predict_proba(_fit_toy(seed=4), X_test, SWAP)

    assert np.array_equal(first, second)


def test_fit_rejects_swap_index_with_wrong_length():
    X, y, blocks = _make_match(200, seed=1)
    with pytest.raises(ValueError):
        fit([X], [y], [blocks], np.array([0, 1, 2]), FAST)


def test_fit_rejects_empty_training_set():
    X, _, _ = _make_match(200, seed=1)
    all_unclear = np.full(len(X), UNGRADED, dtype=np.int8)
    with pytest.raises(ValueError):
        fit([X], [all_unclear], [((0, len(X)),)], SWAP, FAST)


# save / load / temperature

def test_save_then_load_reproduces_predictions(tmp_path):
    model = _fit_toy()
    X_test, _, _ = _make_match(150, seed=8)
    path = tmp_path / "model.joblib"

    save(model, path)
    restored = load(path)

    assert np.array_equal(predict_proba(model, X_test, SWAP),
                          predict_proba(restored, X_test, SWAP))


def test_temperature_one_leaves_probabilities_unchanged():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], dtype=np.float32)
    assert np.allclose(temperature(proba, 1.0), proba, atol=1e-6)


def test_high_temperature_flattens_distribution_and_keeps_rows_normalised():
    proba = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)
    flat = temperature(proba, 10.0)
    assert np.allclose(flat.sum(axis=1), 1.0, atol=1e-6)
    assert flat[0].max() < proba[0].max()


def test_temperature_does_not_mutate_input():
    proba = np.array([[0.6, 0.3, 0.1]], dtype=np.float32)
    before = proba.copy()
    temperature(proba, 2.0)
    assert np.array_equal(proba, before)


def test_temperature_rejects_non_positive_value():
    proba = np.array([[0.6, 0.3, 0.1]], dtype=np.float32)
    with pytest.raises(ValueError):
        temperature(proba, 0.0)
