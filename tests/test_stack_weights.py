"""Tests for the weighted geometric-mean fusion of E29 (experiments/e29_stack_weights.py).

Pure numpy probability math; no data and no trained models. The import of the experiment module
pulls in the training stack (sklearn/numba), so on a light machine without it the tests skip
instead of failing.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))

try:
    from e29_stack_weights import WEIGHTS, fuse_weighted
    IMPORT_ERROR = None
except ImportError as exc:  # e.g. sklearn/numba not installed here
    fuse_weighted = None
    IMPORT_ERROR = exc

try:
    import pytest  # noqa: E402

    pytestmark = pytest.mark.skipif(
        fuse_weighted is None,
        reason="training stack not importable: {}".format(IMPORT_ERROR))
except ImportError:
    pass  # the __main__ runner below still works without pytest


def probs(seed, n=500, k=3):
    rng = np.random.default_rng(seed)
    z = rng.random((n, k)) + 0.05
    return z / z.sum(axis=1, keepdims=True)


def test_endpoints_recover_each_model():
    p, q = probs(0), probs(1)
    assert np.allclose(fuse_weighted(p, q, 0.0), p, atol=1e-6)
    assert np.allclose(fuse_weighted(p, q, 1.0), q, atol=1e-6)


def test_half_matches_e27_fixed_fuse():
    # The 50/50 row must reproduce e27_stack.fuse exactly (same formula, w = 0.5).
    p, q = probs(2), probs(3)
    z = 0.5 * (np.log(np.clip(p, 1e-6, 1)) + np.log(np.clip(q, 1e-6, 1)))
    z = np.exp(z - z.max(axis=1, keepdims=True))
    assert np.allclose(fuse_weighted(p, q, 0.5), z / z.sum(axis=1, keepdims=True))


def test_output_is_a_distribution():
    p, q = probs(4), probs(5)
    for w in WEIGHTS:
        f = fuse_weighted(p, q, w)
        assert np.all(f > 0)
        assert np.allclose(f.sum(axis=1), 1.0)


def test_weight_moves_prediction_monotonically():
    # Where the models disagree, the fused probability of the disputed class moves toward the
    # sequence model's as w grows.
    p = np.array([[0.9, 0.05, 0.05]])
    q = np.array([[0.05, 0.9, 0.05]])
    track = [fuse_weighted(p, q, w)[0, 1] for w in WEIGHTS]
    assert all(b > a for a, b in zip(track, track[1:]))


def test_zero_probabilities_are_safe():
    p = np.array([[1.0, 0.0, 0.0]])
    q = np.array([[0.0, 1.0, 0.0]])
    for w in WEIGHTS:
        f = fuse_weighted(p, q, w)
        assert np.all(np.isfinite(f)) and np.isclose(f.sum(), 1.0)


if __name__ == "__main__":
    if fuse_weighted is None:
        print("skipped: {}".format(IMPORT_ERROR))
    else:
        for name, fn in sorted(list(globals().items())):
            if name.startswith("test_"):
                fn()
                print("{} ok".format(name))
