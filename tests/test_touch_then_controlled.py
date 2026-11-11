"""Tests for the label logic of E28 (experiments/e28_touch_then_controlled.py).

Pure-label checks: no tracking data, no trained model and no official scorer needed, so they
run anywhere (`python -m pytest tests/test_touch_then_controlled.py`). They pin the properties
the experiment relies on when it moves external/controlled.py from the training labels to the
decoded prediction.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "external"))
sys.path.insert(0, REPO)

from controlled import controlled  # noqa: E402
from mtah.labels import A, D, H, X, runs  # noqa: E402

FPS = 25


def seq(*spells):
    """(code, seconds) pairs -> frame-level codes."""
    return np.concatenate([np.full(int(round(s * FPS)), c, dtype=np.int8)
                           for c, s in spells])


def n_spells(codes):
    return sum(1 for c, _, _ in runs(codes) if c in (H, A))


def random_labels(seed, n_spells_=200):
    rng = np.random.default_rng(seed)
    codes = np.concatenate([
        np.full(rng.integers(2, 8 * FPS), rng.choice([H, A, D], p=[0.42, 0.42, 0.16]),
                dtype=np.int8)
        for _ in range(n_spells_)])
    return codes


def test_min_s_zero_is_identity():
    gt = random_labels(0)
    assert np.array_equal(controlled(gt, 0.0), gt)


def test_short_touch_returns_to_previous_team():
    # A 1 s opposing touch inside a long H spell disappears at min_s = 2 s.
    gt = seq((H, 10), (A, 1), (H, 10))
    assert (controlled(gt, 2.0) == H).all()


def test_first_spell_after_restart_is_kept():
    # The first spell after a stoppage stays whatever its length (restart rule).
    gt = seq((D, 30), (A, 1), (H, 10))
    out = controlled(gt, 2.0)
    assert (out[:30 * FPS] == D).all()
    assert (out[30 * FPS:31 * FPS] == A).all()
    assert (out[31 * FPS:] == H).all()


def test_controlled_is_idempotent():
    # Applying the rule twice changes nothing: the E28 sweep can re-apply it to a decoded
    # sequence without the result depending on earlier passes.
    gt = random_labels(1)
    for ms in (1.0, 2.0, 3.0):
        once = controlled(gt, ms)
        assert np.array_equal(controlled(once, ms), once)


def test_sweep_spell_count_non_increasing():
    # A higher threshold merges at least as much: predicted spell counts cannot grow with
    # min_s. Guards the "does it fragment?" reading of the E28 table.
    gt = random_labels(2)
    counts = [n_spells(controlled(gt, ms)) for ms in (0.0, 0.5, 1.0, 2.0, 3.0)]
    assert counts == sorted(counts, reverse=True)


def test_dead_and_ungraded_frames_are_untouched():
    # The rule relabels team spells only; D and X frames must survive bit-identical, so the
    # fixed evaluation labels and the decode blocks are the same as in E25.
    gt = random_labels(3)
    out = controlled(gt, 2.0)
    assert np.array_equal(out == D, gt == D)
    assert np.array_equal(out == X, gt == X)


def test_perfect_touch_prediction_recovers_eval_labels():
    # The consistency the experiment counts on: if the touch-level decoder were perfect,
    # applying the rule to its prediction reproduces exactly the evaluation labels E25
    # trains on, at any threshold.
    gt = random_labels(4)
    for ms in (1.0, 2.0, 3.0):
        assert np.array_equal(controlled(gt.copy(), ms), controlled(gt, ms))


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("{} ok".format(name))
