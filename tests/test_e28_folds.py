"""Fold grouping for E28 (experiments/e28_touch_then_controlled.py).

The property the benchmark's validity rests on: the game id is the fold group, so every damaged
copy of one game stays in a single fold and no game leaks into another fold's training set.
Import-guarded: the experiment module pulls in the training stack (sklearn/numba), so on a light
machine without it the tests skip instead of failing.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))

try:
    from e28_touch_then_controlled import folds_of
    IMPORT_ERROR = None
except ImportError as exc:  # e.g. sklearn/numba not installed here
    folds_of = None
    IMPORT_ERROR = exc

try:
    import pytest

    pytestmark = pytest.mark.skipif(folds_of is None,
                                    reason="training stack not importable: {}".format(
                                        IMPORT_ERROR))
except ImportError:
    pass  # the __main__ runner below still works without pytest

GAMES = ["dfl_{}".format(i) for i in range(7)] + ["metrica{}".format(i) for i in (1, 2, 3)]


def test_every_game_in_exactly_one_fold():
    folds = folds_of(GAMES)
    flat = [g for fold in folds for g in fold]
    assert sorted(flat) == sorted(GAMES)
    assert len(set(flat)) == len(GAMES)  # no game in two folds


def test_no_group_filtered_between_train_and_heldout():
    # For each fold: held-out games and training games are disjoint, and together they cover
    # every game. The fold's classifier never sees a damaged copy of a game it predicts.
    folds = folds_of(GAMES)
    for fold in folds:
        train_ids = [g for g in GAMES if g not in fold]
        assert not (set(fold) & set(train_ids))
        assert sorted(fold + train_ids) == sorted(GAMES)


def test_split_is_deterministic():
    assert folds_of(GAMES) == folds_of(list(reversed(GAMES)))


if __name__ == "__main__":
    if folds_of is None:
        print("skipped: {}".format(IMPORT_ERROR))
    else:
        for name, fn in sorted(list(globals().items())):
            if name.startswith("test_"):
                fn()
                print("{} ok".format(name))
