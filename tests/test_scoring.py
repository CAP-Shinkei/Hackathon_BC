import numpy as np

from mtah.schema import A, D, H, X, Match, swap_teams
from mtah.scoring import score


def _labels():
    return np.array([H] * 100 + [D] * 20 + [A] * 80 + [H] * 50, dtype=np.int8)


def test_perfect_prediction_scores_one():
    gt = _labels()
    result = score(gt, gt.copy(), blocks=((0, len(gt)),))
    assert result["spell_f1"] == 1.0 and result["macro_f1"] == 1.0
    assert result["n_gt_spells"] == 3


def test_constant_guess_scores_zero_spell_f1():
    gt = _labels()
    result = score(gt, np.full(len(gt), H, dtype=np.int8), blocks=((0, len(gt)),))
    assert result["spell_f1"] < 0.5


def test_frames_outside_blocks_are_not_graded():
    gt = _labels()
    pred = gt.copy()
    pred[:10] = A
    assert score(gt, pred, blocks=((10, len(gt)),))["spell_f1"] == 1.0


def test_swap_teams_swaps_labels_and_tracks():
    n = 4
    match = Match("g", np.zeros((n, 2, 2), np.float32), np.ones((n, 3, 2), np.float32),
                  np.zeros((n, 2), np.float32), ((0, n),),
                  labels=np.array([H, A, D, X], dtype=np.int8))
    swapped = swap_teams(match)
    assert swapped.labels.tolist() == [A, H, D, X]
    assert swapped.home_xy.shape == (n, 3, 2)
