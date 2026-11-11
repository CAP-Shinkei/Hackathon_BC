"""The Match container refuses arrays that do not describe one consistent match."""
import numpy as np
import pytest

from mtah.schema import A, D, H, Match, swap_teams

N_FRAMES = 10


def fields(**changes):
    base = {
        "game_id": "synthetic",
        "home_xy": np.zeros((N_FRAMES, 3, 2), dtype=np.float32),
        "away_xy": np.ones((N_FRAMES, 2, 2), dtype=np.float32),
        "ball_xy": np.zeros((N_FRAMES, 2), dtype=np.float32),
        "blocks": ((0, N_FRAMES),),
    }
    return {**base, **changes}


def test_a_consistent_match_is_accepted():
    assert Match(**fields()).n_frames == N_FRAMES


@pytest.mark.parametrize("changes, message", [
    ({"home_xy": np.zeros((N_FRAMES - 1, 3, 2), dtype=np.float32)}, "frame counts differ"),
    ({"away_xy": np.zeros((N_FRAMES, 2, 3), dtype=np.float32)}, "bad coordinate shapes"),
    ({"ball_xy": np.zeros((N_FRAMES, 3), dtype=np.float32)}, "bad coordinate shapes"),
    ({"labels": np.zeros(N_FRAMES + 1, dtype=np.int8)}, "labels length differs"),
    ({"blocks": ((0, N_FRAMES + 1),)}, "bad block"),
    ({"blocks": ((5, 5),)}, "bad block"),
])
def test_inconsistent_arrays_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        Match(**fields(**changes))


def test_swap_teams_exchanges_the_sides_and_their_labels_without_touching_the_original():
    labels = np.array([H, A, D] + [D] * (N_FRAMES - 3), dtype=np.int8)
    match = Match(**fields(labels=labels))

    swapped = swap_teams(match)

    assert swapped.home_xy is match.away_xy and swapped.away_xy is match.home_xy
    assert swapped.labels[:3].tolist() == [A, H, D]
    assert match.labels[:3].tolist() == [H, A, D]


def test_swap_teams_keeps_a_match_without_labels_unlabelled():
    assert swap_teams(Match(**fields())).labels is None
