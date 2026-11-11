"""The official baseline redone on a Match: nearest player, then the organisers' smoothing."""
import numpy as np

from mtah import baseline
from mtah.schema import A, D, H, Match

PITCH = (105.0, 68.0)
N_FRAMES = 6


def make_match():
    """Home player at x=10, away player at x=90; the ball sits next to one of them or is missing."""
    home = np.tile(np.array([[10.0, 34.0]], dtype=np.float32), (N_FRAMES, 1, 1))
    away = np.tile(np.array([[90.0, 34.0]], dtype=np.float32), (N_FRAMES, 1, 1))
    ball = np.array([[12.0, 34.0], [12.0, 34.0], [88.0, 34.0],
                     [88.0, 34.0], [np.nan, np.nan], [50.0, 34.0]], dtype=np.float32)
    return Match(game_id="synthetic", home_xy=home, away_xy=away, ball_xy=ball,
                 blocks=((0, N_FRAMES),), pitch=PITCH)


def test_nearest_team_follows_the_closest_player_and_marks_a_missing_ball():
    codes = baseline.nearest_team(make_match())

    home, away, nobody = (baseline.official_baseline.HOME, baseline.official_baseline.AWAY,
                          baseline.official_baseline.NOBODY)
    assert codes.tolist() == [home, home, away, away, nobody, home]


def test_nearest_team_marks_frames_without_any_player_as_nobody():
    match = make_match()
    empty = np.full_like(match.home_xy, np.nan)
    unseen = Match(game_id="synthetic", home_xy=empty, away_xy=empty, ball_xy=match.ball_xy,
                   blocks=match.blocks, pitch=PITCH)

    codes = baseline.nearest_team(unseen)

    assert (codes == baseline.official_baseline.NOBODY).all()


def test_predict_without_smoothing_maps_letters_to_label_codes():
    codes = baseline.predict(make_match(), window=0)

    assert codes.dtype == np.int8
    assert codes.tolist() == [H, H, A, A, D, H]
