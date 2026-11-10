import json

import numpy as np
import pytest

from mtah.external import skillcorner
from mtah.schema import A, D, FPS, H, X

NAN = np.nan
HOME_ID, AWAY_ID = 1, 2
HOME_PLAYER, AWAY_PLAYER = 11, 21
MATCH_ID = "1874553"
UNKNOWN_OWNER = skillcorner.UNKNOWN


def _meta_json(tmp_path, home_id=HOME_ID, away_id=AWAY_ID, periods=None):
    if periods is None:
        periods = [
            {"period": 1, "start_frame": 0, "end_frame": 19},
            {"period": 2, "start_frame": 20, "end_frame": 39},
        ]
    data = {
        "home_team": {"id": home_id},
        "away_team": {"id": away_id},
        "match_periods": periods,
        "players": [
            {"id": HOME_PLAYER, "team_id": HOME_ID},
            {"id": 12, "team_id": HOME_ID},
            {"id": AWAY_PLAYER, "team_id": AWAY_ID},
        ],
    }
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _record(frame, group=None, ball=None, players=()):
    ball = ball or {"x": 0.0, "y": 0.0, "is_detected": True}
    return {
        "frame": frame,
        "ball_data": ball,
        "possession": {"player_id": None, "group": group},
        "player_data": list(players),
    }


def _write_jsonl(tmp_path, records, name="tracking.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _write_phases(tmp_path, rows, name="phases.csv"):
    path = tmp_path / name
    lines = ["index,frame_start,frame_end"]
    lines += ["{},{},{}".format(i, start, stop) for i, (start, stop) in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_resample_repeats_nearest_source_frame_for_each_output_frame():
    # Arrange
    source = np.arange(5)

    # Act
    result = skillcorner.resample(source, 10, 10)

    # Assert
    np.testing.assert_array_equal(result, [0, 0, 1, 1, 2, 2, 2, 3, 3, 4])


def test_resample_keeps_missing_values_without_interpolating_them():
    # Arrange
    source = np.array([[0.0], [NAN], [2.0]], dtype=np.float32)

    # Act
    result = skillcorner.resample(source, 10, 5)

    # Assert
    np.testing.assert_array_equal(result[:, 0], np.array([0.0, 0.0, NAN, NAN, 2.0]))


def test_resample_returns_empty_track_for_zero_output_frames():
    # Arrange
    source = np.zeros((3, 2))

    # Act
    result = skillcorner.resample(source, 10, 0)

    # Assert
    assert result.shape == (0, 2)


def test_resample_does_not_mutate_its_input():
    # Arrange
    source = np.arange(4.0)
    before = source.copy()

    # Act
    skillcorner.resample(source, 10, 8)

    # Assert
    np.testing.assert_array_equal(source, before)


def test_resample_rejects_an_empty_track():
    # Arrange
    empty = np.zeros((0, 2))

    # Act / Assert
    with pytest.raises(ValueError, match="empty"):
        skillcorner.resample(empty, 10, 5)


@pytest.mark.parametrize("src_fps, n_out", [(0, 5), (-10, 5), (10, -1)])
def test_resample_rejects_non_positive_rates_or_negative_length(src_fps, n_out):
    # Arrange
    source = np.arange(3)

    # Act / Assert
    with pytest.raises(ValueError):
        skillcorner.resample(source, src_fps, n_out)


def test_fill_short_gaps_fills_a_short_unknown_run_between_one_team():
    # Arrange
    owner = np.array([H, H, -1, -1, H, H], dtype=np.int8)

    # Act
    filled = skillcorner.fill_short_gaps(owner, short_frames=50)

    # Assert
    np.testing.assert_array_equal(filled, [H, H, H, H, H, H])


def test_fill_short_gaps_leaves_a_run_at_least_as_long_as_the_limit_unknown():
    # Arrange
    owner = np.array([H, -1, -1, H], dtype=np.int8)

    # Act
    filled = skillcorner.fill_short_gaps(owner, short_frames=2)

    # Assert
    np.testing.assert_array_equal(filled, [H, -1, -1, H])


def test_fill_short_gaps_leaves_a_run_between_different_teams_unknown():
    # Arrange
    owner = np.array([H, -1, A], dtype=np.int8)

    # Act
    filled = skillcorner.fill_short_gaps(owner, short_frames=50)

    # Assert
    np.testing.assert_array_equal(filled, [H, -1, A])


def test_fill_short_gaps_leaves_edge_runs_unknown():
    # Arrange
    owner = np.array([-1, -1, H, A, -1], dtype=np.int8)

    # Act
    filled = skillcorner.fill_short_gaps(owner, short_frames=50)

    # Assert
    np.testing.assert_array_equal(filled, [-1, -1, H, A, -1])


def test_fill_short_gaps_does_not_mutate_its_input():
    # Arrange
    owner = np.array([A, -1, A], dtype=np.int8)

    # Act
    skillcorner.fill_short_gaps(owner, short_frames=50)

    # Assert
    np.testing.assert_array_equal(owner, [A, -1, A])


def test_build_labels_marks_dead_ball_inside_a_half_outside_play():
    # Arrange
    owner = np.array([H, H, H, H], dtype=np.int8)
    in_play = np.array([False, True, True, False])
    in_block = np.array([True, True, True, True])

    # Act
    labels = skillcorner.build_labels(owner, in_play, in_block)

    # Assert
    np.testing.assert_array_equal(labels, [D, H, H, D])


def test_build_labels_marks_frames_outside_the_halves_as_ungraded():
    # Arrange
    owner = np.array([A, A, A], dtype=np.int8)
    in_play = np.array([True, True, True])
    in_block = np.array([False, True, False])

    # Act
    labels = skillcorner.build_labels(owner, in_play, in_block)

    # Assert
    np.testing.assert_array_equal(labels, [X, A, X])


def test_build_labels_marks_in_play_frames_with_unknown_owner_as_ungraded():
    # Arrange
    owner = np.array([-1, A], dtype=np.int8)
    in_play = np.array([True, True])
    in_block = np.array([True, True])

    # Act
    labels = skillcorner.build_labels(owner, in_play, in_block)

    # Assert
    np.testing.assert_array_equal(labels, [X, A])
    assert labels.dtype == np.int8


def test_build_labels_does_not_mutate_its_inputs():
    # Arrange
    owner = np.array([H, H], dtype=np.int8)
    in_play = np.array([True, False])
    in_block = np.array([True, True])

    # Act
    skillcorner.build_labels(owner, in_play, in_block)

    # Assert
    np.testing.assert_array_equal(owner, [H, H])
    np.testing.assert_array_equal(in_play, [True, False])


def test_shift_moves_centred_metres_to_a_corner_origin():
    # Arrange
    pitch = (105.0, 68.0)

    # Act
    shifted = skillcorner._shift(-10.0, 5.0, pitch)

    # Assert
    assert shifted == (42.5, 39.0)


def test_slot_map_numbers_each_side_in_sorted_id_order():
    # Arrange
    roster = {skillcorner.HOME_SIDE: [5, 9], skillcorner.AWAY_SIDE: [3]}

    # Act
    slots = skillcorner._slot_map(roster)

    # Assert
    assert slots == {5: (0, 0), 9: (0, 1), 3: (1, 0)}


def test_read_match_info_returns_frame_count_halves_and_rosters(tmp_path):
    # Arrange
    path = _meta_json(tmp_path)

    # Act
    info = skillcorner.read_match_info(path)

    # Assert
    assert info["n_frames"] == 40
    assert info["halves"] == ((0, 20), (20, 40))
    assert info["roster"] == {0: [HOME_PLAYER, 12], 1: [AWAY_PLAYER]}


def test_read_match_info_rejects_identical_team_ids(tmp_path):
    # Arrange
    path = _meta_json(tmp_path, home_id=HOME_ID, away_id=HOME_ID)

    # Act / Assert
    with pytest.raises(ValueError, match="must differ"):
        skillcorner.read_match_info(path)


def test_read_match_info_rejects_a_match_without_periods(tmp_path):
    # Arrange
    path = _meta_json(tmp_path, periods=[])

    # Act / Assert
    with pytest.raises(ValueError, match="match_periods"):
        skillcorner.read_match_info(path)


def test_read_tracking_drops_extrapolated_positions_and_maps_possession(tmp_path):
    # Arrange
    info = {"n_frames": 2, "halves": ((0, 2),), "roster": {0: [HOME_PLAYER], 1: [AWAY_PLAYER]}}
    records = [
        _record(0, group="home team",
                ball={"x": 1.0, "y": 2.0, "is_detected": True},
                players=[{"player_id": HOME_PLAYER, "x": -10.0, "y": 0.0, "is_detected": True},
                         {"player_id": AWAY_PLAYER, "x": 3.0, "y": 3.0, "is_detected": False}]),
        _record(1, group="away team",
                ball={"x": 9.0, "y": 9.0, "is_detected": False},
                players=[]),
    ]
    path = _write_jsonl(tmp_path, records)

    # Act
    home, away, ball, owner = skillcorner._read_tracking(path, info)

    # Assert
    np.testing.assert_allclose(home[0, 0], [42.5, 34.0])
    assert np.isnan(away[0, 0]).all()
    np.testing.assert_allclose(ball[0], [53.5, 36.0])
    assert np.isnan(ball[1]).all()
    np.testing.assert_array_equal(owner, [skillcorner.HOME_SIDE, skillcorner.AWAY_SIDE])


def test_read_tracking_ignores_records_after_the_last_period(tmp_path):
    # Arrange
    info = {"n_frames": 2, "halves": ((0, 2),), "roster": {0: [HOME_PLAYER], 1: []}}
    path = _write_jsonl(tmp_path, [_record(0, group="home team"), _record(2, group="away team")])

    # Act
    _, _, ball, owner = skillcorner._read_tracking(path, info)

    # Assert
    assert ball.shape == (2, 2)
    np.testing.assert_array_equal(owner, [skillcorner.HOME_SIDE, UNKNOWN_OWNER])


def test_read_tracking_rejects_a_negative_frame_index(tmp_path):
    # Arrange
    info = {"n_frames": 2, "halves": ((0, 2),), "roster": {0: [HOME_PLAYER], 1: []}}
    path = _write_jsonl(tmp_path, [_record(-1)])

    # Act / Assert
    with pytest.raises(ValueError, match="negative"):
        skillcorner._read_tracking(path, info)


def test_phase_mask_marks_inclusive_frame_ranges_and_clips_to_the_match(tmp_path):
    # Arrange
    path = _write_phases(tmp_path, [(2, 4), (7, 9)])

    # Act
    mask = skillcorner._phase_mask(path, 8)

    # Assert
    np.testing.assert_array_equal(mask, [False, False, True, True, True, False, False, True])


def test_download_rejects_a_git_lfs_pointer_and_removes_it(tmp_path):
    # Arrange
    pointer = tmp_path / "pointer.jsonl"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(OSError, match="git-lfs"):
        skillcorner._check_not_lfs_pointer(pointer)
    assert not pointer.exists()


def test_load_rejects_an_unknown_game_id(tmp_path):
    # Arrange
    unknown = "skc_999"

    # Act / Assert
    with pytest.raises(ValueError, match="unknown SkillCorner"):
        skillcorner.load(unknown, data_dir=tmp_path)


def test_load_reports_missing_raw_files_before_building(tmp_path):
    # Arrange
    game_id = skillcorner.GAME_PREFIX + MATCH_ID

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="download"):
        skillcorner.load(game_id, data_dir=tmp_path)


def _write_synthetic_match(tmp_path):
    """Forty 10 fps frames in two halves of twenty; home plays in phases and holds the ball early."""
    _meta_json(tmp_path)
    owner_groups = {f: "home team" for f in list(range(3, 11)) + [14, 15]}
    records = []
    for frame in range(40):
        ball = {"x": 0.0, "y": 0.0, "is_detected": frame % 2 == 0}
        players = [{"player_id": HOME_PLAYER, "x": -10.0, "y": 0.0, "is_detected": True},
                   {"player_id": AWAY_PLAYER, "x": 10.0, "y": 0.0, "is_detected": False}]
        records.append(_record(frame, group=owner_groups.get(frame), ball=ball, players=players))
    _write_jsonl(tmp_path, records, name=MATCH_ID + "_tracking_extrapolated.jsonl")
    _write_phases(tmp_path, [(2, 15), (22, 30)], name=MATCH_ID + "_phases_of_play.csv")
    (tmp_path / (MATCH_ID + "_match.json")).write_text(
        (tmp_path / "meta.json").read_text(encoding="utf-8"), encoding="utf-8")


def test_build_match_end_to_end_produces_25fps_match_with_labels(tmp_path):
    # Arrange
    _write_synthetic_match(tmp_path)

    # Act
    match = skillcorner._build_match(skillcorner.GAME_PREFIX + MATCH_ID, tmp_path)

    # Assert
    assert match.n_frames == 40 * FPS // 10
    assert match.blocks == ((0, 50), (50, 100))
    assert match.home_xy.shape == (100, 2, 2)
    assert np.isnan(match.away_xy).all()
    assert match.labels[0] == D
    assert match.labels[30] == H
    assert match.labels[99] == D
    assert match.labels.dtype == np.int8


def test_build_match_end_to_end_marks_ball_missing_frames_as_nan(tmp_path):
    # Arrange
    _write_synthetic_match(tmp_path)

    # Act
    match = skillcorner._build_match(skillcorner.GAME_PREFIX + MATCH_ID, tmp_path)

    # Assert
    assert np.isfinite(match.ball_xy[0]).all()
    assert np.isnan(match.ball_xy[2]).all()
