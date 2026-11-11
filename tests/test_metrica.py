"""Tests for the Metrica sample-game loader: pure label rules first, then the real data if present."""
import contextlib
import io
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mtah.external import metrica
from mtah.external.metrica import (
    DEFAULT_DATA_DIR,
    EVENT_COLUMNS,
    GAME_IDS,
    RAW_FILES,
    labels_from_events,
)
from mtah.schema import A, D, FPS, H, Match

REAL_DATA_PRESENT = os.path.isdir(DEFAULT_DATA_DIR)


def _events(*rows):
    """Event table rows as (team, type, subtype, period, start_frame, end_frame)."""
    return pd.DataFrame(list(rows), columns=EVENT_COLUMNS)


def test_frames_before_first_event_of_period_are_dead():
    # Arrange
    events = _events(
        ("Home", "SET PIECE", "KICK OFF", 1, 5, 5),
        ("Home", "PASS", "", 1, 6, 8),
    )

    # Act
    labels = labels_from_events(events, 12)

    # Assert
    assert labels[:4].tolist() == [D] * 4
    assert labels[4:7].tolist() == [H] * 3
    assert labels[7:].tolist() == [D] * 5


def test_pass_gives_possession_to_passing_team():
    # Arrange
    events = _events(
        ("Home", "SET PIECE", "KICK OFF", 1, 1, 1),
        ("Away", "PASS", "", 1, 3, 4),
        ("Away", "BALL OUT", "", 1, 10, 12),
    )

    # Act
    labels = labels_from_events(events, 15)

    # Assert
    assert labels[:2].tolist() == [H, H]
    assert labels[2:11].tolist() == [A] * 9
    assert labels[11:].tolist() == [D] * 4


def test_ball_lost_keeps_losing_team_until_opponent_recovers():
    # Arrange
    events = _events(
        ("Home", "SET PIECE", "KICK OFF", 1, 1, 1),
        ("Home", "BALL LOST", "INTERCEPTION", 1, 5, 6),
        ("Away", "RECOVERY", "INTERCEPTION", 1, 9, 9),
        ("Away", "BALL OUT", "", 1, 12, 12),
    )

    # Act
    labels = labels_from_events(events, 14)

    # Assert
    assert labels[:8].tolist() == [H] * 8
    assert labels[8:11].tolist() == [A] * 3
    assert labels[11:].tolist() == [D] * 3


def test_challenge_does_not_change_possession():
    # Arrange
    events = _events(
        ("Home", "SET PIECE", "KICK OFF", 1, 1, 1),
        ("Away", "CHALLENGE", "AERIAL-WON", 1, 5, 5),
        ("Home", "BALL OUT", "", 1, 9, 9),
    )

    # Act
    labels = labels_from_events(events, 10)

    # Assert
    assert labels[:8].tolist() == [H] * 8
    assert labels[8:].tolist() == [D] * 2


def test_ball_out_stops_play_and_pass_while_dead_does_not_revive():
    # Arrange
    events = _events(
        ("Home", "SET PIECE", "KICK OFF", 1, 1, 1),
        ("Home", "BALL OUT", "", 1, 4, 6),
        ("Away", "PASS", "", 1, 8, 9),
        ("Away", "SET PIECE", "THROW IN", 1, 12, 12),
        ("Away", "BALL OUT", "", 1, 15, 15),
    )

    # Act
    labels = labels_from_events(events, 18)

    # Assert
    assert labels[:5].tolist() == [H] * 5
    assert labels[5:11].tolist() == [D] * 6
    assert labels[11:14].tolist() == [A] * 3
    assert labels[14:].tolist() == [D] * 4


@pytest.mark.parametrize(
    ("subtype", "dead_index"),
    [("OFF TARGET-OUT", 4), ("ON TARGET-GOAL", 4), ("ON TARGET-SAVED", 8)],
)
def test_shot_ending_out_or_in_goal_stops_play(subtype, dead_index):
    # Arrange
    events = _events(
        ("Home", "SET PIECE", "KICK OFF", 1, 1, 1),
        ("Home", "SHOT", subtype, 1, 5, 5),
        ("Away", "BALL OUT", "", 1, 9, 9),
    )

    # Act
    labels = labels_from_events(events, 10)

    # Assert
    assert labels[:dead_index].tolist() == [H] * dead_index
    assert labels[dead_index:].tolist() == [D] * (10 - dead_index)


def test_offside_stops_play_for_any_event_type():
    # Arrange
    events = _events(
        ("Home", "SET PIECE", "KICK OFF", 1, 1, 1),
        ("Away", "PASS", "OFFSIDE", 1, 4, 6),
    )

    # Act
    labels = labels_from_events(events, 8)

    # Assert
    assert labels[:3].tolist() == [H] * 3
    assert labels[3:5].tolist() == [A] * 2
    assert labels[5:].tolist() == [D] * 3


def test_period_end_is_dead_until_next_period_set_piece():
    # Arrange
    events = _events(
        ("Home", "SET PIECE", "KICK OFF", 1, 1, 1),
        ("Home", "PASS", "", 1, 3, 5),
        ("Away", "SET PIECE", "KICK OFF", 2, 10, 10),
        ("Away", "BALL OUT", "", 2, 12, 12),
    )

    # Act
    labels = labels_from_events(events, 14)

    # Assert
    assert labels[:4].tolist() == [H] * 4
    assert labels[4:9].tolist() == [D] * 5
    assert labels[9:11].tolist() == [A] * 2
    assert labels[11:].tolist() == [D] * 3


def test_no_events_gives_all_dead_int8_labels_of_length_n_frames():
    # Arrange
    events = pd.DataFrame(columns=EVENT_COLUMNS)

    # Act
    labels = labels_from_events(events, 7)

    # Assert
    assert labels.dtype == np.int8
    assert labels.tolist() == [D] * 7


def test_rejects_unknown_team_name():
    # Arrange
    events = _events(("Referee", "PASS", "", 1, 1, 1))

    # Act / Assert
    with pytest.raises(ValueError, match="Referee"):
        labels_from_events(events, 5)


def test_rejects_event_table_missing_a_required_column():
    # Arrange
    events = _events(("Home", "PASS", "", 1, 1, 1)).drop(columns=["Period"])

    # Act / Assert
    with pytest.raises(ValueError, match="Period"):
        labels_from_events(events, 5)


def test_json_event_row_maps_team_ids_and_list_subtypes():
    # Arrange
    event = {
        "team": {"id": "FIFATMB"},
        "type": {"name": "SHOT"},
        "subtypes": [{"name": "OFF TARGET"}, {"name": "OUT"}],
        "period": 2,
        "start": {"frame": 70000},
        "end": {"frame": 70010},
    }
    team_names = {"FIFATMA": "Home", "FIFATMB": "Away"}

    # Act
    row = metrica._json_event_row(event, team_names)

    # Assert
    assert row == ("Away", "SHOT", "OFF TARGET|OUT", 2, 70000, 70010)


def test_download_skips_files_already_present(tmp_path, monkeypatch):
    # Arrange
    fetched = []
    monkeypatch.setattr(metrica, "_fetch", lambda url, target: fetched.append(target))
    existing = os.path.join(tmp_path, os.path.basename(RAW_FILES["metrica_1"]["events"]))
    open(existing, "w").close()
    total_files = sum(len(files) for files in RAW_FILES.values())

    # Act
    metrica.download(str(tmp_path))

    # Assert
    assert existing not in fetched
    assert len(fetched) == total_files - 1


def test_game_ids_match_raw_file_table():
    # Arrange / Act / Assert
    assert GAME_IDS == tuple(RAW_FILES)


@pytest.mark.skipif(not REAL_DATA_PRESENT, reason="external_data/metrica not downloaded")
def test_metrica_1_loads_with_shapes_fps_length_and_labels():
    # Arrange / Act
    match = metrica.load("metrica_1")

    # Assert
    n_frames = match.n_frames
    assert match.home_xy.shape == (n_frames, match.home_xy.shape[1], 2)
    assert match.away_xy.shape[0] == n_frames
    assert match.ball_xy.shape == (n_frames, 2)
    assert 5700 < n_frames / FPS < 5900
    assert set(np.unique(match.labels).tolist()) <= {H, A, D}
    dead_share = float(np.mean(match.labels == D))
    assert 0.2 < dead_share < 0.6


def _events_with(*rows, period_none=False):
    events = _events(*rows)
    if period_none:
        events["Period"] = None
    return events


def test_labels_reject_negative_frame_count():
    # Arrange
    events = _events(("Home", "PASS", "", 1, 1, 1))

    # Act / Assert
    with pytest.raises(ValueError, match="non-negative"):
        labels_from_events(events, -1)


def test_labels_reject_events_without_period():
    # Arrange
    events = _events_with(("Home", "PASS", "", 1, 1, 1), period_none=True)

    # Act / Assert
    with pytest.raises(ValueError, match="Period"):
        labels_from_events(events, 5)


def test_labels_reject_start_frame_below_one():
    # Arrange
    events = _events(("Home", "PASS", "", 1, 0, 1))

    # Act / Assert
    with pytest.raises(ValueError, match="1-based"):
        labels_from_events(events, 5)


def test_words_of_subtypes_handles_missing_single_and_list_values():
    # Arrange / Act / Assert
    assert metrica._words_of_subtypes(None) == ""
    assert metrica._words_of_subtypes({"name": "OUT"}) == "OUT"
    assert metrica._words_of_subtypes([{"name": "OFF TARGET"}, {"name": "GOAL"}]) == "OFF TARGET|GOAL"


def test_json_event_row_rejects_team_id_that_is_neither_home_nor_away():
    # Arrange
    event = {"team": {"id": "UNKNOWN"}, "type": {"name": "PASS"}, "subtypes": None,
             "period": 1, "start": {"frame": 1}, "end": {"frame": 2}}

    # Act / Assert
    with pytest.raises(ValueError, match="UNKNOWN"):
        metrica._json_event_row(event, {"FIFATMA": "Home"})


def test_place_scales_metres_and_leaves_missing_frames_nan():
    # Arrange
    frames = np.array([2])
    values = np.array([[0.5, 0.5]])

    # Act
    placed = metrica._place(frames, values, 3)

    # Assert
    assert placed[1] == pytest.approx([52.5, 34.0])
    assert np.isnan(placed[0]).all() and np.isnan(placed[2]).all()


@pytest.mark.parametrize("frames", [[1, 1], [0, 2]])
def test_place_rejects_duplicate_or_zero_based_frame_numbers(frames):
    # Arrange
    values = np.zeros((len(frames), 2))

    # Act / Assert
    with pytest.raises(ValueError, match="1-based"):
        metrica._place(np.array(frames), values, 3)


def test_blocks_span_each_period_from_its_first_to_its_last_frame():
    # Arrange
    frames = np.array([1, 2, 3, 4, 5])
    periods = np.array([1, 1, 2, 2, 2])

    # Act
    blocks = metrica._blocks(frames, periods)

    # Assert
    assert blocks == ((0, 2), (2, 5))


def test_ground_team_rejects_two_teams_on_the_same_ground():
    # Arrange
    teams = [_team("T1", "home", ["P1"]), _team("T3", "home", ["P3"])]

    # Act / Assert
    with pytest.raises(ValueError, match="home"):
        metrica._ground_team(teams, "home")


def test_read_csv_tracking_rejects_file_without_ball_column(tmp_path):
    # Arrange
    path = _tracking_csv(tmp_path / "home.csv", [1], [1], [(0.5, 0.5)], ball_xy=None)

    # Act / Assert
    with pytest.raises(ValueError, match="Ball"):
        metrica._read_csv_tracking(path)


def _tracking_csv(path, frames, periods, player_xy, ball_xy=None):
    """Metrica-style CSV: two info rows, then a header whose unnamed columns hold the y values."""
    header = "Frame,Period,Time [s],Player1,"
    if ball_xy is not None:
        header += ",Ball,"
    lines = ["Teams,,", "Players,,", header]
    for k, frame in enumerate(frames):
        values = [frame, periods[k], frame / FPS, *player_xy[k]]
        if ball_xy is not None:
            values += list(ball_xy[k])
        lines.append(",".join(str(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_parse_csv_game_builds_match_from_tracking_and_events(tmp_path):
    # Arrange
    frames, periods = [1, 2, 3, 4], [1, 1, 1, 1]
    names = RAW_FILES["metrica_1"]
    _tracking_csv(tmp_path / os.path.basename(names["home"]), frames, periods,
                  [(0.5, 0.5)] * 4, ball_xy=[(0.25, 0.5)] * 4)
    _tracking_csv(tmp_path / os.path.basename(names["away"]), frames, periods,
                  [(0.1, 0.1)] * 4, ball_xy=[(0.25, 0.5)] * 4)
    pd.DataFrame([("Home", "SET PIECE", "KICK OFF", 1, 1, 1),
                  ("Home", "BALL OUT", "", 1, 4, 4)],
                 columns=EVENT_COLUMNS).to_csv(tmp_path / os.path.basename(names["events"]), index=False)

    # Act
    match = metrica._parse_game("metrica_1", str(tmp_path))

    # Assert
    assert match.home_xy.shape == (4, 1, 2)
    assert match.home_xy[0, 0] == pytest.approx([52.5, 34.0])
    assert match.away_xy[0, 0] == pytest.approx([10.5, 6.8], abs=1e-4)
    assert match.ball_xy[0] == pytest.approx([26.25, 34.0])
    assert match.blocks == ((0, 4),)
    assert match.labels[0] == H and match.labels[-1] == D


@dataclass(frozen=True)
class _Player:
    player_id: str


def _team(team_id, ground, player_ids):
    return SimpleNamespace(team_id=team_id, ground=SimpleNamespace(value=ground),
                           players=[_Player(pid) for pid in player_ids])


def _record(frame_id, period, player_xy, ball_xy):
    players_data = {_Player(pid): SimpleNamespace(coordinates=SimpleNamespace(x=x, y=y))
                    for pid, (x, y) in player_xy.items()}
    ball = None if ball_xy is None else SimpleNamespace(x=ball_xy[0], y=ball_xy[1])
    return SimpleNamespace(frame_id=frame_id,
                           period=None if period is None else SimpleNamespace(id=period),
                           players_data=players_data, ball_coordinates=ball)


def _fake_dataset(teams, records):
    return SimpleNamespace(metadata=SimpleNamespace(teams=teams), records=records)


def test_parse_epts_game_places_players_ball_and_periods(tmp_path, monkeypatch):
    # Arrange
    teams = [_team("T1", "home", ["P1"]), _team("T2", "away", ["P2"])]
    records = [
        _record(1, 1, {"P1": (0.5, 0.5), "P2": (0.25, 0.25)}, (0.1, 0.2)),
        _record(2, 1, {"P1": (0.5, 0.5)}, None),
        _record(3, None, {}, None),
    ]
    dataset = _fake_dataset(teams, records)
    events = {"data": [{"team": {"id": "T1"}, "type": {"name": "SET PIECE"}, "subtypes": None,
                        "period": 1, "start": {"frame": 1}, "end": {"frame": 1}}]}
    (tmp_path / "Sample_Game_3_events.json").write_text(json.dumps(events), encoding="utf-8")
    monkeypatch.setattr(metrica.kloppy_metrica, "load_tracking_epts",
                        lambda meta_data, raw_data: dataset)

    # Act
    match = metrica._parse_game("metrica_3", str(tmp_path))

    # Assert
    assert match.home_xy.shape == (3, 1, 2)
    assert match.home_xy[0, 0] == pytest.approx([52.5, 34.0])
    assert match.away_xy[0, 0] == pytest.approx([26.25, 17.0])
    assert np.isnan(match.away_xy[1, 0]).all()
    assert np.isnan(match.ball_xy[1]).all()
    assert np.isnan(match.home_xy[2]).all()
    assert match.blocks == ((0, 2),)
    assert match.labels[0] == H


def test_epts_events_rejects_event_team_missing_from_metadata(tmp_path):
    # Arrange
    teams = [_team("T1", "home", ["P1"]), _team("T2", "away", ["P2"])]
    dataset = _fake_dataset(teams, [])
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"data": [{
        "team": {"id": "GHOST"}, "type": {"name": "PASS"}, "subtypes": None,
        "period": 1, "start": {"frame": 1}, "end": {"frame": 1}}]}), encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match="GHOST"):
        metrica._epts_events(str(path), dataset)


def test_load_rejects_unknown_game_id_before_touching_disk(tmp_path):
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="unknown Metrica game"):
        metrica.load("metrica_9", str(tmp_path))


def test_load_reads_existing_npz_cache_without_parsing(tmp_path):
    # Arrange
    match = Match(game_id="metrica_1", home_xy=np.zeros((2, 1, 2), np.float32),
                  away_xy=np.ones((2, 1, 2), np.float32), ball_xy=np.zeros((2, 2), np.float32),
                  blocks=((0, 2),), labels=np.array([H, A], dtype=np.int8))
    metrica._write_cache(match, str(tmp_path / "metrica_1.npz"))

    # Act
    loaded = metrica.load("metrica_1", str(tmp_path))

    # Assert
    assert loaded.game_id == "metrica_1"
    assert loaded.blocks == ((0, 2),)
    assert loaded.labels.tolist() == [H, A]
    assert np.array_equal(loaded.away_xy, match.away_xy)


def test_load_parses_and_caches_when_no_npz_present(tmp_path, monkeypatch):
    # Arrange
    match = Match(game_id="metrica_2", home_xy=np.zeros((1, 1, 2), np.float32),
                  away_xy=np.zeros((1, 1, 2), np.float32), ball_xy=np.zeros((1, 2), np.float32),
                  blocks=((0, 1),), labels=np.array([D], dtype=np.int8))
    monkeypatch.setattr(metrica, "_fetch_missing", lambda game_id, directory: None)
    monkeypatch.setattr(metrica, "_parse_game", lambda game_id, directory: match)

    # Act
    loaded = metrica.load("metrica_2", str(tmp_path))

    # Assert
    assert loaded is match
    assert (tmp_path / "metrica_2.npz").exists()


@pytest.mark.parametrize(("game_id", "parser"), [("metrica_1", "csv"), ("metrica_3", "epts")])
def test_parse_game_dispatches_on_game_format(game_id, parser, monkeypatch):
    # Arrange
    calls = []
    monkeypatch.setattr(metrica, "_parse_csv_game", lambda gid, files: calls.append(("csv", gid)))
    monkeypatch.setattr(metrica, "_parse_epts_game", lambda gid, files: calls.append(("epts", gid)))

    # Act
    metrica._parse_game(game_id, "unused_dir")

    # Assert
    assert calls == [(parser, game_id)]


def test_fetch_writes_response_body_through_part_file(tmp_path, monkeypatch):
    # Arrange
    target = str(tmp_path / "events.json")
    monkeypatch.setattr(metrica.urllib.request, "urlopen",
                        lambda url, timeout: contextlib.nullcontext(io.BytesIO(b"payload")))

    # Act
    metrica._fetch("https://example.test/events.json", target)

    # Assert
    assert open(target, "rb").read() == b"payload"
    assert not os.path.exists(target + ".part")
