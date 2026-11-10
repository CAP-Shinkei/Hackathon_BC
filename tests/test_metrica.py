"""Tests for the Metrica sample-game loader: pure label rules first, then the real data if present."""
import os

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
from mtah.schema import A, D, FPS, H

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
