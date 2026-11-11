"""Round trips and parser checks for the ATD reader and writer, on synthetic matches only."""
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from mtah.atd_export import export_atd, export_kickoff_tagging
from mtah.io_atd import load_atd, team_of_slots
from mtah.schema import Match

PITCH = (105.0, 68.0)
N_FRAMES = 50
BLOCKS = ((0, 30), (30, 50))
HOME_PLAYERS = 4
AWAY_PLAYERS = 3
ROUND_TRIP_TOLERANCE_PER_METRE = 1e-5
FPS = 25
KICKOFF_BLOCKS = ((100, 500), (1000, 1400))
SPEED_TEST_FRAMES = 200_000
SPEED_TEST_SLOTS = 46
SPEED_TEST_SECONDS = 90.0
SPEED_TEST_GROUP = "0.5,0.5,0.0"
SPEED_TEST_BALL = "0.25,0.25,0.0"


def make_match(n_frames=N_FRAMES, home_players=HOME_PLAYERS, away_players=AWAY_PLAYERS, seed=0):
    rng = np.random.default_rng(seed)
    home = rng.uniform(0.0, 1.0, (n_frames, home_players, 2)) * np.array(PITCH)
    away = rng.uniform(0.0, 1.0, (n_frames, away_players, 2)) * np.array(PITCH)
    ball = rng.uniform(0.0, 1.0, (n_frames, 2)) * np.array(PITCH)
    home[5:8, 1] = np.nan
    away[10:13, 2] = np.nan
    ball[20:25] = np.nan
    return Match(
        game_id="synthetic",
        home_xy=home.astype(np.float32),
        away_xy=away.astype(np.float32),
        ball_xy=ball.astype(np.float32),
        blocks=BLOCKS if n_frames == N_FRAMES else ((0, n_frames),),
        pitch=PITCH,
    )


def assert_within_tolerance(actual, expected):
    """Same NaN pattern, and every present value within 1e-5 of the pitch size on its axis."""
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    tolerance = np.broadcast_to(ROUND_TRIP_TOLERANCE_PER_METRE * np.array(PITCH), actual.shape)
    present = ~np.isnan(expected)
    assert np.all(np.abs(actual - expected)[present] <= tolerance[present])


def test_export_then_load_round_trips_positions_and_missing_values(tmp_path):
    match = make_match()
    meta_path, raw_path = export_atd(match, tmp_path, "synthetic")

    loaded = load_atd(meta_path, raw_path, n_frames=match.n_frames, pitch=PITCH,
                      blocks=BLOCKS, game_id="synthetic")

    assert meta_path.endswith("synthetic_atd.xml")
    assert raw_path.endswith("synthetic_atd.txt")
    assert loaded.blocks == BLOCKS
    assert loaded.labels is None
    assert_within_tolerance(loaded.home_xy, match.home_xy)
    assert_within_tolerance(loaded.away_xy, match.away_xy)
    assert_within_tolerance(loaded.ball_xy, match.ball_xy)


def test_team_of_slots_lists_home_slots_first_then_away(tmp_path):
    meta_path, _ = export_atd(make_match(), tmp_path, "synthetic")

    slots = team_of_slots(meta_path)

    assert slots.dtype == np.int8
    assert list(slots) == [0] * HOME_PLAYERS + [1] * AWAY_PLAYERS


def test_kickoff_tagging_has_one_instance_per_block_starting_at_block_start(tmp_path):
    out_path = export_kickoff_tagging(KICKOFF_BLOCKS, tmp_path / "tagging.xml", fps=FPS)

    instances = list(ET.parse(out_path).getroot().iter("instance"))

    starts = [round(float(instance.findtext("start")) * FPS) for instance in instances]
    assert starts == [start for start, _ in KICKOFF_BLOCKS]
    for instance in instances:
        assert instance.findtext("code").lower() == "set pieces"
        assert float(instance.findtext("end")) - float(instance.findtext("start")) == 2.0
        groups = {label.findtext("group"): label.findtext("text")
                  for label in instance.findall("label")}
        assert groups["Type"] == "Kick Off (Start)"
    halves = [{label.findtext("group"): label.findtext("text")
               for label in instance.findall("label")}["Half"] for instance in instances]
    assert halves == ["1st Half", "2nd Half"]


def test_load_atd_rejects_line_with_wrong_number_of_player_groups(tmp_path):
    meta_path, _ = export_atd(make_match(), tmp_path, "synthetic")
    bad_raw = tmp_path / "bad_atd.txt"
    bad_raw.write_text("1:NaN,NaN,NaN;0.5,0.5,0.0:NaN,NaN,NaN\n")

    with pytest.raises(ValueError, match="player groups"):
        load_atd(meta_path, bad_raw, n_frames=1, pitch=PITCH, blocks=((0, 1),))


def test_load_atd_rejects_frame_id_beyond_n_frames(tmp_path):
    match = make_match()
    meta_path, raw_path = export_atd(match, tmp_path, "synthetic")

    with pytest.raises(ValueError, match="frame"):
        load_atd(meta_path, raw_path, n_frames=match.n_frames - 1, pitch=PITCH, blocks=BLOCKS)


def test_load_atd_leaves_frames_absent_from_the_file_as_nan(tmp_path):
    match = make_match()
    meta_path, raw_path = export_atd(match, tmp_path, "synthetic")
    lines = Path(raw_path).read_text().splitlines(keepends=True)
    del lines[1]
    gappy_raw = tmp_path / "gappy_atd.txt"
    gappy_raw.write_text("".join(lines))

    loaded = load_atd(meta_path, gappy_raw, n_frames=match.n_frames, pitch=PITCH, blocks=BLOCKS)

    assert np.isnan(loaded.home_xy[1]).all()
    assert np.isnan(loaded.away_xy[1]).all()
    assert np.isnan(loaded.ball_xy[1]).all()
    assert_within_tolerance(loaded.home_xy[0], match.home_xy[0])


def test_load_atd_reads_200000_lines_of_46_slots_within_time_budget(tmp_path):
    meta_path, _ = export_atd(
        make_match(n_frames=1, home_players=SPEED_TEST_SLOTS // 2,
                   away_players=SPEED_TEST_SLOTS // 2),
        tmp_path, "speed")
    players = ";".join([SPEED_TEST_GROUP] * SPEED_TEST_SLOTS)
    raw_path = tmp_path / "speed_atd.txt"
    raw_path.write_text("".join(
        "{}:{}:{}\n".format(frame_id, players, SPEED_TEST_BALL)
        for frame_id in range(1, SPEED_TEST_FRAMES + 1)))

    started = time.perf_counter()
    loaded = load_atd(meta_path, raw_path, n_frames=SPEED_TEST_FRAMES, pitch=PITCH,
                      blocks=((0, SPEED_TEST_FRAMES),))
    elapsed = time.perf_counter() - started

    assert loaded.home_xy.shape == (SPEED_TEST_FRAMES, SPEED_TEST_SLOTS // 2, 2)
    assert elapsed < SPEED_TEST_SECONDS


def test_load_atd_warns_about_truncated_lines_but_not_blank_ones(tmp_path):
    match = make_match()
    meta_path, raw_path = export_atd(match, tmp_path, "synthetic")
    lines = Path(raw_path).read_text().splitlines(keepends=True)
    lines[1] = lines[1].rsplit(":", 1)[0] + "\n"
    truncated_raw = tmp_path / "truncated_atd.txt"
    truncated_raw.write_text("".join(lines) + "\n\n")

    with pytest.warns(UserWarning, match="1 non-empty lines"):
        loaded = load_atd(meta_path, truncated_raw, n_frames=match.n_frames, pitch=PITCH,
                          blocks=BLOCKS)

    assert np.isnan(loaded.ball_xy[1]).all()


@pytest.mark.parametrize("n_frames, pitch, message", [
    (0, PITCH, "n_frames must be positive"),
    (N_FRAMES, (105.0,), "two positive lengths"),
    (N_FRAMES, (105.0, 0.0), "two positive lengths"),
])
def test_load_atd_rejects_impossible_frame_counts_and_pitches(tmp_path, n_frames, pitch, message):
    meta_path, raw_path = export_atd(make_match(), tmp_path, "synthetic")

    with pytest.raises(ValueError, match=message):
        load_atd(meta_path, raw_path, n_frames=n_frames, pitch=pitch, blocks=BLOCKS)


GOOD_GROUP = "0.5,0.5,0.0"
GOOD_PLAYERS = ";".join([GOOD_GROUP] * (HOME_PLAYERS + AWAY_PLAYERS))
BAD_PLAYERS = ";".join([GOOD_GROUP] * (HOME_PLAYERS + AWAY_PLAYERS - 1) + ["0.5,oops,0.0"])


@pytest.mark.parametrize("line, message", [
    ("1:" + GOOD_PLAYERS + ":0.5,0.5", "values, expected"),
    ("1:" + BAD_PLAYERS + ":" + GOOD_GROUP, "non-numeric value"),
    ("one:" + GOOD_PLAYERS + ":" + GOOD_GROUP, "is not an integer"),
    ("0:" + GOOD_PLAYERS + ":" + GOOD_GROUP, "outside 1"),
])
def test_load_atd_names_the_line_of_a_malformed_frame_record(tmp_path, line, message):
    meta_path, _ = export_atd(make_match(), tmp_path, "synthetic")
    bad_raw = tmp_path / "bad_atd.txt"
    bad_raw.write_text(line + "\n")

    with pytest.raises(ValueError, match="line 1: .*" + message):
        load_atd(meta_path, bad_raw, n_frames=1, pitch=PITCH, blocks=((0, 1),))
