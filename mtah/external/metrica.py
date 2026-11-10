"""Metrica Sports public sample games (Sample_Game_1..3) as Match objects with possession labels.

Coordinates: the raw 0..1 values are scaled to metres (x * 105, y * 68). Frames are 1-based at
25 fps and continue across periods; frame f is array index f - 1. A block is the frame span of
one period. Games 1 and 2 are CSV; game 3 is FIFA EPTS tracking (read with kloppy) plus JSON events.

Label rules (H, A, D), applied in frame order over the event table:
- Possession: a PASS, SHOT, RECOVERY, BALL LOST or SET PIECE event gives the ball to its team.
  BALL LOST names the team that lost it, so that team keeps the ball until the opponent's next
  on-ball event. CHALLENGE, CARRY, CARD and other events never change possession.
- Dead ball: the ball is dead (D) from the end frame of an event that stops play: BALL OUT,
  FAULT RECEIVED, any event with an OFFSIDE subtype, and a SHOT whose subtype ends OUT or in a
  GOAL. The end frame of a period (its last event end) also stops play.
- Dead ball lasts until the start frame of the next SET PIECE, which revives play for its team.
  On-ball events while dead update the owner but do not revive play.
- Frames before the first event of a period are D. Dead-ball state carries over the period
  boundary, so every period starts dead until its kick-off.
"""
import json
import os
import re
import shutil
import urllib.request
from collections import namedtuple

import numpy as np
import pandas as pd
from kloppy import metrica as kloppy_metrica

from mtah.schema import A, D, H, Match

GAME_IDS = ("metrica_1", "metrica_2", "metrica_3")
PITCH = (105.0, 68.0)
RAW_BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "external_data", "metrica")
DOWNLOAD_TIMEOUT_S = 120
CSV_HEADER_ROWS = 2  # two rows of team and jersey info sit above the column names
RAW_FILES = {
    "metrica_1": {
        "events": "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
        "home": "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "away": "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    },
    "metrica_2": {
        "events": "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
        "home": "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "away": "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    },
    "metrica_3": {
        "events": "Sample_Game_3/Sample_Game_3_events.json",
        "metadata": "Sample_Game_3/Sample_Game_3_metadata.xml",
        "tracking": "Sample_Game_3/Sample_Game_3_tracking.txt",
    },
}
EVENT_COLUMNS = ("Team", "Type", "Subtype", "Period", "Start Frame", "End Frame")
TEAM_CODES = {"Home": H, "Away": A}
SET_PIECE = "SET PIECE"
ON_BALL_TYPES = frozenset(("PASS", "SHOT", "RECOVERY", "BALL LOST"))
STOP_TYPES = frozenset(("BALL OUT", "FAULT RECEIVED"))
SHOT = "SHOT"
SHOT_STOP_WORDS = frozenset(("OUT", "GOAL"))
OFFSIDE_WORD = "OFFSIDE"
PRIORITY_STOP, PRIORITY_CHANGE = 0, 1  # at one frame, a stop is applied before a revival
WORD_SPLIT = re.compile(r"[\s|\-]+")
TrackingRows = namedtuple("TrackingRows", "frames periods home away ball")
CsvTracking = namedtuple("CsvTracking", "frames periods players ball")


def download(dest_dir=None):
    """Fetch every raw file of the three games into dest_dir, skipping files already there."""
    directory = dest_dir or DEFAULT_DATA_DIR
    os.makedirs(directory, exist_ok=True)
    for game_id in GAME_IDS:
        _fetch_missing(game_id, directory)


def _fetch_missing(game_id, directory):
    for relative in RAW_FILES[game_id].values():
        target = os.path.join(directory, os.path.basename(relative))
        if not os.path.exists(target):
            _fetch(RAW_BASE_URL + relative, target)


def _fetch(url, target):
    partial = target + ".part"
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle)
    os.replace(partial, target)


def load(game_id, data_dir=None):
    """The game as a Match with labels. Raw files are fetched when missing; the result is cached."""
    _check_game_id(game_id)
    directory = data_dir or DEFAULT_DATA_DIR
    cache = os.path.join(directory, game_id + ".npz")
    if os.path.exists(cache):
        return _read_cache(game_id, cache)
    _fetch_missing(game_id, directory)
    match = _parse_game(game_id, directory)
    _write_cache(match, cache)
    return match


def _check_game_id(game_id):
    if game_id not in GAME_IDS:
        raise ValueError("unknown Metrica game {!r}; expected one of {}".format(game_id, GAME_IDS))


def labels_from_events(events, n_frames):
    """H/A/D label per frame (int8) from an event table with the EVENT_COLUMNS. See module rules."""
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative, got {}".format(n_frames))
    table = _normalise(events)
    frames, owners, dead = _state_timeline(_changes(table))
    return _paint(frames, owners, dead, n_frames)


def _normalise(events):
    """The event table with team codes, frame indices and word sets, validated."""
    missing = [column for column in EVENT_COLUMNS if column not in events.columns]
    if missing:
        raise ValueError("event table lacks columns {}".format(missing))
    unknown = sorted(set(events["Team"]) - set(TEAM_CODES))
    if unknown:
        raise ValueError("unknown team names in events: {}".format(unknown))
    if events["Period"].isna().any():
        raise ValueError("events without a Period")
    starts = pd.to_numeric(events["Start Frame"], errors="coerce")
    if starts.isna().any() or (starts < 1).any():
        raise ValueError("Start Frame must be present and 1-based")
    ends = pd.to_numeric(events["End Frame"], errors="coerce").fillna(starts)
    ends = np.maximum(ends, starts)
    return pd.DataFrame({
        "team": events["Team"].map(TEAM_CODES).astype(int),
        "type": events["Type"].fillna("").astype(str).str.strip(),
        "words": events["Subtype"].fillna("").astype(str).map(_words),
        "period": events["Period"],
        "start": starts.astype(int) - 1,
        "end": ends.astype(int) - 1,
    })


def _words(text):
    return frozenset(word for word in WORD_SPLIT.split(str(text)) if word)


def _changes(table):
    """Every (frame_index, priority, owner_or_None, dead_or_None) update in the event table."""
    changes = []
    for row in table.itertuples(index=False):
        changes.extend(_event_changes(row))
    for period_end in table.groupby("period")["end"].max():
        changes.append((int(period_end), PRIORITY_STOP, None, True))
    return changes


def _event_changes(row):
    changes = []
    if row.type == SET_PIECE:
        changes.append((row.start, PRIORITY_CHANGE, row.team, False))
    elif row.type in ON_BALL_TYPES:
        changes.append((row.start, PRIORITY_CHANGE, row.team, None))
    if _stops_play(row):
        changes.append((row.end, PRIORITY_STOP, None, True))
    return changes


def _stops_play(row):
    if row.type in STOP_TYPES or OFFSIDE_WORD in row.words:
        return True
    return row.type == SHOT and bool(row.words & SHOT_STOP_WORDS)


def _state_timeline(changes):
    """Frame indices where the state changes, with the owner and dead flag from each one on."""
    ordered = sorted(changes, key=lambda change: (change[0], change[1]))
    frames, owners, dead_flags = [0], [D], [True]
    owner, dead = D, True
    for frame, _, new_owner, new_dead in ordered:
        owner = owner if new_owner is None else new_owner
        dead = dead if new_dead is None else new_dead
        frames.append(frame)
        owners.append(owner)
        dead_flags.append(dead)
    return np.array(frames), np.array(owners), np.array(dead_flags)


def _paint(frames, owners, dead_flags, n_frames):
    positions = np.searchsorted(frames, np.arange(n_frames), side="right") - 1
    owner = owners[positions]
    return np.where(dead_flags[positions] | (owner == D), D, owner).astype(np.int8)


def _words_of_subtypes(subtypes):
    if subtypes is None:
        return ""
    if isinstance(subtypes, dict):
        return subtypes["name"]
    return "|".join(item["name"] for item in subtypes)


def _json_event_row(event, team_names):
    """One FIFA-style JSON event as an EVENT_COLUMNS row. team_names maps team ids to Home/Away."""
    team = team_names.get(event["team"]["id"])
    if team is None:
        raise ValueError("event team {!r} is neither home nor away".format(event["team"]["id"]))
    return (team, event["type"]["name"], _words_of_subtypes(event["subtypes"]),
            event["period"], event["start"]["frame"], event["end"]["frame"])


def _read_cache(game_id, path):
    with np.load(path) as data:
        blocks = tuple((int(start), int(stop)) for start, stop in data["blocks"])
        return Match(game_id=game_id, home_xy=data["home_xy"], away_xy=data["away_xy"],
                     ball_xy=data["ball_xy"], blocks=blocks, labels=data["labels"])


def _write_cache(match, path):
    partial = path + ".tmp.npz"
    np.savez(partial, home_xy=match.home_xy, away_xy=match.away_xy, ball_xy=match.ball_xy,
             blocks=np.asarray(match.blocks, dtype=np.int64), labels=match.labels)
    os.replace(partial, path)


def _parse_game(game_id, directory):
    files = {role: os.path.join(directory, os.path.basename(relative))
             for role, relative in RAW_FILES[game_id].items()}
    if game_id == "metrica_3":
        return _parse_epts_game(game_id, files)
    return _parse_csv_game(game_id, files)


def _parse_csv_game(game_id, files):
    home = _read_csv_tracking(files["home"])
    away = _read_csv_tracking(files["away"])
    n_frames = int(max(home.frames.max(), away.frames.max()))
    labels = labels_from_events(pd.read_csv(files["events"]), n_frames)
    return Match(game_id=game_id,
                 home_xy=_place(home.frames, home.players, n_frames),
                 away_xy=_place(away.frames, away.players, n_frames),
                 ball_xy=_place(home.frames, home.ball, n_frames),
                 blocks=_blocks(home.frames, home.periods), labels=labels)


def _read_csv_tracking(path):
    table = pd.read_csv(path, skiprows=CSV_HEADER_ROWS)
    names = [str(name) for name in table.columns]
    values = table.to_numpy(dtype=np.float64)
    player_pairs = [(i, i + 1) for i, name in enumerate(names) if name.startswith("Player")]
    ball_pairs = [(i, i + 1) for i, name in enumerate(names) if name == "Ball"]
    if len(ball_pairs) != 1 or not player_pairs:
        raise ValueError("{}: expected player columns and exactly one Ball column".format(path))
    return CsvTracking(
        frames=table["Frame"].to_numpy(dtype=np.int64),
        periods=table["Period"].to_numpy(dtype=np.int64),
        players=_pairs_to_xy(values, player_pairs),
        ball=_pairs_to_xy(values, ball_pairs)[:, 0, :],
    )


def _pairs_to_xy(values, pairs):
    return np.stack([values[:, [i, j]] for i, j in pairs], axis=1)


def _place(frames, values, n_frames):
    """Scale values to metres and put row k at array index frames[k] - 1; missing frames are NaN."""
    index = np.asarray(frames, dtype=np.int64) - 1
    if index.min() < 0 or len(np.unique(index)) != len(index):
        raise ValueError("frame numbers must be unique and 1-based")
    out = np.full((n_frames,) + values.shape[1:], np.nan, dtype=np.float32)
    out[index] = values * np.asarray(PITCH, dtype=np.float32)
    return out


def _blocks(frames, periods):
    spans = []
    for period in np.unique(periods):
        in_period = frames[periods == period]
        spans.append((int(in_period.min()) - 1, int(in_period.max())))
    return tuple(sorted(spans))


def _parse_epts_game(game_id, files):
    dataset = kloppy_metrica.load_tracking_epts(meta_data=files["metadata"], raw_data=files["tracking"])
    rows = _epts_rows(dataset)
    n_frames = int(rows.frames.max())
    labels = labels_from_events(_epts_events(files["events"], dataset), n_frames)
    in_period = rows.periods >= 0
    return Match(game_id=game_id,
                 home_xy=_place(rows.frames, rows.home, n_frames),
                 away_xy=_place(rows.frames, rows.away, n_frames),
                 ball_xy=_place(rows.frames, rows.ball, n_frames),
                 blocks=_blocks(rows.frames[in_period], rows.periods[in_period]),
                 labels=labels)


def _epts_rows(dataset):
    """Per-record arrays from a kloppy EPTS dataset; records outside every period get period -1."""
    home_team = _ground_team(dataset.metadata.teams, "home")
    away_team = _ground_team(dataset.metadata.teams, "away")
    slots = _slot_map((home_team, away_team))
    records = dataset.records
    xy = [np.full((len(records), len(team.players), 2), np.nan) for team in (home_team, away_team)]
    ball = np.full((len(records), 2), np.nan)
    for row, frame in enumerate(records):
        for player, data in frame.players_data.items():
            side, slot = slots[player.player_id]
            xy[side][row, slot] = _point_xy(data.coordinates)
        ball[row] = _point_xy(frame.ball_coordinates)
    frames = np.array([frame.frame_id for frame in records], dtype=np.int64)
    periods = np.array([frame.period.id if frame.period is not None else -1 for frame in records],
                       dtype=np.int64)
    return TrackingRows(frames, periods, xy[0], xy[1], ball)


def _ground_team(teams, ground):
    matches = [team for team in teams if team.ground.value == ground]
    if len(matches) != 1:
        raise ValueError("expected exactly one {} team, found {}".format(ground, len(matches)))
    return matches[0]


def _slot_map(teams):
    """Player id -> (side, slot): side 0 is home, 1 is away; slot is the player's position in its team."""
    return {player.player_id: (side, slot)
            for side, team in enumerate(teams) for slot, player in enumerate(team.players)}


def _point_xy(point):
    if point is None:
        return (np.nan, np.nan)
    return (point.x, point.y)


def _epts_events(path, dataset):
    home_id = _ground_team(dataset.metadata.teams, "home").team_id
    away_id = _ground_team(dataset.metadata.teams, "away").team_id
    team_names = {home_id: "Home", away_id: "Away"}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)["data"]
    return pd.DataFrame([_json_event_row(event, team_names) for event in data], columns=EVENT_COLUMNS)
