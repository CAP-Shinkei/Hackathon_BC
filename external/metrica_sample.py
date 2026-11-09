"""Loaders for the public Metrica Sports sample games (github.com/metrica-sports/sample-data).

These three anonymised matches are our external data: tracking synchronised with event data, so
frame-level possession labels can be derived for them and used to tune and validate the
possession model without touching the competition match.

Frame convention matches the competition: a tracking/event frame number f (counted from 1) is
array index f - 1.

Games 1 and 2 ship as Metrica's legacy CSV (one tracking file per team, events CSV); Game 3 as
FIFA/EPTS (metadata XML + raw text, the same family as the competition ATD) with events JSON.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                            "external_data", "metrica_sample_data", "data")
ROOT = os.environ.get("METRICA_SAMPLE_DIR", DEFAULT_ROOT)
GAMES = (1, 2, 3)
PITCH = (105.0, 68.0)  # metres, stated in the sample-data README
HOME, AWAY = 0, 1


def _game_dir(game):
    return os.path.join(ROOT, "Sample_Game_{}".format(game))


def load_events(game):
    """Events as a DataFrame: team (HOME/AWAY), type, subtype ('' or 'A-B-C'), period,
    start, end (frames, counted from 1), sorted by start frame then original order."""
    if game in (1, 2):
        raw = pd.read_csv(os.path.join(_game_dir(game),
                                       "Sample_Game_{}_RawEventsData.csv".format(game)))
        ev = pd.DataFrame({
            "team": np.where(raw["Team"] == "Home", HOME, AWAY),
            "type": raw["Type"].str.upper(),
            "subtype": raw["Subtype"].fillna("").str.upper(),
            "period": raw["Period"].astype(int),
            "start": raw["Start Frame"].astype(int),
            "end": raw["End Frame"].astype(int),
        })
    else:
        home_id = _epts_team_ids(game)[0]
        with open(os.path.join(_game_dir(game), "Sample_Game_3_events.json")) as handle:
            data = json.load(handle)["data"]
        rows = []
        for e in data:
            subs = e["subtypes"]
            if subs is None:
                subs = []
            elif isinstance(subs, dict):
                subs = [subs]
            rows.append({
                "team": HOME if e["team"]["id"] == home_id else AWAY,
                "type": e["type"]["name"].upper(),
                "subtype": "-".join(s["name"] for s in subs).upper(),
                "period": int(e["period"]),
                "start": int(e["start"]["frame"]),
                "end": int(e["end"]["frame"]),
            })
        ev = pd.DataFrame(rows)
    # Legacy files write End Frame 0 for instantaneous events (e.g. the kick-off).
    ev["end"] = np.maximum(ev["end"], ev["start"])
    return ev.reset_index(drop=True).sort_values(["start"], kind="stable").reset_index(drop=True)


def _epts_metadata(game):
    return ET.parse(os.path.join(_game_dir(game),
                                 "Sample_Game_{}_metadata.xml".format(game))).getroot()


def _local(tag):
    return tag.split("}")[-1]


def _epts_team_ids(game):
    """Team ids in file order; the first is the home side (FIFA/EPTS convention)."""
    return [el.get("id") for el in _epts_metadata(game).iter() if _local(el.tag) == "Team"]


def load_tracking(game):
    """(ball, players, teams):
    ball    (n, 2) normalised x, y; NaN where the ball is not tracked
    players (n, k, 2) normalised x, y per player column; NaN where absent
    teams   (n, k) HOME/AWAY of each player column on each frame (-1 where unused)
    """
    if game in (1, 2):
        blocks, team_cols = [], []
        ball = None
        for side, code in (("Home", HOME), ("Away", AWAY)):
            path = os.path.join(_game_dir(game),
                                "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, side))
            arr = pd.read_csv(path, skiprows=2).iloc[:, 3:].to_numpy(dtype=float)
            if ball is None:
                ball = arr[:, -2:]
            xy = arr[:, :-2].reshape(len(arr), -1, 2)
            blocks.append(xy)
            team_cols += [code] * xy.shape[1]
        players = np.concatenate(blocks, axis=1)
        teams = np.broadcast_to(np.array(team_cols, dtype=np.int8), players.shape[:2]).copy()
        return ball, players, teams

    root = _epts_metadata(game)
    home_id = _epts_team_ids(game)[0]
    team_of_player = {el.get("id"): (HOME if el.get("teamId") == home_id else AWAY)
                      for el in root.iter() if _local(el.tag) == "Player"}
    player_of_channel = {el.get("id"): el.get("playerId")
                         for el in root.iter() if _local(el.tag) == "PlayerChannel"}
    specs = []
    for spec in (el for el in root.iter() if _local(el.tag) == "DataFormatSpecification"):
        refs = [el.get("playerChannelId") for el in spec.iter()
                if _local(el.tag) == "PlayerChannelRef"][0::2]  # x channel of each (x, y) pair
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      [team_of_player[player_of_channel[r]] for r in refs]))

    frames, ball_rows, player_rows = [], [], []
    with open(os.path.join(_game_dir(game), "Sample_Game_3_tracking.txt")) as handle:
        for line in handle:
            fid, plist, b = line.rstrip("\n").split(":")
            frames.append(int(fid))
            ball_rows.append([float(v) for v in b.split(",")[:2]])
            player_rows.append([[float(v) for v in g.split(",")[:2]] for g in plist.split(";")])
    n = max(frames)
    k = max(len(r) for r in player_rows)
    ball = np.full((n, 2), np.nan)
    players = np.full((n, k, 2), np.nan)
    teams = np.full((n, k), -1, dtype=np.int8)
    for fid, b, p in zip(frames, ball_rows, player_rows):
        ball[fid - 1] = b
        players[fid - 1, :len(p)] = p
    for start, end, team_list in specs:
        teams[start - 1:end, :len(team_list)] = team_list
    return ball, players, teams
