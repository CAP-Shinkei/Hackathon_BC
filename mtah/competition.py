"""The competition match (FC Barcelona vs Athletic Club, tactical recording): constants and loaders.

Constants are the organisers' published facts (official repo DATA.md and games.py), not
measurements we made on the match. Data files are found under MTAH_DATA_DIR (default: ../data next
to this repository), with the names the organisers ship.

The rules forbid sending this data to hosted AI services: code that touches it prints aggregates only.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
DATA = os.environ.get("MTAH_DATA_DIR", DEFAULT_DATA)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "cache")

N_FRAMES = 199319                      # rows of a submission, DATA.md
FPS = 25
PITCH = (105.3, 68.0)                  # metres, games.py ("measured from the corrected tracking")
GRADED = [(11185, 90652), (110808, 190198)]  # video frames per half, stop exclusive, DATA.md
HOME_NAME, AWAY_NAME = "Barcelona", "Atletic Club"  # Smart Tagging Team values, DATA.md
HOME, AWAY = 0, 1


def _path(suffix):
    return os.path.join(DATA, "tactical", "BAR-ATH_tactical" + suffix)


def _local(tag):
    return tag.split("}")[-1]


def load_tracking(use_cache=True):
    """(ball, players, teams) in the layout of mtah.evidence, indexed by video frame
    (ATD frame_id - 1). Track slots keep the team the metadata assigns them."""
    cache = os.path.join(CACHE, "bar_ath_atd.npz")
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        return z["ball"], z["players"], z["teams"]

    root = ET.parse(_path("_atd.xml")).getroot()
    team_ids = [el.get("id") for el in root.iter() if _local(el.tag) == "Team"]
    slot_team = np.array([HOME if el.get("teamId") == team_ids[0] else AWAY
                          for el in root.iter() if _local(el.tag) == "Player"], dtype=np.int8)

    ball = np.full((N_FRAMES, 2), np.nan)
    players = np.full((N_FRAMES, len(slot_team), 2), np.nan)
    with open(_path("_atd.txt")) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            ball[f] = [float(v) for v in parts[2].split(",")[:2]]
            for j, group in enumerate(parts[1].split(";")):
                players[f, j] = [float(v) for v in group.split(",")[:2]]
    teams = np.broadcast_to(slot_team, players.shape[:2]).copy()
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cache, ball=ball, players=players, teams=teams)
    return ball, players, teams


def load_smart_tagging():
    """[{code, start, end, tags}] with start/end as video frames (round(seconds * 25), DATA.md)."""
    out = []
    for inst in ET.parse(_path("_smart_tagging.xml")).getroot().iter("instance"):
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * FPS)),
            "end": int(round(float(inst.findtext("end")) * FPS)),
            "tags": {(lab.findtext("group") or "").strip().lower():
                     (lab.findtext("text") or "").strip().lower()
                     for lab in inst.findall("label")},
        })
    return out
