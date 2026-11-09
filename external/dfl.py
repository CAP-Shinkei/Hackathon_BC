"""Loader for the IDSSE / DFL open dataset (Bassek, Rein, Weber, Memmert, Scientific Data 12:195,
2025; figshare 10.6084/m9.figshare.28196177; CC BY 4.0, data (c) DFL).

Seven Bundesliga 2022/23 matches with 25 Hz TRACAB tracking. Every ball frame carries the DFL's own
per-frame BallPossession (1 home, 2 away) and BallStatus (0 dead, 1 alive): the same kind of
frame-level truth the challenge's ground truth is derived from, so no event-based reconstruction
is needed.

Output layout matches metrica_sample.load_tracking plus labels, on one timeline per match: first
half, a block of HALF_GAP X frames, second half. The X block keeps the halves apart for the scorer.
"""

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                            "external_data", "idsse_dfl")
ROOT = os.environ.get("IDSSE_DIR", DEFAULT_ROOT)
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
HALF_GAP = 250
H, A, D, X = 0, 1, 2, 3
HOME, AWAY = 0, 1
SECTIONS = ("firstHalf", "secondHalf")


def _file(kind, match):
    paths = glob.glob(os.path.join(ROOT, "DFL_{}_*DFL-MAT-{}.xml".format(kind, match)))
    if len(paths) != 1:
        raise FileNotFoundError("{} {}: found {}".format(kind, match, paths))
    return paths[0]


def team_roles(match):
    """{team_id: HOME/AWAY} from the match information file."""
    root = ET.parse(_file("02_01_matchinformation", match)).getroot()
    return {t.get("TeamId"): HOME if t.get("Role") == "home" else AWAY
            for t in root.iter("Team") if t.get("Role") in ("home", "guest")}


def load(match):
    """dict(name, ball, players, teams, gt, pitch). ball/players normalised 0-1 (origin at a
    corner) so the code written for the Metrica games applies unchanged; NaN where not tracked."""
    roles = team_roles(match)
    pitch = None
    halves = {s: {"ball": {}, "players": [], "status": {}, "poss": {}} for s in SECTIONS}
    current = None
    for event, el in ET.iterparse(_file("04_03_positions_raw_observed", match),
                                  events=("start", "end")):
        if event == "start":
            if el.tag == "FrameSet":
                team = el.get("TeamId")
                current = (el.get("GameSection"), team,
                           "ball" if team == "BALL" else roles.get(team))
                if current[2] not in (None, "ball"):
                    halves[current[0]]["players"].append((current[2], {}))
            continue
        if el.tag == "PitchSize":
            pitch = (float(el.get("X")), float(el.get("Y")))
        elif el.tag == "Frame" and current is not None and current[2] is not None:
            n = int(el.get("N"))
            xy = (float(el.get("X")), float(el.get("Y")))
            half = halves[current[0]]
            if current[2] == "ball":
                half["ball"][n] = xy
                half["status"][n] = int(el.get("BallStatus"))
                half["poss"][n] = int(el.get("BallPossession"))
            else:
                half["players"][-1][1][n] = xy
            el.clear()
        elif el.tag == "FrameSet":
            current = None
            el.clear()

    blocks = []
    for s in SECTIONS:
        half = halves[s]
        frames = sorted(set(half["status"]))
        lo, hi = frames[0], frames[-1] + 1
        n = hi - lo
        ball = np.full((n, 2), np.nan)
        gt = np.full(n, X, dtype=np.int8)
        for f in frames:
            ball[f - lo] = half["ball"][f]
            gt[f - lo] = D if half["status"][f] == 0 else (H if half["poss"][f] == 1 else A)
        k = len(half["players"])
        players = np.full((n, k, 2), np.nan)
        teams = np.full((n, k), -1, dtype=np.int8)
        for j, (team, frames_xy) in enumerate(half["players"]):
            idx = np.array([f - lo for f in frames_xy if lo <= f < hi], dtype=int)
            if len(idx):
                players[idx, j] = [frames_xy[f + lo] for f in idx]
                teams[idx, j] = team
        blocks.append((ball, players, teams, gt))

    k = max(b[1].shape[1] for b in blocks)
    pad_p = lambda p: np.concatenate([p, np.full((len(p), k - p.shape[1], 2), np.nan)], axis=1)
    pad_t = lambda t: np.concatenate([t, np.full((len(t), k - t.shape[1]), -1, np.int8)], axis=1)
    gap_ball = np.full((HALF_GAP, 2), np.nan)
    ball = np.concatenate([blocks[0][0], gap_ball, blocks[1][0]])
    players = np.concatenate([pad_p(blocks[0][1]), np.full((HALF_GAP, k, 2), np.nan),
                              pad_p(blocks[1][1])])
    teams = np.concatenate([pad_t(blocks[0][2]), np.full((HALF_GAP, k), -1, np.int8),
                            pad_t(blocks[1][2])])
    gt = np.concatenate([blocks[0][3], np.full(HALF_GAP, X, np.int8), blocks[1][3]])

    # Metres from the centre spot -> 0-1 from a corner, as in the Metrica and competition data.
    scale = np.asarray(pitch)
    ball = ball / scale + 0.5
    players = players / scale + 0.5
    return {"name": "dfl_" + match, "ball": ball, "players": players, "teams": teams,
            "gt": gt, "pitch": pitch}
