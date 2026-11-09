"""Loader for the SkillCorner open data (github.com/SkillCorner/opendata, MIT licence, credit
SkillCorner): 20 A-League 2024/25 matches of broadcast tracking at 10 fps.

Unlike the Metrica and DFL games, this tracking comes from an automatic system and keeps its real
ball gaps: each ball frame says whether the ball was detected or only extrapolated, and frames
without any data (replays, close-ups) are empty. That makes it the only external source of real
gap patterns, although a broadcast camera follows the ball and so loses it less than a fixed
tactical camera does.

Labels: the ball is live inside SkillCorner's phases of play ("only defined when the ball is in
play", README) and dead outside them. During live play the team is the per-frame possession group
of the tracking file, carried forward over frames where it is empty. These labels are SkillCorner's
own automatic output, not human annotation.

Output layout matches dfl.load: 25 fps (upsampled), first half, a block of HALF_GAP X frames,
second half; ball and players normalised 0-1 from a corner. Extra arrays:
  ball_detected  ball detected by the tracker on this frame (the ball array is NaN elsewhere)
  has_data       the frame carries any tracking at all
  ball_z         ball height in metres where detected
"""

import json
import os

import numpy as np

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                            "external_data", "skillcorner_opendata", "data", "matches")
ROOT = os.environ.get("SKILLCORNER_DIR", DEFAULT_ROOT)
SRC_FPS, FPS = 10, 25
HALF_GAP = 250
H, A, D, X = 0, 1, 2, 3
GROUP = {"home team": H, "away team": A}


def matches():
    return sorted(d for d in os.listdir(ROOT) if d.isdigit())


def _path(match, suffix):
    return os.path.join(ROOT, match, "{}_{}".format(match, suffix))


def _source(match):
    """Per source frame (10 fps): ball xyz + detected flag, player positions by slot, possession
    group, plus the match metadata."""
    meta = json.load(open(_path(match, "match.json")))
    home_id = meta["home_team"]["id"]
    slot = {p["id"]: j for j, p in enumerate(meta["players"])}
    team = np.array([H if p["team_id"] == home_id else A for p in meta["players"]], np.int8)
    rows = [json.loads(line) for line in open(_path(match, "tracking_extrapolated.jsonl"))]
    n = max(r["frame"] for r in rows) + 1
    ball = np.full((n, 3), np.nan)
    detected = np.zeros(n, bool)
    has_data = np.zeros(n, bool)
    players = np.full((n, len(slot), 2), np.nan)
    group = np.full(n, -1, np.int8)
    for r in rows:
        f = r["frame"]
        b = r["ball_data"] or {}
        if b.get("x") is not None:
            ball[f] = (b["x"], b["y"], b["z"] if b.get("z") is not None else np.nan)
            detected[f] = bool(b.get("is_detected"))
        pd_ = r["player_data"] or []
        has_data[f] = len(pd_) > 0 or b.get("x") is not None
        for p in pd_:
            j = slot.get(p["player_id"])
            if j is not None:
                players[f, j] = (p["x"], p["y"])
        g = (r.get("possession") or {}).get("group")
        group[f] = GROUP.get(g, -1)
    return meta, ball, detected, has_data, players, team, group


def _phases(match):
    """[(frame_start, frame_end inclusive, team code)] in source frames."""
    import csv
    out = []
    with open(_path(match, "phases_of_play.csv")) as handle:
        for row in csv.DictReader(handle):
            out.append((int(row["frame_start"]), int(row["frame_end"]),
                        int(row["team_in_possession_id"])))
    return out


def _resample(x, lo, hi):
    """Linear interpolation from SRC_FPS to FPS on source frames [lo, hi); NaN where either
    neighbour is NaN. Returns (values, nearest source index)."""
    n_out = int((hi - 1 - lo) * FPS / SRC_FPS) + 1
    s = lo + np.arange(n_out) * SRC_FPS / FPS
    i0 = np.floor(s).astype(int)
    i1 = np.minimum(i0 + 1, hi - 1)
    w = (s - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - w) + x[i1] * w, np.rint(s).astype(int).clip(lo, hi - 1), i0, i1


def load(match):
    match = str(match)
    meta, ball3, detected, has_data, players_src, team, group = _source(match)
    home_id = meta["home_team"]["id"]
    phases = _phases(match)
    n_src = len(ball3)
    live = np.zeros(n_src, bool)
    phase_team = np.full(n_src, -1, np.int8)
    for a, b, t in phases:
        live[a:b + 1] = True
        phase_team[a:b + 1] = H if t == home_id else A
    gt_src = np.full(n_src, X, np.int8)
    for p in meta["match_periods"]:
        gt_src[p["start_frame"]:p["end_frame"] + 1] = D
    # Team during live play: the possession group, carried forward inside each phase; the
    # phase's team where nothing came before.
    cur, prev_live = -1, False
    for f in range(n_src):
        if not live[f] or gt_src[f] == X:
            prev_live = False
            continue
        if not prev_live:
            cur = phase_team[f]
        if group[f] >= 0:
            cur = group[f]
        gt_src[f] = cur
        prev_live = True

    ball_src = np.where(detected[:, None], ball3, np.nan)
    parts = []
    for p in sorted(meta["match_periods"], key=lambda q: q["period"]):
        lo, hi = p["start_frame"], min(p["end_frame"] + 1, n_src)
        b, near, i0, i1 = _resample(ball_src, lo, hi)
        pl, _, _, _ = _resample(players_src, lo, hi)
        parts.append(dict(ball=b[:, :2], z=b[:, 2], players=pl, gt=gt_src[near],
                          det=detected[i0] & detected[i1], data=has_data[near]))

    k = players_src.shape[1]
    gap = lambda shape, v: np.full(shape, v)
    ball = np.concatenate([parts[0]["ball"], gap((HALF_GAP, 2), np.nan), parts[1]["ball"]])
    z = np.concatenate([parts[0]["z"], gap(HALF_GAP, np.nan), parts[1]["z"]])
    players = np.concatenate([parts[0]["players"], gap((HALF_GAP, k, 2), np.nan),
                              parts[1]["players"]])
    gt = np.concatenate([parts[0]["gt"], gap(HALF_GAP, X).astype(np.int8), parts[1]["gt"]])
    det = np.concatenate([parts[0]["det"], gap(HALF_GAP, False), parts[1]["det"]])
    data = np.concatenate([parts[0]["data"], gap(HALF_GAP, False), parts[1]["data"]])
    teams = np.where(np.isnan(players[:, :, 0]), -1, team[None, :]).astype(np.int8)

    # Metres from the centre spot -> 0-1 from a corner, as in the other datasets.
    pitch = (float(meta["pitch_length"]), float(meta["pitch_width"]))
    scale = np.asarray(pitch)
    return {"name": "sc_" + match, "ball": ball / scale + 0.5, "players": players / scale + 0.5,
            "teams": teams, "gt": gt, "pitch": pitch, "ball_detected": det, "has_data": data,
            "ball_z": z}
