"""SkillCorner open data (MIT, 2024/25 A-League sample) as Match objects, for validation only.

This is broadcast-derived tracking at 10 fps. An object is NaN in a frame unless SkillCorner
flags it as detected (is_detected true); extrapolated positions are dropped, not interpolated.
Metadata and phases come from the match JSON and the phases-of-play CSV. The tracking JSON-lines
file is stored with git-lfs and is fetched through the media host.

Coordinates: SkillCorner centres the pitch on (0, 0) with x along the long side. They are shifted
to x in [0, 105] and y in [0, 68] metres, as the README documents a 105 x 68 m pitch.

Labels (H, A, D, X) at 25 fps:
- Possession: H or A from the per-frame possession group ("home team" / "away team"). Unknown
  stretches shorter than 2 s between two frames of the same team are filled with that team.
- Dead ball (D): a frame inside a live half that no phase of play covers. SkillCorner defines
  phases of play only while the ball is in play, so the gaps between phases are stoppages.
- X: frames outside the two live halves and in-play frames whose possession stays unknown.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import requests

from mtah.schema import A, D, FPS, H, X, Match

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "external_data" / "skillcorner"
RAW_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match}/{match}_{suffix}"
LFS_URL = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match}/{match}_{suffix}"
FILE_SUFFIXES = {
    "meta": "match.json",
    "phases": "phases_of_play.csv",
    "tracking": "tracking_extrapolated.jsonl",
}
LFS_KINDS = frozenset(("tracking",))
LFS_POINTER_PREFIX = b"version https://git-lfs"
LFS_POINTER_PROBE_BYTES = 64
HTTP_OK = 200
REQUEST_TIMEOUT_S = 120
DOWNLOAD_CHUNK_BYTES = 1 << 20
CACHE_SUFFIX = ".npz"

MATCH_IDS = (
    "1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846", "1986691",
    "1996435", "1996436", "2006229", "2006363", "2007448", "2007721", "2010085", "2011166",
    "2013725", "2015213", "2016236", "2017461",
)
GAME_PREFIX = "skc_"
GAME_IDS = tuple(GAME_PREFIX + match_id for match_id in MATCH_IDS)

SOURCE_FPS = 10
PITCH = (105.0, 68.0)
UNKNOWN = -1
POSSESSION_CODES = {"home team": H, "away team": A}
HOME_SIDE, AWAY_SIDE = 0, 1
SHORT_UNKNOWN_S = 2.0
SHORT_UNKNOWN_FRAMES = int(SHORT_UNKNOWN_S * FPS)  # runs strictly shorter than this are filled


def _resolve_dir(data_dir):
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR


def _match_id(game_id):
    """The bare match id behind a game id ('skc_1874553'); ValueError when unknown."""
    bare = game_id[len(GAME_PREFIX):] if game_id.startswith(GAME_PREFIX) else None
    if bare not in MATCH_IDS:
        raise ValueError("unknown SkillCorner game id {!r}; expected one of {}".format(game_id, GAME_IDS))
    return bare


def _cache_path(data_dir, game_id):
    return data_dir / (game_id + CACHE_SUFFIX)


def _stream_to_file(url, target):
    """Stream a URL to disk through a .part file so an interrupted download never looks complete."""
    partial = target.with_name(target.name + ".part")
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_S) as response:
        if response.status_code != HTTP_OK:
            raise OSError("{} answered HTTP {} instead of 200".format(url, response.status_code))
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
    os.replace(partial, target)


def _check_not_lfs_pointer(path):
    """Raise when a download returned a git-lfs pointer instead of the real file; remove it."""
    with open(path, "rb") as handle:
        head = handle.read(LFS_POINTER_PROBE_BYTES)
    if head.startswith(LFS_POINTER_PREFIX) or os.path.getsize(path) == 0:
        os.remove(path)
        raise OSError("{} is a git-lfs pointer or empty, not the data file".format(path.name))


def download(dest_dir=None):
    """Fetch metadata, phases and tracking of all matches into dest_dir; present files are kept."""
    data_dir = _resolve_dir(dest_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for match_id in MATCH_IDS:
        for kind, suffix in FILE_SUFFIXES.items():
            target = _raw_path(data_dir, match_id, kind)
            if target.exists():
                continue
            base = LFS_URL if kind in LFS_KINDS else RAW_URL
            _stream_to_file(base.format(match=match_id, suffix=suffix), target)
            _check_not_lfs_pointer(target)


def read_match_info(meta_path):
    """Frame count, the two live halves as 10 fps spans, and each side's sorted roster ids."""
    with open(meta_path, encoding="utf-8") as handle:
        data = json.load(handle)
    periods = data.get("match_periods") or []
    if not periods:
        raise ValueError("{}: no match_periods".format(meta_path))
    home_id, away_id = data["home_team"]["id"], data["away_team"]["id"]
    if home_id == away_id:
        raise ValueError("{}: home and away team ids must differ".format(meta_path))
    roster = {
        HOME_SIDE: sorted(p["id"] for p in data["players"] if p["team_id"] == home_id),
        AWAY_SIDE: sorted(p["id"] for p in data["players"] if p["team_id"] == away_id),
    }
    return {
        "n_frames": max(int(p["end_frame"]) for p in periods) + 1,
        "halves": tuple((int(p["start_frame"]), int(p["end_frame"]) + 1) for p in periods),
        "roster": roster,
    }


def _slot_map(roster):
    """player id -> (side, slot): each side's players are slots in sorted-id order."""
    return {pid: (side, slot)
            for side, ids in roster.items() for slot, pid in enumerate(ids)}


def _shift(x_value, y_value, pitch):
    """SkillCorner centred metres to [0, length] x [0, width] metres."""
    length, width = pitch
    return (float(x_value) + length / 2.0, float(y_value) + width / 2.0)


def _record_frame(record, n_frames):
    """The record's frame index, or None when it lies after the last period (outside both halves)."""
    frame = int(record["frame"])
    if frame < 0:
        raise ValueError("negative frame index {}".format(frame))
    return frame if frame < n_frames else None


def _store_ball(record, frame, pitch, ball_xy, owner):
    ball = record["ball_data"]
    if ball.get("is_detected") is True and ball.get("x") is not None:
        ball_xy[frame] = _shift(ball["x"], ball["y"], pitch)
    owner[frame] = POSSESSION_CODES.get(record["possession"]["group"], UNKNOWN)


def _store_players(record, frame, pitch, slot_of, team_xy):
    for entry in record["player_data"]:
        slot = slot_of.get(entry["player_id"])
        if slot is None or entry.get("is_detected") is not True or entry.get("x") is None:
            continue
        side, index = slot
        team_xy[side][frame, index] = _shift(entry["x"], entry["y"], pitch)


def _read_tracking(path, info, pitch=PITCH):
    """Per-frame arrays at 10 fps: home_xy, away_xy [T, slots, 2], ball_xy [T, 2], owner [T] (H/A/UNKNOWN)."""
    n_frames = info["n_frames"]
    slot_of = _slot_map(info["roster"])
    team_xy = {side: np.full((n_frames, len(ids), 2), np.nan, dtype=np.float32)
               for side, ids in info["roster"].items()}
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    owner = np.full(n_frames, UNKNOWN, dtype=np.int8)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = _record_frame(record, n_frames)
            if frame is None:
                continue
            _store_ball(record, frame, pitch, ball_xy, owner)
            _store_players(record, frame, pitch, slot_of, team_xy)
    return team_xy[HOME_SIDE], team_xy[AWAY_SIDE], ball_xy, owner


def _phase_mask(path, n_frames):
    """Frames covered by a phase of play at 10 fps (phases are inclusive of frame_end)."""
    mask = np.zeros(n_frames, dtype=bool)
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            start, stop = int(row["frame_start"]), int(row["frame_end"]) + 1
            mask[start:min(stop, n_frames)] = True
    return mask


def resample(arr_10fps, src_fps, n_out):
    """Nearest-frame resampling to FPS: output i takes source rint(i * src_fps / FPS). No interpolation."""
    source = np.asarray(arr_10fps)
    if source.shape[0] == 0:
        raise ValueError("cannot resample an empty track")
    if src_fps <= 0 or n_out < 0:
        raise ValueError("src_fps must be positive and n_out non-negative, got {} and {}".format(src_fps, n_out))
    index = np.rint(np.arange(n_out) * (src_fps / FPS)).astype(np.int64)
    return source[np.minimum(index, source.shape[0] - 1)]


def _unknown_runs(owner):
    """(start, stop) of every maximal run of UNKNOWN in owner."""
    unknown = np.concatenate(([False], owner == UNKNOWN, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(unknown))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def fill_short_gaps(owner, short_frames):
    """A copy where each unknown run shorter than short_frames, bounded by the same team on both
    sides, takes that team. Runs at the match edges or between different teams stay unknown."""
    filled = np.array(owner, copy=True)
    for start, stop in _unknown_runs(owner):
        if stop - start >= short_frames or start == 0 or stop == len(owner):
            continue
        if owner[start - 1] == owner[stop]:
            filled[start:stop] = owner[start - 1]
    return filled


def build_labels(owner, in_play, in_block):
    """int8 H/A/D/X per frame: D inside a live half but outside play, H/A from owner while in play,
    X everywhere else (outside the halves, or in play with an unknown owner)."""
    labels = np.full(len(owner), X, dtype=np.int8)
    labels[in_block & ~in_play] = D
    scored = in_block & in_play & (owner != UNKNOWN)
    labels[scored] = owner[scored]
    return labels


def _span_mask(spans, n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in spans:
        mask[start:stop] = True
    return mask


def _to_output_frame(source_frame):
    return int(round(source_frame * FPS / SOURCE_FPS))


def _raw_path(data_dir, match_id, kind):
    return data_dir / "{}_{}".format(match_id, FILE_SUFFIXES[kind])


def _build_match(game_id, data_dir):
    match_id = _match_id(game_id)
    info = read_match_info(_raw_path(data_dir, match_id, "meta"))
    home_10, away_10, ball_10, owner_10 = _read_tracking(_raw_path(data_dir, match_id, "tracking"), info)
    phase_10 = _phase_mask(_raw_path(data_dir, match_id, "phases"), info["n_frames"])
    n_out = int(round(info["n_frames"] * FPS / SOURCE_FPS))
    blocks = tuple((_to_output_frame(start), _to_output_frame(stop)) for start, stop in info["halves"])
    in_block = _span_mask(blocks, n_out)
    in_play = resample(phase_10, SOURCE_FPS, n_out).astype(bool)
    owner = fill_short_gaps(resample(owner_10, SOURCE_FPS, n_out), SHORT_UNKNOWN_FRAMES)
    return Match(
        game_id=game_id,
        home_xy=resample(home_10, SOURCE_FPS, n_out),
        away_xy=resample(away_10, SOURCE_FPS, n_out),
        ball_xy=resample(ball_10, SOURCE_FPS, n_out),
        blocks=blocks,
        pitch=PITCH,
        labels=build_labels(owner, in_play, in_block),
    )


def _save_cache(path, match):
    partial = path.with_name(path.name + ".part")
    with open(partial, "wb") as handle:
        np.savez(handle,
                 home_xy=match.home_xy, away_xy=match.away_xy, ball_xy=match.ball_xy,
                 labels=match.labels, blocks=np.asarray(match.blocks, dtype=np.int64),
                 pitch=np.asarray(match.pitch, dtype=np.float64))
    os.replace(partial, path)


def _load_cache(game_id, path):
    with np.load(path, allow_pickle=False) as cache:
        return Match(
            game_id=game_id,
            home_xy=cache["home_xy"],
            away_xy=cache["away_xy"],
            ball_xy=cache["ball_xy"],
            labels=cache["labels"],
            blocks=tuple(tuple(block) for block in cache["blocks"].tolist()),
            pitch=tuple(cache["pitch"].tolist()),
        )


def load(game_id, data_dir=None):
    """A Match for one SkillCorner game, read from its cached .npz or built from the raw files and cached."""
    _match_id(game_id)
    data_dir = _resolve_dir(data_dir)
    cache_path = _cache_path(data_dir, game_id)
    if cache_path.exists():
        return _load_cache(game_id, cache_path)
    match_id = _match_id(game_id)
    for kind in FILE_SUFFIXES:
        raw = _raw_path(data_dir, match_id, kind)
        if not raw.exists():
            raise FileNotFoundError("{} is missing; run download() first".format(raw))
    match = _build_match(game_id, data_dir)
    _save_cache(cache_path, match)
    return match
