"""Open DFL / IDSSE Bundesliga tracking (Bassek et al. 2025, CC BY 4.0) as Match objects.

The positions XML holds one FrameSet per object and half. Ball frames carry BallStatus
(1 alive, 0 dead) and BallPossession (1 home, 2 away); player frames carry X and Y only.
Coordinates are metres centred on the pitch centre and are shifted to [0, length] x [0, width].
Each half has its own frame counter; the two halves are concatenated into one timeline.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from lxml import etree

from mtah.schema import A, D, H, X, Match

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "external_data" / "dfl"
FIGSHARE_URL = "https://api.figshare.com/v2/file/download/{file_id}"
HTTP_OK = 200
REQUEST_TIMEOUT_S = 120
DOWNLOAD_CHUNK_BYTES = 1 << 20

GAME_PREFIX = "dfl_"
FILE_IDS = {
    "J03WPY": {"meta": 51643487, "positions": 51643526},
    "J03WN1": {"meta": 51643472, "positions": 51643517},
    "J03WMX": {"meta": 51643475, "positions": 51643514},
    "J03WOH": {"meta": 51643478, "positions": 51643520},
    "J03WQQ": {"meta": 51643484, "positions": 51643529},
    "J03WOY": {"meta": 51643481, "positions": 51643523},
    "J03WR9": {"meta": 51643490, "positions": 51643532},
}
GAME_IDS = tuple(GAME_PREFIX + match_id for match_id in FILE_IDS)

BALL_TEAM_ID = "BALL"
HOME_POSSESSION = 1
AWAY_POSSESSION = 2
BALL_DEAD = 0
BALL_ALIVE = 1
MISSING_STATUS = -1
MISSING_POSSESSION = 0
HALF_BY_SECTION = {"firstHalf": 0, "secondHalf": 1}
HALF_INDICES = (0, 1)
CACHE_SUFFIX = ".npz"


@dataclass(frozen=True)
class _FrameSet:
    """One FrameSet: a single object (player or ball) during one half."""

    half: int
    team: str
    person_id: str
    numbers: np.ndarray  # int64 [k] frame counters
    xy: np.ndarray  # float32 [k, 2] shifted metres
    status: np.ndarray = None  # int8 [k] BallStatus, ball only
    possession: np.ndarray = None  # int8 [k] BallPossession, ball only


def _resolve_dir(data_dir):
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR


def _match_id(game_id):
    """The bare match id ('J03WMX') behind a game id ('dfl_J03WMX'); ValueError when unknown."""
    bare = game_id[len(GAME_PREFIX):] if game_id.startswith(GAME_PREFIX) else None
    if bare not in FILE_IDS:
        raise ValueError("unknown DFL game id {!r}; expected one of {}".format(game_id, GAME_IDS))
    return bare


def _xml_path(data_dir, game_id, kind):
    return data_dir / "{}_{}.xml".format(game_id, kind)


def _stream_to_file(url, target):
    """Stream a URL to disk through a .part file so an interrupted download never looks complete."""
    partial = target.with_name(target.name + ".part")
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_S) as response:
        if response.status_code != HTTP_OK:
            raise OSError("{} answered HTTP {} instead of 200".format(url, response.status_code))
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
    if os.path.getsize(partial) == 0:
        os.remove(partial)
        raise OSError("{} returned an empty body".format(url))
    os.replace(partial, target)


def download(dest_dir=None):
    """Fetch the meta and positions XML of all seven matches; files already present are kept."""
    data_dir = _resolve_dir(dest_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for match_id, file_ids in FILE_IDS.items():
        game_id = GAME_PREFIX + match_id
        for kind, file_id in file_ids.items():
            target = _xml_path(data_dir, game_id, kind)
            if target.exists():
                logger.info("already present: %s", target.name)
                continue
            logger.info("downloading %s", target.name)
            _stream_to_file(FIGSHARE_URL.format(file_id=file_id), target)


def read_match_info(meta_path):
    """Pitch size in metres and home/away team ids from the match information XML."""
    root = etree.parse(str(meta_path)).getroot()
    environment = root.find(".//Environment")
    if environment is None:
        raise ValueError("{}: no Environment element with pitch size".format(meta_path))
    roles = {team.get("Role"): team.get("TeamId") for team in root.iter("Team")}
    if "home" not in roles or "guest" not in roles:
        raise ValueError("{}: home and guest Team elements are required".format(meta_path))
    pitch = (float(environment.get("PitchX")), float(environment.get("PitchY")))
    return {"pitch": pitch, "home_team_id": roles["home"], "away_team_id": roles["guest"]}


def _iter_frame_sets(xml_path):
    """Stream (attributes, frames) per FrameSet; frames are attribute dicts. Parsed nodes are freed."""
    attributes, frames = None, None
    for event, elem in etree.iterparse(str(xml_path), events=("start", "end"), huge_tree=True):
        if event == "start" and elem.tag == "FrameSet":
            attributes, frames = dict(elem.attrib), []
        elif event == "end" and elem.tag == "Frame" and frames is not None:
            frames.append(dict(elem.attrib))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet" and frames is not None:
            yield attributes, frames
            elem.clear()
            frames = None


def _check_pitch(pitch):
    length, width = (float(value) for value in pitch)
    if length <= 0 or width <= 0:
        raise ValueError("pitch must be positive metres, got {}".format(pitch))
    return length, width


def _half_index(attributes):
    section = attributes.get("GameSection")
    if section not in HALF_BY_SECTION:
        raise ValueError("unexpected GameSection {!r}; only first and second halves are supported"
                         .format(section))
    return HALF_BY_SECTION[section]


def _frame_number(frame):
    if "N" not in frame:
        raise ValueError("Frame without N attribute")
    return int(frame["N"])


def _coord(value, shift):
    """A coordinate in shifted metres; a missing attribute becomes NaN."""
    return np.nan if value is None else float(value) + shift


def _int_attr(frame, key, default):
    return int(frame[key]) if key in frame else default


def _player_set(attributes, frames, half, centre):
    numbers = np.array([_frame_number(f) for f in frames], dtype=np.int64)
    xy = np.array([[_coord(f.get("X"), centre[0]), _coord(f.get("Y"), centre[1])] for f in frames],
                  dtype=np.float32).reshape(-1, 2)
    return _FrameSet(half, attributes.get("TeamId"), attributes.get("PersonId"), numbers, xy)


def _ball_set(attributes, frames, half, centre):
    base = _player_set(attributes, frames, half, centre)
    status = np.array([_int_attr(f, "BallStatus", MISSING_STATUS) for f in frames], dtype=np.int8)
    possession = np.array([_int_attr(f, "BallPossession", MISSING_POSSESSION) for f in frames],
                          dtype=np.int8)
    return _FrameSet(half, BALL_TEAM_ID, "", base.numbers, base.xy, status, possession)


def _collect_sets(xml_path, home_team_id, away_team_id, centre):
    """Every FrameSet of interest; team labels are 'home', 'away' or 'ball' (others are dropped)."""
    sets = []
    for attributes, frames in _iter_frame_sets(xml_path):
        half = _half_index(attributes)
        team_id = attributes.get("TeamId")
        if team_id == BALL_TEAM_ID:
            sets.append(_ball_set(attributes, frames, half, centre))
        elif team_id in (home_team_id, away_team_id):
            side = "home" if team_id == home_team_id else "away"
            if not frames:
                continue
            player = _player_set(attributes, frames, half, centre)
            sets.append(_FrameSet(half, side, attributes.get("PersonId"), player.numbers, player.xy))
    return sets


def _half_spans(sets):
    """(first frame counter, frame count) per half, from every object's frame counters."""
    spans = []
    for half in HALF_INDICES:
        numbers = [s.numbers for s in sets if s.half == half and s.numbers.size]
        if not numbers:
            raise ValueError("no frames found for half {}".format(half + 1))
        all_numbers = np.concatenate(numbers)
        start = int(all_numbers.min())
        spans.append((start, int(all_numbers.max()) - start + 1))
    return tuple(spans)


def _global_index(frame_set, spans, offsets):
    """Row index in the concatenated timeline for each frame of a FrameSet."""
    return offsets[frame_set.half] + frame_set.numbers - spans[frame_set.half][0]


def _ball_track(sets, spans, offsets, total):
    ball_xy = np.full((total, 2), np.nan, dtype=np.float32)
    status = np.full(total, MISSING_STATUS, dtype=np.int8)
    possession = np.full(total, MISSING_POSSESSION, dtype=np.int8)
    for ball in (s for s in sets if s.team == BALL_TEAM_ID):
        index = _global_index(ball, spans, offsets)
        ball_xy[index] = ball.xy
        status[index] = ball.status
        possession[index] = ball.possession
    return ball_xy, status, possession


def _team_track(sets, side, spans, offsets, total):
    """Players as fixed slots ordered by sorted person id; NaN where a player is off the pitch."""
    team_sets = [s for s in sets if s.team == side]
    if not team_sets:
        raise ValueError("no player frames found for the {} team".format(side))
    person_ids = sorted({s.person_id for s in team_sets})
    slot_of = {person_id: slot for slot, person_id in enumerate(person_ids)}
    xy = np.full((total, len(person_ids), 2), np.nan, dtype=np.float32)
    for player in team_sets:
        index = _global_index(player, spans, offsets)
        xy[index, slot_of[player.person_id]] = player.xy
    return np.array(person_ids), xy


def parse_positions(xml_path, home_team_id, away_team_id, pitch):
    """Stream a DFL positions XML into arrays: home_xy/away_xy [T, slots, 2], ball_xy [T, 2],
    ball_status and ball_possession [T] (-1 / 0 when missing), blocks and home_ids/away_ids."""
    if home_team_id == away_team_id:
        raise ValueError("home and away team ids must differ")
    length, width = _check_pitch(pitch)
    centre = (length / 2.0, width / 2.0)
    sets = _collect_sets(xml_path, home_team_id, away_team_id, centre)
    spans = _half_spans(sets)
    offsets = (0, spans[0][1])
    total = spans[0][1] + spans[1][1]
    ball_xy, status, possession = _ball_track(sets, spans, offsets, total)
    home_ids, home_xy = _team_track(sets, "home", spans, offsets, total)
    away_ids, away_xy = _team_track(sets, "away", spans, offsets, total)
    blocks = tuple((offset, offset + count) for offset, (_, count) in zip(offsets, spans, strict=True))
    return {
        "home_xy": home_xy, "away_xy": away_xy, "ball_xy": ball_xy,
        "ball_status": status, "ball_possession": possession,
        "blocks": blocks, "home_ids": home_ids, "away_ids": away_ids,
    }


def ball_labels(ball_status, ball_possession):
    """D while the ball is dead, H or A from possession while alive, X when either is unknown."""
    labels = np.full(len(ball_status), X, dtype=np.int8)
    alive = ball_status == BALL_ALIVE
    labels[alive & (ball_possession == HOME_POSSESSION)] = H
    labels[alive & (ball_possession == AWAY_POSSESSION)] = A
    labels[ball_status == BALL_DEAD] = D
    return labels


def _match_from_arrays(game_id, arrays, pitch):
    return Match(
        game_id=game_id,
        home_xy=arrays["home_xy"],
        away_xy=arrays["away_xy"],
        ball_xy=arrays["ball_xy"],
        blocks=arrays["blocks"],
        pitch=pitch,
        labels=ball_labels(arrays["ball_status"], arrays["ball_possession"]),
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
    """A Match for one DFL game, read from its cached .npz or parsed from the XML and cached."""
    _match_id(game_id)
    data_dir = _resolve_dir(data_dir)
    cache_path = data_dir / (game_id + CACHE_SUFFIX)
    if cache_path.exists():
        return _load_cache(game_id, cache_path)
    meta_path = _xml_path(data_dir, game_id, "meta")
    positions_path = _xml_path(data_dir, game_id, "positions")
    for path in (meta_path, positions_path):
        if not path.exists():
            raise FileNotFoundError("{} is missing; run download() first".format(path))
    info = read_match_info(meta_path)
    arrays = parse_positions(positions_path, info["home_team_id"], info["away_team_id"], info["pitch"])
    match = _match_from_arrays(game_id, arrays, info["pitch"])
    _save_cache(cache_path, match)
    return match
