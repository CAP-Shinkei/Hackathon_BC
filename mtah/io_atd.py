"""Read the competition tracking format (ATD) into a Match.

The ATD is a FIFA/EPTS metadata XML plus a raw text file, one line per frame:

    <frame_id>:<x,y,speed>;<x,y,speed>;...:<ball_x,ball_y,ball_speed>

frame_id counts from 1 and the array index is frame_id - 1. Coordinates are normalised 0..1 and are
scaled to metres here. A missing player or ball is written NaN,NaN,NaN and becomes NaN in the Match.
Team membership of each slot comes from the metadata, through the official starter parser, so this
module and the reference baseline can never disagree about which slot is which team.
"""
import os
import sys
import warnings

import numpy as np

from mtah.schema import Match

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _folder in (os.path.join(ROOT, "official", "starter"), os.path.join(ROOT, "official")):
    if _folder not in sys.path:
        sys.path.insert(0, _folder)
from baseline import team_of_each_track  # noqa: E402

# Params
FIELD_SEP = ":"
GROUP_SEP = ";"
COORD_SEP = ","
FIELD_PARTS = 3  # frame id, player groups, ball group
VALUES_PER_GROUP = 3  # x, y, speed
FIRST_FRAME_ID = 1


def team_of_slots(meta_path):
    """Array over track slots: 0 for a home slot, 1 for an away slot, in file order."""
    return team_of_each_track(meta_path)


def load_atd(meta_path, raw_path, n_frames, pitch, blocks, game_id="match"):
    """The ATD as a Match in metres, with no labels. Frames the file omits stay NaN."""
    if n_frames < 1:
        raise ValueError("{}: n_frames must be positive, got {}".format(game_id, n_frames))
    if len(pitch) != 2 or min(pitch) <= 0:
        raise ValueError("{}: pitch must be two positive lengths, got {}".format(game_id, pitch))

    slots = team_of_slots(meta_path)
    frames = _read_frames(raw_path, n_frames, len(slots))
    xy = frames[:, :, :2] * np.array(pitch, dtype=np.float32)
    return Match(
        game_id=game_id,
        home_xy=np.ascontiguousarray(xy[:, :len(slots)][:, slots == 0]),
        away_xy=np.ascontiguousarray(xy[:, :len(slots)][:, slots == 1]),
        ball_xy=np.ascontiguousarray(xy[:, len(slots)]),
        blocks=tuple(blocks),
        pitch=tuple(pitch),
    )


def _read_frames(raw_path, n_frames, n_slots):
    """float32 [n_frames, n_slots + 1, 3]: every slot's x, y, speed, then the ball last."""
    frames = np.full((n_frames, n_slots + 1, VALUES_PER_GROUP), np.nan, dtype=np.float32)
    skipped = 0
    with open(raw_path) as handle:
        for line_no, line in enumerate(handle, start=1):
            parsed = _parse_line(line, n_slots, n_frames, line_no)
            if parsed is not None:
                index, values = parsed
                frames[index] = values
            elif line.strip():
                skipped += 1
    if skipped:
        warnings.warn("{}: {} non-empty lines are not frame records and were skipped".format(
            raw_path, skipped), stacklevel=2)
    return frames


def _parse_line(line, n_slots, n_frames, line_no):
    """(frame index, [n_slots + 1, 3] values), or None for a line that is not a frame record."""
    parts = line.strip().split(FIELD_SEP)
    if len(parts) != FIELD_PARTS:
        return None
    groups = parts[1].split(GROUP_SEP)
    if len(groups) != n_slots:
        raise ValueError("line {}: {} player groups, expected {}".format(
            line_no, len(groups), n_slots))
    frame_id = _frame_id(parts[0], n_frames, line_no)
    tokens = COORD_SEP.join(groups + [parts[2]]).split(COORD_SEP)
    expected = VALUES_PER_GROUP * (n_slots + 1)
    if len(tokens) != expected:
        raise ValueError("line {}: {} values, expected {}".format(line_no, len(tokens), expected))
    try:
        values = np.array(tokens, dtype=np.float32).reshape(n_slots + 1, VALUES_PER_GROUP)
    except ValueError as error:
        raise ValueError("line {}: non-numeric value ({})".format(line_no, error)) from error
    return frame_id - FIRST_FRAME_ID, values


def _frame_id(text, n_frames, line_no):
    """The frame id as an int, checked against the frame count."""
    try:
        frame_id = int(text)
    except ValueError as error:
        raise ValueError("line {}: frame id {!r} is not an integer".format(line_no, text)) from error
    if not FIRST_FRAME_ID <= frame_id < FIRST_FRAME_ID + n_frames:
        raise ValueError("line {}: frame {} outside 1..{}".format(
            line_no, frame_id, n_frames))
    return frame_id
