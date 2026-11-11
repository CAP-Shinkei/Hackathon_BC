"""Write a Match in the competition tracking format, and a kick-off tagging file for the tagging loader.

The pair written by export_atd is what io_atd.load_atd and the official starter read back, so a
submission pipeline can be exercised end to end on any game that has a Match. Coordinates are
normalised by the match pitch and written to six decimals, which keeps the round-trip error below
1e-5 of the pitch size. Speed is not carried by a Match, so it is always written as 0.0.
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

# Params
COORD_DECIMALS = 6
SPEED_TEXT = "0.0"
MISSING_GROUP = "NaN,NaN,NaN"
FIELD_SEP = ":"
GROUP_SEP = ";"
FIRST_FRAME_ID = 1
HOME_TEAM_ID = "home"
AWAY_TEAM_ID = "away"
TEAM_NAMES = ("Home", "Away")
KICKOFF_CODE = "SET PIECES"
KICKOFF_TYPE = "Kick Off (Start)"
KICKOFF_SECONDS = 2.0
HALF_NAMES = ("1st Half", "2nd Half")


def export_atd(match, out_dir, stem):
    """Write <stem>_atd.xml and <stem>_atd.txt into out_dir. Home slots first, then away."""
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, stem + "_atd.xml")
    raw_path = os.path.join(out_dir, stem + "_atd.txt")
    _write_meta(meta_path, match.home_xy.shape[1], match.away_xy.shape[1])
    _write_raw(raw_path, match)
    return meta_path, raw_path


def export_kickoff_tagging(blocks, out_path, fps=25):
    """A SportsCode-style XML with one 'Kick Off (Start)' set piece per block, for the tagging loader."""
    if not blocks:
        raise ValueError("no blocks to tag")
    if len(blocks) > len(HALF_NAMES):
        raise ValueError("{} blocks, but only {} halves are named".format(
            len(blocks), len(HALF_NAMES)))
    if fps <= 0:
        raise ValueError("fps must be positive, got {}".format(fps))

    root = ET.Element("ALL_INSTANCES")
    for number, (start, _stop) in enumerate(blocks, start=1):
        _add_kickoff(root, number, start / fps, HALF_NAMES[number - 1])
    _write_xml(root, out_path)
    return out_path


def _write_meta(meta_path, n_home, n_away):
    root = ET.Element("MetaData")
    for team_id, name in zip((HOME_TEAM_ID, AWAY_TEAM_ID), TEAM_NAMES, strict=True):
        team = ET.SubElement(root, "Team", id=team_id)
        ET.SubElement(team, "Name").text = name
    slot = 0
    for team_id, count in ((HOME_TEAM_ID, n_home), (AWAY_TEAM_ID, n_away)):
        for _ in range(count):
            slot += 1
            ET.SubElement(root, "Player", id=str(slot), teamId=team_id,
                          ShirtNumber=str(slot))
    _write_xml(root, meta_path)


def _write_xml(root, path):
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _write_raw(raw_path, match):
    """One line per frame: 1-based frame id, player groups, then the ball."""
    slot_xy = np.concatenate(
        [match.home_xy, match.away_xy, match.ball_xy[:, np.newaxis, :]], axis=1)
    normalised = (slot_xy.astype(np.float64) / np.array(match.pitch, dtype=np.float64)).tolist()
    with open(raw_path, "w") as handle:
        handle.writelines(_format_line(index, frame) + "\n"
                          for index, frame in enumerate(normalised))


def _format_line(index, frame):
    groups = [_format_group(xy) for xy in frame]
    return FIELD_SEP.join([str(index + FIRST_FRAME_ID), GROUP_SEP.join(groups[:-1]), groups[-1]])


def _format_group(xy):
    x, y = xy
    if math.isnan(x) or math.isnan(y):
        return MISSING_GROUP
    return "{:.{d}f},{:.{d}f},{}".format(x, y, SPEED_TEXT, d=COORD_DECIMALS)


def _add_kickoff(root, number, start_seconds, half):
    instance = ET.SubElement(root, "instance")
    ET.SubElement(instance, "ID").text = str(number)
    ET.SubElement(instance, "start").text = "{:.6f}".format(start_seconds)
    ET.SubElement(instance, "end").text = "{:.6f}".format(start_seconds + KICKOFF_SECONDS)
    ET.SubElement(instance, "code").text = KICKOFF_CODE
    _add_label(instance, "Type", KICKOFF_TYPE)
    _add_label(instance, "Half", half)


def _add_label(instance, group, text):
    label = ET.SubElement(instance, "label")
    ET.SubElement(label, "group").text = group
    ET.SubElement(label, "text").text = text
