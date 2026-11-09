"""Shared helpers for experiments on the external games."""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "external"))

import metrica_sample as ms  # noqa: E402
from mtah.labels import read_labels  # noqa: E402

LABEL_DIR = os.path.join(REPO, "..", "external_data", "labels", "metrica_sample")
CACHE = os.path.join(REPO, "out", "cache")


def load_game(game):
    """dict(name, ball, players, teams, gt, pitch) for one Metrica sample game; the tracking is
    cached as .npz because Game 3's text format takes a while to parse."""
    path = os.path.join(CACHE, "metrica_{}.npz".format(game))
    if os.path.isfile(path):
        z = np.load(path)
        ball, players, teams = z["ball"], z["players"], z["teams"]
    else:
        ball, players, teams = ms.load_tracking(game)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, ball=ball, players=players, teams=teams)
    gt = read_labels(os.path.join(LABEL_DIR, "Sample_Game_{}_gt.csv".format(game)))
    if len(gt) != len(ball):
        raise ValueError("game {}: {} labels vs {} tracking frames".format(game, len(gt),
                                                                           len(ball)))
    return {"name": "metrica{}".format(game), "ball": ball, "players": players,
            "teams": teams, "gt": gt, "pitch": ms.PITCH}


def external_games():
    return [load_game(g) for g in ms.GAMES]


def load_dfl(match):
    """One IDSSE/DFL match in the same dict layout, cached as .npz (the XML takes ~10 s)."""
    import dfl
    path = os.path.join(CACHE, "dfl_{}.npz".format(match))
    if not os.path.isfile(path):
        g = dfl.load(match)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, **{k: g[k] for k in ("ball", "players", "teams", "gt", "pitch")})
    z = np.load(path)
    return {"name": "dfl_" + match, "ball": z["ball"], "players": z["players"],
            "teams": z["teams"], "gt": z["gt"], "pitch": tuple(z["pitch"])}


def dfl_games():
    import dfl
    return [load_dfl(m) for m in dfl.MATCHES]


SC_KEYS = ("ball", "players", "teams", "gt", "pitch", "ball_detected", "has_data", "ball_z")


def load_skillcorner(match):
    """One SkillCorner open-data match (real ball gaps) in the same dict layout plus
    ball_detected / has_data / ball_z, cached as .npz."""
    import skillcorner
    path = os.path.join(CACHE, "skillcorner_{}.npz".format(match))
    if not os.path.isfile(path):
        g = skillcorner.load(match)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, **{k: g[k] for k in SC_KEYS})
    z = np.load(path)
    g = {k: z[k] for k in SC_KEYS}
    g["pitch"] = tuple(g["pitch"])
    g["name"] = "sc_{}".format(match)
    return g


def skillcorner_games():
    import skillcorner
    return [load_skillcorner(m) for m in skillcorner.matches()]


def table(rows, keys):
    """Plain-text table of dict rows."""
    widths = [max(len(k), *(len(fmt(r[k])) for r in rows)) for k in keys]
    lines = ["  ".join(k.rjust(w) for k, w in zip(keys, widths))]
    lines += ["  ".join(fmt(r[k]).rjust(w) for k, w in zip(keys, widths)) for r in rows]
    return "\n".join(lines)


def fmt(v):
    if isinstance(v, float):
        return "{:.4f}".format(v)
    return str(v)
