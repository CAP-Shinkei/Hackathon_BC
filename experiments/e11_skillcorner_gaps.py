"""E11: real ball-gap patterns (SkillCorner broadcast tracking) against the gap simulator.

The same aggregates are computed on SkillCorner's real gaps and on the simulator's gaps over the
Metrica + DFL games (35% and 50% live missing):
  - how much ball is missing in live and dead play, and on graded frames;
  - live gap lengths;
  - what makes a gap start during live play: hazard of the ball disappearing on the next frame,
    relative to the overall hazard, by players within 2 m, ball speed and (SkillCorner only)
    ball height;
  - where in a dead stretch the ball is seen (first vs last quarter: the simulator shows it only
    in a tail at the restart spot).

    python experiments/e11_skillcorner_gaps.py
"""

import numpy as np

from common import dfl_games, external_games, load_skillcorner, table
from gapsim import Params, damage
import skillcorner

FPS = 25


def runs_of(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return e[0::2], e[1::2]


def stats(ball, players, gt, pitch, z=None, has_data=None):
    scale = np.asarray(pitch, dtype=float)
    seen = ~np.isnan(ball).any(axis=1)
    live = np.isin(gt, (0, 1))
    dead = gt == 2
    graded = gt != 3
    out = {"missing_graded": 1 - seen[graded].mean(), "missing_live": 1 - seen[live].mean(),
           "seen_dead": seen[dead].mean()}
    if has_data is not None:
        out["no_data_live"] = 1 - has_data[live].mean()
        out["no_data_dead"] = 1 - has_data[dead].mean()

    # Live gaps: maximal runs of unseen frames fully inside live play.
    a, b = runs_of(~seen & live)
    lens = (b - a) / FPS
    out["live_gaps_per_min"] = len(lens) / (live.sum() / FPS / 60)
    out["gap_median_s"] = float(np.median(lens)) if len(lens) else np.nan
    out["gap_p90_s"] = float(np.percentile(lens, 90)) if len(lens) else np.nan
    out["gap_share_over_5s"] = float(lens[lens > 5].sum() / max(lens.sum(), 1e-9))

    # Gap-start hazard: frame t-1 seen and live, frame t unseen and live.
    bm = ball * scale
    d = np.linalg.norm(players * scale - bm[:, None, :], axis=2)
    crowd = np.nansum(d < 2.0, axis=1)
    speed = np.full(len(ball), np.nan)
    speed[1:] = np.linalg.norm(np.diff(bm, axis=0), axis=1) * FPS
    prev_ok = np.zeros(len(ball), bool)
    prev_ok[2:] = seen[1:-1] & seen[:-2] & live[1:-1] & live[2:]
    start = np.zeros(len(ball), bool)
    start[1:] = ~seen[1:]
    base = start[prev_ok].mean()
    out["hazard_per_s"] = base * FPS
    cond_crowd = np.zeros(len(ball), int)
    cond_crowd[1:] = crowd[:-1]
    cond_speed = np.full(len(ball), np.nan)
    cond_speed[1:] = speed[:-1]
    for name, m in (("crowd0", cond_crowd == 0), ("crowd1", cond_crowd == 1),
                    ("crowd2+", cond_crowd >= 2), ("slow<5", cond_speed < 5),
                    ("fast>15", cond_speed > 15)):
        sel = prev_ok & m
        out["rel_" + name] = start[sel].mean() / base if sel.sum() > 100 else np.nan
    if z is not None:
        cond_z = np.full(len(ball), np.nan)
        cond_z[1:] = z[:-1]
        for name, m in (("z<1m", cond_z < 1.0), ("z>2m", cond_z > 2.0)):
            sel = prev_ok & m
            out["rel_" + name] = start[sel].mean() / base if sel.sum() > 100 else np.nan

    # Ball seen in the first and the last quarter of dead stretches longer than 4 s.
    a, b = runs_of(dead)
    first, last = [], []
    for s, e in zip(a, b):
        if e - s < 4 * FPS:
            continue
        q = (e - s) // 4
        first.append(seen[s:s + q].mean())
        last.append(seen[e - q:e].mean())
    out["dead_seen_first_q"] = float(np.mean(first))
    out["dead_seen_last_q"] = float(np.mean(last))
    return out


def summarise(rows, source):
    keys = [k for k in rows[0] if k != "game"]
    return dict(source=source, games=len(rows),
                **{k: float(np.nanmean([r[k] for r in rows])) for k in keys})


def main():
    real = []
    for m in skillcorner.matches():
        g = load_skillcorner(m)
        real.append(dict(game=g["name"], **stats(g["ball"], g["players"], g["gt"], g["pitch"],
                                                 g["ball_z"], g["has_data"])))
        print(g["name"], "done", flush=True)
    keys = list(real[0].keys())
    print()
    print("SkillCorner, real gaps, per match")
    print(table(real, keys[:9]))
    print(table(real, [keys[0]] + keys[9:]))

    rows = [summarise(real, "skillcorner_real")]
    for rate in (0.35, 0.5):
        sim = []
        for g in external_games() + dfl_games():
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(live_missing=rate), seed=7)
            sim.append(dict(game=g["name"], **stats(ball, players, g["gt"], g["pitch"])))
        rows.append(summarise(sim, "gapsim_{:.0%}".format(rate)))
    # Simulator applied to SkillCorner's detected ball, with its own labels: separates the
    # simulator's choices from differences between the datasets.
    sim = []
    for m in skillcorner.matches():
        g = load_skillcorner(m)
        ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                               Params(live_missing=0.35), seed=7)
        sim.append(dict(game=g["name"], **stats(ball, players, g["gt"], g["pitch"])))
    rows.append(summarise(sim, "gapsim_35%_on_sc"))
    print()
    print("Means over games")
    allk = ["source", "games"] + [k for k in keys if k != "game"]
    for r in rows:
        for k in allk:
            r.setdefault(k, np.nan)
    print(table(rows, allk[:10]))
    print(table(rows, [allk[0]] + allk[10:]))


if __name__ == "__main__":
    main()
