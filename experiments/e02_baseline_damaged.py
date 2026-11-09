"""E02: the official baseline method on gap-damaged external games, across simulator settings.

    python experiments/e02_baseline_damaged.py
"""

import numpy as np

from common import external_games, table
from gapsim import Params, damage, describe
from mtah.evidence import nearest_team
from mtah.scoring import score
from mtah.smooth import majority_vote

SETTINGS = {
    "clean": None,
    "live15": Params(live_missing=0.15),
    "live35": Params(live_missing=0.35),
    "live50": Params(live_missing=0.50),
    "live35_nodeadsight": Params(live_missing=0.35, dead_visible=0.0),
}
WINDOWS = (0, 25, 50, 125)


def main():
    games = external_games()
    rows, sim = [], []
    for name, params in SETTINGS.items():
        for g in games:
            ball, players = g["ball"], g["players"]
            if params is not None:
                ball, players = damage(ball, players, g["teams"], g["gt"], g["pitch"], params,
                                       seed=1)
            sim.append(dict(setting=name, game=g["name"], **describe(ball, g["gt"])))
            near = nearest_team(ball, players, g["teams"], g["pitch"])
            for w in WINDOWS:
                s = score(g["gt"], majority_vote(near, w))
                rows.append(dict(setting=name, game=g["name"], window=w, **s))
    print(table(sim, list(sim[0].keys())))
    print()
    summary = []
    for name in SETTINGS:
        for w in WINDOWS:
            sel = [r for r in rows if r["setting"] == name and r["window"] == w]
            summary.append({"setting": name, "window": w,
                            "spellF1": float(np.mean([r["spell_f1_mean"] for r in sel])),
                            "macroF1": float(np.mean([r["macro_f1"] for r in sel])),
                            "D_F1": float(np.mean([r["d_f1"] for r in sel])),
                            "pred_spells": int(np.mean([r["n_pred_spells"] for r in sel])),
                            "gt_spells": int(np.mean([r["n_gt_spells"] for r in sel]))})
    print(table(summary, list(summary[0].keys())))


if __name__ == "__main__":
    main()
