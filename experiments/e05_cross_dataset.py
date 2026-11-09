"""E05: v1 across datasets and possession definitions.

Fit v1 (tables counted, temperature and minimum spell chosen) on one dataset, score once on the
other: DFL (7 matches) -> Metrica (3 games) and Metrica -> DFL. Both damaged by the gap simulator.
Repeated for three label definitions: touch-level (as shipped), and "controlled" possession with
spells under 2 s or 3 s handed back to the previous team (external/controlled.py).

    python experiments/e05_cross_dataset.py
"""

import numpy as np

from common import dfl_games, external_games, table
from controlled import controlled
from gapsim import Params, damage
from mtah import v1
from mtah.evidence import nearest_team
from mtah.labels import graded_blocks
from mtah.scoring import score
from mtah.smooth import majority_vote

DEFINITIONS = (0, 2, 3)
TEST_RATES = (0.15, 0.35, 0.5)


def observations(games, rates, seed0):
    out = []
    for g in games:
        for i, r in enumerate(rates):
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(live_missing=r), seed=seed0 + i)
            out.append((g, r, nearest_team(ball, players, g["teams"], g["pitch"])))
    return out


def main():
    data = {"metrica": external_games(), "dfl": dfl_games()}
    train_obs = {k: observations(v, v1.GAP_RATES, 100) for k, v in data.items()}
    test_obs = {k: observations(v, TEST_RATES, 7) for k, v in data.items()}
    rows = []
    for min_s in DEFINITIONS:
        for train, test in (("dfl", "metrica"), ("metrica", "dfl")):
            pairs = [(controlled(g["gt"], min_s), near) for g, _, near in train_obs[train]]
            model = v1.fit(pairs)
            for rate in TEST_RATES:
                f, b, n_pred, n_gt = [], [], [], []
                for g, r, near in test_obs[test]:
                    if r != rate:
                        continue
                    gt = controlled(g["gt"], min_s)
                    s = score(gt, v1.predict(near, graded_blocks(gt), model))
                    f.append(s["spell_f1_mean"])
                    n_pred.append(s["n_pred_spells"])
                    n_gt.append(s["n_gt_spells"])
                    b.append(score(gt, majority_vote(near, 50))["spell_f1_mean"])
                rows.append(dict(def_min_s=min_s, train=train, test=test, live_missing=rate,
                                 temp=model["temperature"], min_spell=model["min_spell"],
                                 v1_spellF1=float(np.mean(f)), base50=float(np.mean(b)),
                                 pred_spells=int(np.mean(n_pred)), gt_spells=int(np.mean(n_gt))))
            print("definition {} s, {} -> {} done".format(min_s, train, test), flush=True)
    print()
    print(table(rows, list(rows[0].keys())))


if __name__ == "__main__":
    main()
