"""E03: v1 = nearest player -> team-symmetric HMM (counted tables) -> Viterbi -> merge short spells.

Leave-one-game-out: for each external test game, the tables are counted and the temperature and
minimum spell length are chosen on the other two games (damaged with training seeds), then the
test game is scored once (damaged with a different seed).

    python experiments/e03_hmm_v1.py
"""

import itertools

import numpy as np

from common import external_games, table
from gapsim import Params, damage
from mtah.evidence import nearest_team
from mtah.hmm import count_tables, decode, merge_short
from mtah.labels import graded_blocks
from mtah.scoring import score
from mtah.smooth import majority_vote

TRAIN_SETTINGS = [Params(live_missing=r) for r in (0.15, 0.35, 0.5)]
TEST_SETTINGS = {"live15": Params(live_missing=0.15), "live35": Params(live_missing=0.35),
                 "live50": Params(live_missing=0.5)}
TEMPERATURES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
MIN_SPELLS = (1, 25, 50, 75, 100)


def nearest(g, params, seed):
    ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"], params,
                           seed=seed)
    return nearest_team(ball, players, g["teams"], g["pitch"])


def run(near, gt, tables, temperature, min_spell):
    blocks = graded_blocks(gt)
    path = decode(near, blocks, *tables, temperature=temperature)
    return merge_short(path, blocks, min_spell)


def main():
    games = external_games()
    train_obs = {g["name"]: [nearest(g, p, seed=100 + i) for i, p in enumerate(TRAIN_SETTINGS)]
                 for g in games}
    rows = []
    for test in games:
        train = [g for g in games if g is not test]
        pairs = [(g["gt"], near) for g in train for near in train_obs[g["name"]]]
        tables = count_tables(pairs)
        grid = []
        for temp, ms in itertools.product(TEMPERATURES, MIN_SPELLS):
            f = np.mean([score(gt, run(near, gt, tables, temp, ms))["spell_f1_mean"]
                         for gt, near in pairs])
            grid.append((f, temp, ms))
        f_train, temp, ms = max(grid)
        for name, params in TEST_SETTINGS.items():
            near = nearest(test, params, seed=7)
            s = score(test["gt"], run(near, test["gt"], tables, temp, ms))
            b = score(test["gt"], majority_vote(near, 50))
            rows.append(dict(test=test["name"], setting=name, temp=temp, min_spell=ms,
                             train_spellF1=float(f_train), spellF1=s["spell_f1_mean"],
                             base50_spellF1=b["spell_f1_mean"], macroF1=s["macro_f1"],
                             D_F1=s["d_f1"], pred_spells=s["n_pred_spells"],
                             gt_spells=s["n_gt_spells"]))
        print("test {}: chose temperature {} min_spell {} (train spellF1 {:.4f})".format(
            test["name"], temp, ms, f_train), flush=True)
    print()
    print(table(rows, list(rows[0].keys())))
    print()
    for name in TEST_SETTINGS:
        sel = [r for r in rows if r["setting"] == name]
        print("{}: v1 spellF1 {:.4f}  baseline_smooth50 {:.4f}  v1 macroF1 {:.4f}".format(
            name, np.mean([r["spellF1"] for r in sel]),
            np.mean([r["base50_spellF1"] for r in sel]), np.mean([r["macroF1"] for r in sel])))


if __name__ == "__main__":
    main()
