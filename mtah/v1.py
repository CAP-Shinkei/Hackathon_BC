"""v1: nearest player -> team-symmetric HMM -> Viterbi -> merge short spells.

All fitted quantities come from the external games (Metrica sample data), each damaged with the
gap simulator at three gap rates:
  - the emission and transition tables are counted;
  - the emission temperature and the minimum spell length are chosen on the grid below by the
    selection rule (mean spellF1 over the damaged external games).
"""

import itertools

import numpy as np

from mtah.evidence import nearest_team
from mtah.hmm import count_tables, decode, merge_short
from mtah.labels import graded_blocks
from mtah.scoring import score

GAP_RATES = (0.15, 0.35, 0.5)
TEMPERATURES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
MIN_SPELLS = (1, 25, 50, 75, 100)


def damaged_observations(games, damage, params_cls, seed0=100):
    """[(gt, nearest-team codes)] for every external game x gap rate."""
    pairs = []
    for g in games:
        for i, rate in enumerate(GAP_RATES):
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   params_cls(live_missing=rate), seed=seed0 + i)
            pairs.append((g["gt"], nearest_team(ball, players, g["teams"], g["pitch"])))
    return pairs


def predict(near, blocks, model):
    path = decode(near, blocks, model["log_emis"], model["log_trans"], model["temperature"])
    return merge_short(path, blocks, model["min_spell"])


def fit(pairs):
    log_emis, log_trans = count_tables(pairs)
    best = None
    for temp, ms in itertools.product(TEMPERATURES, MIN_SPELLS):
        model = {"log_emis": log_emis, "log_trans": log_trans, "temperature": temp,
                 "min_spell": ms}
        f = float(np.mean([score(gt, predict(near, graded_blocks(gt), model))["spell_f1_mean"]
                           for gt, near in pairs]))
        if best is None or f > best[0]:
            best = (f, model)
    best[1]["external_spell_f1"] = best[0]
    return best[1]
