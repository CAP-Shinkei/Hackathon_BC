"""E06: v2 (feature classifier + hybrid HMM) across datasets, against v1 and the baseline.

Same protocol as E05: fit on one dataset (DFL or Metrica, damaged at 3 gap rates), score once on
the other (damaged with a different seed).

    python experiments/e06_v2_cross_dataset.py [min_s ...]
"""

import sys

import numpy as np

from common import dfl_games, external_games, table
from controlled import controlled
from gapsim import Params, damage
from mtah import features, v1, v2
from mtah.evidence import nearest_team
from mtah.labels import graded_blocks
from mtah.scoring import score
from mtah.smooth import majority_vote

TEST_RATES = (0.15, 0.35, 0.5)


def prepare(games, rates, seed0):
    out = []
    for g in games:
        for i, r in enumerate(rates):
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(live_missing=r), seed=seed0 + i)
            near = nearest_team(ball, players, g["teams"], g["pitch"])
            Xg, names = features.compute(ball, players, g["teams"], g["pitch"], near)
            out.append({"game": g["name"], "rate": r, "gt": g["gt"], "near": near, "X": Xg,
                        "names": names})
    return out


def main():
    definitions = [float(a) for a in sys.argv[1:]] or [2.0]
    data = {"metrica": external_games(), "dfl": dfl_games()}
    train = {k: prepare(v, v1.GAP_RATES, 100) for k, v in data.items()}
    test = {k: prepare(v, TEST_RATES, 7) for k, v in data.items()}
    print("features ready", flush=True)
    rows = []
    for min_s in definitions:
        for src, dst in (("dfl", "metrica"), ("metrica", "dfl")):
            items = [(d["game"], d["X"], controlled(d["gt"], min_s), d["names"])
                     for d in train[src]]
            m2 = v2.fit(items)
            m1 = v1.fit([(controlled(d["gt"], min_s), d["near"]) for d in train[src]])
            print("def {} {}->{}: v2 temp {} min_spell {} (oof {:.4f})".format(
                min_s, src, dst, m2["temperature"], m2["min_spell"], m2["oof_spell_f1"]),
                flush=True)
            for rate in TEST_RATES:
                acc = {"v2": [], "v1": [], "base50": [], "n2": [], "ngt": []}
                for d in test[dst]:
                    if d["rate"] != rate:
                        continue
                    gt = controlled(d["gt"], min_s)
                    blocks = graded_blocks(gt)
                    s2 = score(gt, v2.predict(m2, d["X"], d["names"], blocks, len(gt)))
                    acc["v2"].append(s2["spell_f1_mean"])
                    acc["n2"].append(s2["n_pred_spells"])
                    acc["ngt"].append(s2["n_gt_spells"])
                    acc["v1"].append(score(gt, v1.predict(d["near"], blocks, m1))["spell_f1_mean"])
                    acc["base50"].append(score(gt, majority_vote(d["near"], 50))["spell_f1_mean"])
                rows.append(dict(def_min_s=min_s, train=src, test=dst, live_missing=rate,
                                 v2=float(np.mean(acc["v2"])), v1=float(np.mean(acc["v1"])),
                                 base50=float(np.mean(acc["base50"])),
                                 v2_spells=int(np.mean(acc["n2"])), gt_spells=int(np.mean(acc["ngt"]))))
    print()
    print(table(rows, list(rows[0].keys())))


if __name__ == "__main__":
    main()
