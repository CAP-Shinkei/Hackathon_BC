"""E13: false ball detections, the centre-spot blind spot, and a rule that ignores balls far
outside the pitch.

Two artefacts seen in the competition tracking's aggregates are missing from the simulator: false
ball detections far outside the pitch, and no ball near the centre spot before a kick-off.
gapsim.Params now has both (off by default). The rule mtah.evidence.drop_far_outside treats a ball
more than 10 m beyond the lines as not seen.

Models (v2 + fill + HSMM, fitted as the submission is):
  std            trained on the simulator as before
  std+rule       same, with the rule applied to training and test tracking
  art            trained on damage that includes the artefacts (5 false bursts per minute,
                 centre blind radius 3 m)
  art+rule       both
Test sets: fit on one dataset, score once on the other, with plain damage (35% / 50% live missing)
and with the artefacts at 2 and 10 false bursts per minute (35% live missing).

    python experiments/e13_artefacts.py
"""

import numpy as np

from common import dfl_games, external_games, table
from e12_visibility_ablation import evaluate, featurise, fit
from gapsim import Params, damage
from mtah import v1
from mtah.evidence import drop_far_outside

TRAIN_ART = dict(false_per_min=5.0, centre_blind_m=3.0)
TESTS = (("std35", 0.35, {}), ("std50", 0.5, {}),
         ("art2_35", 0.35, dict(false_per_min=2.0, centre_blind_m=3.0)),
         ("art10_35", 0.35, dict(false_per_min=10.0, centre_blind_m=3.0)))


def build(games, profiles, seed0, rule):
    """[(tag, live_missing, extra Params)] -> featurised damaged games."""
    out = []
    for g in games:
        for i, (tag, rate, extra) in enumerate(profiles):
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(live_missing=rate, **extra), seed=seed0 + i)
            if rule:
                ball = drop_far_outside(ball, g["pitch"])
            out.append(featurise(g["name"], ball, players, g["teams"], g["pitch"], g["gt"], tag))
    return out


def main():
    data = {"metrica": external_games(), "dfl": dfl_games()}
    rows = []
    for model_name, art, rule in (("std", False, False), ("std+rule", False, True),
                                  ("art", True, False), ("art+rule", True, True)):
        extra = TRAIN_ART if art else {}
        profiles = [("train", r, extra) for r in v1.GAP_RATES]
        train = {k: build(v, profiles, 100, rule) for k, v in data.items()}
        test = {k: build(v, TESTS, 7, rule) for k, v in data.items()}
        for src, dst in (("dfl", "metrica"), ("metrica", "dfl")):
            model = fit(train[src], "full")
            print("{} {}->{}: hsmm temp {}  oof {:.4f}".format(
                model_name, src, dst, model["hsmm_temperature"], model["oof_spell_f1_hsmm"]),
                flush=True)
            for tag, v in evaluate(model, test[dst], "full").items():
                v = np.array(v)
                rows.append(dict(model=model_name, train=src, test=dst, gaps=tag,
                                 spellF1=float(v[:, 0].mean()), baseline=float(v[:, 1].mean()),
                                 pred_spells=int(v[:, 2].mean()), gt_spells=int(v[:, 3].mean()),
                                 pred_D=float(v[:, 4].mean()), gt_D=float(v[:, 5].mean())))
            del model
        del train, test
    print()
    print(table(rows, list(rows[0].keys())))


if __name__ == "__main__":
    main()
