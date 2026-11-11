"""Train on the 10 clean games (damaged by the config's regimes), score on SkillCorner as is.

SkillCorner is broadcast tracking with real ball and player losses, so this is the one check of
a model against gaps nobody simulated.

    python scripts/validate_real_gaps.py e17_mix_train [more names...]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from mtah import baseline, experiment, labels, model  # noqa: E402
from mtah.external import skillcorner  # noqa: E402
from mtah.scoring import score  # noqa: E402
from run_experiment import CONFIGS, CONTROLLED_S, feature_fn, load_games  # noqa: E402

N_VALIDATION_GAMES = 8


def load_validation():
    games = [skillcorner.load(g) for g in skillcorner.GAME_IDS[:N_VALIDATION_GAMES]]
    return [g.with_labels(labels.controlled(g.labels, g.blocks, CONTROLLED_S)) for g in games]


def main(names):
    train, validation = load_games(), load_validation()
    floor = np.mean([score(g.labels, baseline.predict(g, 50), g.blocks)["spell_f1"]
                     for g in validation])
    print("official baseline (window 50) on real gaps: spellF1 {:.4f}".format(floor), flush=True)
    for name in names:
        cfg, featurise = CONFIGS[name], feature_fn(CONFIGS[name])
        models, swap = experiment.fit_fold(train, cfg, featurise)
        rows = []
        for game in validation:
            X, _ = experiment.featurise_cached(game, None, 0, cfg, featurise)
            proba = np.mean([model.predict_proba(m, X, swap) for m in models], axis=0)
            rows.append({"game": game.game_id,
                         **experiment.decode_and_score(proba, game, train, cfg)})
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ("spell_f1", "macro_f1", "d_f1")}
        with open(os.path.join(ROOT, "out", "results", name + "_real.json"), "w") as handle:
            json.dump({"rows": rows, "mean": mean}, handle, indent=1)
        print("== {} on real gaps  spellF1 {spell_f1:.4f}  macroF1 {macro_f1:.4f}  "
              "D-F1 {d_f1:.4f}".format(name, **mean), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
