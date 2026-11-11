"""Train on the 10 clean games (damaged by the config's regimes), score on SkillCorner as is.

SkillCorner is broadcast tracking with real ball and player losses, so this is the one check of
a model against gaps nobody simulated.

    python scripts/validate_real_gaps.py e17_mix_train [more names...]
    python scripts/validate_real_gaps.py --with-real e18_v2

--with-real also trains on the SkillCorner games that are not used for validation, as loaded.
"""
import json
import os
import sys

from dataclasses import asdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from mtah import baseline, experiment, features, labels, model  # noqa: E402
from mtah.external import skillcorner  # noqa: E402
from mtah.scoring import score  # noqa: E402
from run_experiment import CONFIGS, CONTROLLED_S, feature_fn, load_games  # noqa: E402

N_VALIDATION_GAMES = 8


def load_real(game_ids):
    games = [skillcorner.load(g) for g in game_ids]
    return [g.with_labels(labels.controlled(g.labels, g.blocks, CONTROLLED_S)) for g in games]


def fit(train, real, cfg, featurise):
    """Models from the clean games under the config's regimes plus real-gap games as loaded."""
    if not real:
        return experiment.fit_fold(train, cfg, featurise)
    jobs = [(m, regime, cfg.train_seed + k) for m in train
            for k, regime in enumerate(cfg.train_regimes)] + [(m, None, 0) for m in real]
    featurised = [experiment.featurise_cached(m, regime, seed, cfg, featurise)
                  for m, regime, seed in jobs]
    swap = features.swap_index(featurised[0][1])
    models = [model.fit([X for X, _ in featurised], [m.labels for m, _, _ in jobs],
                        [m.blocks for m, _, _ in jobs], swap,
                        model.Params(**{**asdict(cfg.params), "seed": seed}))
              for seed in cfg.model_seeds]
    return models, swap


def main(args):
    with_real = "--with-real" in args
    names = [a for a in args if a != "--with-real"]
    train = load_games()
    validation = load_real(skillcorner.GAME_IDS[:N_VALIDATION_GAMES])
    real = load_real(skillcorner.GAME_IDS[N_VALIDATION_GAMES:]) if with_real else []
    suffix = "_plusreal" if with_real else ""
    floor = np.mean([score(g.labels, baseline.predict(g, 50), g.blocks)["spell_f1"]
                     for g in validation])
    print("official baseline (window 50) on real gaps: spellF1 {:.4f}".format(floor), flush=True)
    for name in names:
        cfg, featurise = CONFIGS[name], feature_fn(CONFIGS[name])
        models, swap = fit(train, real, cfg, featurise)
        rows = []
        for game in validation:
            X, _ = experiment.featurise_cached(game, None, 0, cfg, featurise)
            proba = np.mean([model.predict_proba(m, X, swap) for m in models], axis=0)
            rows.append({"game": game.game_id,
                         **experiment.decode_and_score(proba, game, train + real, cfg)})
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ("spell_f1", "macro_f1", "d_f1")}
        with open(os.path.join(ROOT, "out", "results", name + suffix + "_real.json"), "w") as handle:
            json.dump({"rows": rows, "mean": mean}, handle, indent=1)
        print("== {} on real gaps  spellF1 {spell_f1:.4f}  macroF1 {macro_f1:.4f}  "
              "D-F1 {d_f1:.4f}".format(name + suffix, **mean), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
