"""Run named experiment configs through the leave-one-game-out protocol and save the rows.

    python scripts/run_experiment.py e16_base [more names...]
"""
import json
import os
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mtah import experiment, features, features_v2, labels  # noqa: E402
from mtah.experiment import Config  # noqa: E402
from mtah.external import dfl, metrica, skillcorner  # noqa: E402

CONTROLLED_S = 2.0
N_REAL_VALIDATION = 8  # SkillCorner games never trained on
N_REAL_TRAIN = 12
BENCH = ("random35", "mixed41_art", "dark50")
MIX = ("random35", "mixed41", "mixed41_art", "dark50")
BASE = Config(name="e16_base")
CONFIGS = {c.name: c for c in (
    BASE,
    replace(BASE, name="e16_t05", temperature=0.5),
    replace(BASE, name="e16_t2", temperature=2.0),
    replace(BASE, name="e16_t4", temperature=4.0),
    replace(BASE, name="e17_old_train", temperature=4.0, eval_regimes=BENCH),
    replace(BASE, name="e17_mix_train", temperature=4.0, eval_regimes=BENCH,
            train_regimes=("random35", "mixed41", "mixed41_art", "dark50")),
    replace(BASE, name="e17_mix_t8", temperature=8.0, eval_regimes=BENCH,
            train_regimes=("random35", "mixed41", "mixed41_art", "dark50")),
    replace(BASE, name="e17_mix_t2", temperature=2.0, eval_regimes=BENCH,
            train_regimes=("random35", "mixed41", "mixed41_art", "dark50")),
    replace(BASE, name="e18_v2", temperature=4.0, eval_regimes=BENCH, feature_version="v2",
            train_regimes=MIX),
    # E19: decoder only, so these reuse the cached probabilities of e17_mix_train.
    replace(BASE, name="e19_min25", temperature=4.0, eval_regimes=BENCH, train_regimes=MIX,
            min_frames=25),
    replace(BASE, name="e19_min50", temperature=4.0, eval_regimes=BENCH, train_regimes=MIX,
            min_frames=50),
    replace(BASE, name="e19_t3", temperature=3.0, eval_regimes=BENCH, train_regimes=MIX),
    replace(BASE, name="e19_t6", temperature=6.0, eval_regimes=BENCH, train_regimes=MIX),
    replace(BASE, name="e19_step3", temperature=4.0, eval_regimes=BENCH, train_regimes=MIX,
            step=3),
    replace(BASE, name="e24_v2_real", temperature=4.0, eval_regimes=BENCH, feature_version="v2",
            train_regimes=MIX, real_games=N_REAL_TRAIN),
    replace(BASE, name="e22_clean", temperature=4.0, eval_regimes=BENCH, feature_version="v2",
            train_regimes=MIX, real_games=N_REAL_TRAIN, clean_ball=True),
    replace(BASE, name="e21_seeds5", temperature=4.0, eval_regimes=BENCH, train_regimes=MIX,
            model_seeds=(0, 1, 2, 3, 4)),
)}


def feature_fn(cfg):
    """The featuriser a config asks for: its feature_version names the module."""
    return features_v2.featurise if cfg.feature_version.startswith("v2") else features.featurise


def load_games(min_s=CONTROLLED_S):
    games = [dfl.load(g) for g in dfl.GAME_IDS] + [metrica.load(g) for g in metrica.GAME_IDS]
    return [g.with_labels(labels.controlled(g.labels, g.blocks, min_s)) for g in games]


def load_real(game_ids, min_s=CONTROLLED_S):
    games = [skillcorner.load(g) for g in game_ids]
    return [g.with_labels(labels.controlled(g.labels, g.blocks, min_s)) for g in games]


def real_train_games(cfg):
    ids = skillcorner.GAME_IDS[N_REAL_VALIDATION:N_REAL_VALIDATION + cfg.real_games]
    return load_real(ids)


def main(names):
    games = load_games()
    os.makedirs(os.path.join(ROOT, "out", "results"), exist_ok=True)
    for name in names:
        rows = experiment.run(games, CONFIGS[name], feature_fn(CONFIGS[name]),
                              real=real_train_games(CONFIGS[name]))
        summary = experiment.summarise(rows)
        with open(os.path.join(ROOT, "out", "results", name + ".json"), "w") as handle:
            json.dump({"rows": rows, "summary": summary}, handle, indent=1)
        by_regime = {r: sum(x["spell_f1"] for x in rows if x["regime"] == r)
                     / max(1, sum(1 for x in rows if x["regime"] == r))
                     for r in CONFIGS[name].eval_regimes}
        print("   by regime: " + "  ".join("{} {:.3f}".format(*kv) for kv in by_regime.items()))
        print("== {}  spellF1 {:.4f}  macroF1 {:.4f}  D-F1 {:.4f}".format(
            name, summary["spell_f1"], summary["macro_f1"], summary["d_f1"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
