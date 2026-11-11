"""Train the possession bundle on the external games and save it.

    python scripts/train.py --out model/bundle.joblib

Games are loaded exactly as scripts/run_experiment.py loads them (controlled possession, 2 s).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
for folder in (ROOT, SCRIPTS):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from mtah import gapsim, model, submission  # noqa: E402
from mtah.external import skillcorner  # noqa: E402
from run_experiment import load_games, load_real  # noqa: E402

DEFAULT_REGIMES = ("random35", "mixed41", "mixed41_art", "dark50")
DEFAULT_MODEL_SEEDS = (0,)
DEFAULT_TEMPERATURE = 4.0
DEFAULT_STEP = 5
DEFAULT_MIN_FRAMES = 0
DEFAULT_FEATURES = "features_v2"
DEFAULT_REAL_GAMES = len(skillcorner.GAME_IDS)
DEFAULT_OUT = os.path.join(ROOT, "model", "bundle.joblib")
TRAIN_SEED = 1


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES),
                        choices=sorted(gapsim.REGIMES), help="damage regimes to train under")
    parser.add_argument("--model-seeds", nargs="+", type=int, default=list(DEFAULT_MODEL_SEEDS))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--features", default=DEFAULT_FEATURES, help="feature module in mtah/")
    parser.add_argument("--real-games", type=int, default=DEFAULT_REAL_GAMES,
                        help="SkillCorner games (real gaps) to add, taken from the end of the list")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to save the bundle")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    games = load_games()
    real = load_real(skillcorner.GAME_IDS[len(skillcorner.GAME_IDS) - args.real_games:])
    bundle = submission.train_bundle(
        games, tuple(args.regimes), TRAIN_SEED, tuple(args.model_seeds), model.Params(),
        args.temperature, args.step, args.min_frames, feature_module=args.features,
        real_matches=real)
    submission.save_bundle(bundle, args.out)
    print("trained {} model(s) on {} games under {} regime(s) plus {} real-gap games, "
          "{} features".format(len(bundle.models), len(games), len(args.regimes), len(real),
                               len(bundle.names)))
    print("saved {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
