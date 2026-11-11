"""E27: a small sequence model (transformer) over the per-frame clues, alone and stacked with the
gradient-boosting classifier. The model is mtah/sequence.py.

Protocol: E25's test sets and E25's fitting scheme. Games are split in the same three folds;
each fold's model predicts its held-out training games (out-of-fold, for choosing the HSMM
temperature) and the test games; the test prediction is the mean of the three fold models.

This script trains and saves the probabilities (out/e27/<name>_*.npz); e27_stack.py decodes
and scores them, alone and stacked with a gradient-boosting variant of E25.

    python experiments/e27_sequence.py            (needs PyTorch; --no-aux drops the two
                                                   training-only outputs)
    python experiments/e27_stack.py nn mix+ctx+turn
"""

import argparse
import os
import time

import numpy as np
import torch

import e25_benchmark as e25
from common import REPO, dfl_games, external_games, skillcorner_games
from gapsim import Params, damage
from mtah import sequence
from mtah.labels import X as UNGRADED

OUT = os.path.join(REPO, "out", "e27")


def copies(games, profiles):
    """E25's featurised damaged copies plus the hidden-ball target."""
    out = []
    for g in games:
        for tag, fields, seed in profiles:
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(**fields), seed=seed)
            d = e25.featurise(g["name"], ball, players, g["teams"], g["pitch"], g["gt"], tag)
            d["aux"] = sequence.true_distances(g["ball"], players, g["teams"], g["pitch"])
            out.append(d)
    return out


def fit_predict(train, tests, families, epochs, aux, log):
    """({train key: out-of-fold proba}, [test proba]): the test prediction is the mean of the
    three fold models."""
    cols = [i for i, f in enumerate(train[0]["family"]) if f in families]
    names = [train[0]["names"][i] for i in cols]
    oof, models = sequence.fit_folds(train, cols, epochs, aux, log)
    return oof, [sequence.ensemble(models, d["X"][:, cols], names) for d in tests]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--families", default="base+ctx+turn")
    parser.add_argument("--epochs", type=int, default=12,
                        help="passes over the damaged copies (five copies of each game)")
    parser.add_argument("--no-aux", action="store_true")
    parser.add_argument("--cells", default="cross", choices=("cross", "sc", "all"))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--name", default="nn")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    families = tuple(args.families.split("+"))
    log = lambda s: print(s, flush=True)  # noqa: E731
    path = lambda src, dst: os.path.join(OUT, "{}_{}_to_{}.npz".format(  # noqa: E731
        args.name, src, dst))

    profiles = [(t, f, 100 + i) for i, (t, f) in enumerate(e25.MIX)]
    load = {"metrica": external_games, "dfl": dfl_games}
    wanted = [("dfl", "metrica"), ("metrica", "dfl")] if args.cells in ("cross", "all") else []
    if args.cells in ("sc", "all"):
        wanted.append(("metrica+dfl", "skillcorner"))
    for src, dst in wanted:
        if os.path.isfile(path(src, dst)):  # finished in an earlier run
            continue
        # One direction in memory at a time.
        if dst == "skillcorner":
            tr = copies(external_games() + dfl_games(), profiles)
            te = []
            for g in skillcorner_games()[:e25.SC_HELD_OUT]:
                d = e25.featurise(g["name"], g["ball"], g["players"], g["teams"], g["pitch"],
                                  g["gt"], "sc_real")
                d["aux"] = np.full((len(d["y"]), 2), np.nan, np.float32)
                te.append(d)
        else:
            tr, te = copies(load[src](), profiles), copies(load[dst](), e25.TESTS)
        log("{}->{}: features ready".format(src, dst))
        t0 = time.time()
        oof, test_p = fit_predict(tr, te, families, args.epochs, not args.no_aux, log)
        np.savez_compressed(
            path(src, dst),
            **{"oof_proba|" + k: p.astype(np.float32) for k, p in oof.items()},
            **{"proba|{}|{}".format(d["game"], d["tag"]): p.astype(np.float32)
               for d, p in zip(te, test_p)})
        graded = [(d["y"] != UNGRADED, d["y"], p) for d, p in zip(te, test_p)]
        log("{}->{}: saved, {:.0f} s; frame accuracy on the test copies {:.3f}".format(
            src, dst, time.time() - t0,
            float(np.mean([np.mean(p[g].argmax(axis=1) == y[g]) for g, y, p in graded]))))
        del tr, te, oof, test_p


if __name__ == "__main__":
    main()
