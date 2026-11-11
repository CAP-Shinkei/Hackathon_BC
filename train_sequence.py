"""Train the sequence model that make_submission.py --stack adds to the gradient-boosting
classifier (E27), and choose the stacked HSMM temperature.

    python train_sequence.py            # writes model/sequence_mix_ctx_turn.pt  (needs PyTorch)

Same external training data as `make_submission.py --model v2 --min-s 2 --fill --drop-outside
--regimes mix --context --turnover`. Games are split in the three folds of mtah.v2.fit; each
fold gives one sequence model, and the three are kept and averaged at prediction time. The HSMM
temperature of the stack is chosen on the out-of-fold probabilities of both models, as
mtah.v2.add_hsmm does for the classifier alone.

The weights file is committed: the submission is regenerated from it, not from a retraining
(a retraining gives the same numbers only on the same machine and thread count).
"""

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "external"))
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import dfl_games, external_games  # noqa: E402
from controlled import controlled  # noqa: E402
from gapsim import TRAIN_EXTRA, Params, damage  # noqa: E402
from mtah import hsmm, v1, v2  # noqa: E402
from mtah.evidence import drop_far_outside  # noqa: E402
from mtah.labels import graded_blocks  # noqa: E402
from mtah.scoring import score  # noqa: E402


def _decode_score(args):
    y, proba, prior, log_dur, log_trans, temperature = args
    pred = v2.hsmm_decode(proba, graded_blocks(y), prior, log_dur, log_trans, temperature, len(y))
    return score(y, pred)["spell_f1_mean"]


def main():
    import torch
    from make_submission import v2_features
    from mtah import sequence

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-s", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=12,
                        help="passes over the damaged copies (five copies of each game), as E27")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--out", default=os.path.join(HERE, "model", "sequence_mix_ctx_turn.pt"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    log = lambda s: print(s, flush=True)  # noqa: E731
    pool = ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn"))

    profiles = [dict(live_missing=rate) for rate in v1.GAP_RATES] + list(TRAIN_EXTRA)
    train = []
    for g in external_games() + dfl_games():
        for i, fields in enumerate(profiles):
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(**fields), seed=100 + i)
            aux = sequence.true_distances(g["ball"], players, g["teams"], g["pitch"])
            ball = drop_far_outside(ball, g["pitch"])
            Xg, names = v2_features(ball, players, g["teams"], g["pitch"],
                                    graded_blocks(g["gt"]), True, True, True)
            train.append(dict(game=g["name"], tag=str(i), X=Xg, names=names,
                              y=controlled(g["gt"], args.min_s), gt=g["gt"], aux=aux))
    log("features ready: {} copies, {} clues".format(len(train), len(names)))

    # Out-of-fold probabilities of the classifier, as mtah.v2.fit computes them.
    ids = sorted({d["game"] for d in train})
    gbm = {}
    for fold in [ids[i::3] for i in range(3)]:
        clf = v2.train_classifier([(d["X"], d["y"], names) for d in train
                                   if d["game"] not in fold])
        for d in train:
            if d["game"] in fold:
                gbm["{}|{}".format(d["game"], d["tag"])] = v2.symmetric_proba(clf, d["X"], names)
    log("classifier out-of-fold probabilities ready")
    oof, models = sequence.fit_folds(train, list(range(len(names))), args.epochs, True, log)

    labels = [d["y"] for d in train]
    prior = v2.class_prior(labels)
    log_dur, log_trans = hsmm.count_durations(labels)
    keys = ["{}|{}".format(d["game"], d["tag"]) for d in train]
    best = {}
    for label, proba in (("classifier", gbm), ("sequence", oof),
                         ("stack", {k: sequence.fuse(gbm[k], oof[k]) for k in keys})):
        tasks = [(d["y"], proba[k], prior, log_dur, log_trans, T)
                 for T in v2.HSMM_TEMPS for d, k in zip(train, keys)]
        f1 = np.array(list(pool.map(_decode_score, tasks))).reshape(len(v2.HSMM_TEMPS), -1)
        f1 = f1.mean(axis=1)
        best[label] = (v2.HSMM_TEMPS[int(np.argmax(f1))], float(f1.max()))
        log("{}: hsmm temperature {}, out-of-fold external spellF1 {:.4f}".format(
            label, *best[label]))
    pool.shutdown()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sequence.save(args.out, models, names, temperature=best["stack"][0],
                  oof_spell_f1=best["stack"][1], min_s=args.min_s, epochs=args.epochs)
    log("wrote {}".format(os.path.relpath(args.out, HERE)))


if __name__ == "__main__":
    main()
