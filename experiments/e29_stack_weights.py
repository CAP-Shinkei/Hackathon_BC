"""E29: the stacking weight is a choice, measure it - geometric-mean sweep GBM x sequence model.

Álvaro's reading of E27 (2026-11-11): the stack fuses the gradient-boosting classifier and the
sequence model with a FIXED 50/50 geometric mean (e27_stack.fuse: "No weight is fitted"), so
half of the fused score comes from a model that alone loses to candidate 1. A more conservative
mix (e.g. 75% classifier / 25% sequence model) could keep the dark-stoppage gain without paying
for it on the other cells. This script measures that, on the saved probabilities only: no
training, no new features, the competition match is not involved.

Weights swept: 0 / 25 / 50 / 75 / 100 % of the sequence model (0 and 100 recover each model
alone, so the table also separates "better model" from "better mix").

Protocol, fixed before running:
  - the weight is chosen on the OUT-OF-FOLD probabilities of the training games only (the same
    internal selection the HSMM temperature already uses), never by looking at the test cells.
    The oof_proba/oof_y entries of the E25/E27 dumps come from classifiers whose folds exclude
    the game being predicted, and the outer-test games of each run (the dst dataset) never
    appear in them. The full test sweep is printed to read the curve, but picking the
    best-looking weight from it would make the number optimistic, so the reported weight is
    the out-of-fold one.
  - HSMM, duration tables, priors, test sets and seeds: e27_stack's, unchanged. For each weight
    the temperature is chosen out-of-fold as in E27.
  - selection number = mean spellF1 of the std35/std50 cells, comparable with candidate 1's
    0.6742 and the 50/50 stack's 0.6745.

Proposed decision rule (a proposal for the team, not a validated result): a weight other than
50/50 is worth keeping only if, under this same protocol, it beats the 50/50 row and the w=0
row (the classifier alone) on the selection cells, without a clear loss on the dark and sc_real
cells. The E27 figures (0.6745 for the stack, 0.6742 for candidate 1) orient that comparison;
they are not fixed thresholds.

Needs the E25 and E27 dumps (out/e25/<variant>_*.npz, out/e27/<name>_*.npz); it writes no
dumps of its own, only out/e27/weights_<name>_<variant>.json. Minutes, not hours.

Status: written 2026-11-11, NOT RUN - no results yet.

    python experiments/e29_stack_weights.py nn mix+ctx+turn
"""

import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import e25_benchmark as e25
from common import REPO, table
from mtah import v2

NN_OUT = os.path.join(REPO, "out", "e27")
RUNS = (("dfl", "metrica"), ("metrica", "dfl"), ("metrica+dfl", "skillcorner"))
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)  # share of the sequence model in the geometric mean


def fuse_weighted(p, q, w):
    """Geometric mean of two probability rows, weight w on q (the sequence model).
    w=0 -> p, w=1 -> q, w=0.5 -> e27_stack.fuse."""
    z = (1.0 - w) * np.log(np.clip(p, 1e-6, 1)) + w * np.log(np.clip(q, 1e-6, 1))
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def main(name, variant):
    pool = ProcessPoolExecutor(12, mp_context=multiprocessing.get_context("spawn"))
    rows, per_game, chosen = [], {}, {}
    for src, dst in RUNS:
        path = os.path.join(NN_OUT, "{}_{}_to_{}.npz".format(name, src, dst))
        if not os.path.isfile(path):
            continue
        nn = np.load(path)
        gb = np.load(os.path.join(e25.OUT, "{}_{}_to_{}.npz".format(variant, src, dst)))
        prior, log_dur, log_trans = gb["prior"], gb["log_dur"], gb["log_trans"]
        train = [k[len("oof_proba|"):] for k in nn.files if k.startswith("oof_proba|")]
        tests = [k[len("proba|"):] for k in nn.files if k.startswith("proba|")]
        for w in WEIGHTS:
            fuse = lambda k, pre: fuse_weighted(gb[pre + k], nn[pre + k], w)  # noqa: E731
            tasks = [(gb["oof_y|" + k], fuse(k, "oof_proba|").astype(float), prior, log_dur,
                      log_trans, T) for T in v2.HSMM_TEMPS for k in train]
            f1 = np.array([s["spell_f1_mean"] for _, s in pool.map(e25._decode_score, tasks)])
            f1 = f1.reshape(len(v2.HSMM_TEMPS), len(train)).mean(axis=1)
            T = v2.HSMM_TEMPS[int(np.argmax(f1))]
            oof = float(f1.max())  # weight and temperature see only these training games
            done = list(pool.map(e25._decode_score, [
                (gb["y|" + k], fuse(k, "proba|").astype(float), prior, log_dur, log_trans, T)
                for k in tests]))
            for k, (_, s) in zip(tests, done):
                per_game["w{:.2f}|{}".format(w, k)] = s["spell_f1_mean"]
            for tag in dict.fromkeys(k.split("|")[1] for k in tests):
                v = [s for k, (_, s) in zip(tests, done) if k.split("|")[1] == tag]
                rows.append(dict(w=w, train=src, test=dst, gaps=tag, hsmm_temp=T, oof=oof,
                                 spell_f1=float(np.mean([s["spell_f1_mean"] for s in v])),
                                 macro_f1=float(np.mean([s["macro_f1"] for s in v])),
                                 n_pred=float(np.mean([s["n_pred_spells"] for s in v])),
                                 n_gt=float(np.mean([s["n_gt_spells"] for s in v]))))
            print("w={:.2f} {}->{}: hsmm temp {}, oof {:.4f}".format(w, src, dst, T, oof),
                  flush=True)
        # the reported weight is the out-of-fold choice, made before any test number exists
        oof_by_w = {w: float(np.mean([r["oof"] for r in rows
                                      if r["w"] == w and r["train"] == src])) for w in WEIGHTS}
        chosen["{}_to_{}".format(src, dst)] = max(WEIGHTS, key=lambda w: (oof_by_w[w], -w))
    pool.shutdown()
    out = os.path.join(NN_OUT, "weights_{}_{}.json".format(name, variant))
    with open(out, "w") as handle:
        json.dump(dict(rows=rows, per_game=per_game, chosen_on_oof=chosen), handle, indent=1)
    print()
    print(table(rows, list(rows[0].keys())))
    for w in WEIGHTS:
        mine = [r for r in rows if r["w"] == w]
        sel = [r["spell_f1"] for r in mine if r["gaps"] in e25.SELECTION]
        if sel:
            extra = " ".join("{} {:.4f}".format(
                tag, float(np.mean([r["spell_f1"] for r in mine if r["gaps"] == tag])))
                for tag in ("art10", "dark", "sc_real"))
            print("== w={:.2f}: selection {:.4f} | {}".format(w, float(np.mean(sel)), extra))
    print()
    for run, w in chosen.items():
        print("== {}: weight chosen out-of-fold {:.2f}".format(run, w))


if __name__ == "__main__":
    main(*sys.argv[1:3])
