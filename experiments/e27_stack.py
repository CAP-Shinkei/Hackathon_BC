"""E27, stacking from saved probabilities: the geometric mean of the sequence model's
probabilities (out/e27/<name>_*.npz, written by e27_sequence.py) and a gradient-boosting
variant's (out/e25/<variant>_*.npz, written by e25_benchmark.py), decoded by the same HSMM with
the temperature chosen on the out-of-fold probabilities of the training games. No weight is
fitted: both models count the same.

    python experiments/e27_stack.py nn mix+ctx+turn
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


def fuse(p, q):
    z = 0.5 * (np.log(np.clip(p, 1e-6, 1)) + np.log(np.clip(q, 1e-6, 1)))
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def main(name, variant):
    pool = ProcessPoolExecutor(12, mp_context=multiprocessing.get_context("spawn"))
    rows, per_game = [], {}
    for src, dst in RUNS:
        path = os.path.join(NN_OUT, "{}_{}_to_{}.npz".format(name, src, dst))
        if not os.path.isfile(path):
            continue
        nn = np.load(path)
        gb = np.load(os.path.join(e25.OUT, "{}_{}_to_{}.npz".format(variant, src, dst)))
        prior, log_dur, log_trans = gb["prior"], gb["log_dur"], gb["log_trans"]
        train = [k[len("oof_proba|"):] for k in nn.files if k.startswith("oof_proba|")]
        tests = [k[len("proba|"):] for k in nn.files if k.startswith("proba|")]
        models = (("gbm", lambda k, pre: gb[pre + k]), ("nn", lambda k, pre: nn[pre + k]),
                  ("gbm x nn", lambda k, pre: fuse(gb[pre + k], nn[pre + k])))
        for label, proba in models:
            tasks = [(gb["oof_y|" + k], proba(k, "oof_proba|").astype(float), prior, log_dur,
                      log_trans, T) for T in v2.HSMM_TEMPS for k in train]
            f1 = np.array([s["spell_f1_mean"] for _, s in pool.map(e25._decode_score, tasks)])
            f1 = f1.reshape(len(v2.HSMM_TEMPS), len(train)).mean(axis=1)
            T = v2.HSMM_TEMPS[int(np.argmax(f1))]
            done = list(pool.map(e25._decode_score, [
                (gb["y|" + k], proba(k, "proba|").astype(float), prior, log_dur, log_trans, T)
                for k in tests]))
            for k, (_, s) in zip(tests, done):
                per_game["{}|{}".format(label, k)] = s["spell_f1_mean"]
            for tag in dict.fromkeys(k.split("|")[1] for k in tests):
                v = [s for k, (_, s) in zip(tests, done) if k.split("|")[1] == tag]
                rows.append(dict(model=label, train=src, test=dst, gaps=tag, hsmm_temp=T,
                                 oof=float(f1.max()),
                                 spell_f1=float(np.mean([s["spell_f1_mean"] for s in v])),
                                 macro_f1=float(np.mean([s["macro_f1"] for s in v])),
                                 n_pred=float(np.mean([s["n_pred_spells"] for s in v])),
                                 n_gt=float(np.mean([s["n_gt_spells"] for s in v])),
                                 team_acc=float(np.mean([s["team_acc_both_alive"] for s in v]))))
            print("{} {}->{}: hsmm temp {}, oof {:.4f}".format(label, src, dst, T, f1.max()),
                  flush=True)
    pool.shutdown()
    with open(os.path.join(NN_OUT, "stack_{}_{}.json".format(name, variant)), "w") as handle:
        json.dump(dict(rows=rows, per_game=per_game), handle, indent=1)
    print()
    print(table(rows, list(rows[0].keys())))
    for label in dict.fromkeys(r["model"] for r in rows):
        for tags, what in ((e25.SELECTION, "selection"), (("art10",), "art10"),
                           (("dark",), "dark"), (("sc_real",), "sc_real")):
            v = [r["spell_f1"] for r in rows if r["model"] == label and r["gaps"] in tags]
            if v:
                print("== {}: {} {:.4f}".format(label, what, float(np.mean(v))))


if __name__ == "__main__":
    main(*sys.argv[1:3])
