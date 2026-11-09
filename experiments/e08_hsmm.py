"""E08: v2 evidence decoded by Viterbi + merge (v2) or by the duration-aware HSMM (v2+hsmm).

Same classifier for both. The HSMM's durations and transitions are counted on the training
dataset's labels; its emission temperature is chosen on the training dataset's out-of-fold
predictions. Fit on one dataset, score once on the other, controlled-possession rule 2 s.

    python experiments/e08_hsmm.py
"""

import numpy as np

from common import dfl_games, external_games, table
from controlled import controlled
from e07_ballfill import damaged
from mtah import hsmm, v1, v2
from mtah.labels import graded_blocks
from mtah.scoring import score

MIN_S = 2.0
TEST_RATES = (0.35, 0.5)
HSMM_TEMPS = (0.1, 0.2, 0.3, 0.5, 1.0)


def hsmm_labels(proba, gt, prior, log_dur, log_trans, temp):
    emis = temp * (np.log(np.clip(proba, 1e-6, 1)) - np.log(prior))
    return hsmm.decode(emis, graded_blocks(gt), log_dur, log_trans, len(gt))


def main():
    data = {"metrica": external_games(), "dfl": dfl_games()}
    train = {k: damaged(v, v1.GAP_RATES, 100) for k, v in data.items()}
    test = {k: damaged(v, TEST_RATES, 7) for k, v in data.items()}
    print("features ready", flush=True)
    rows = []
    for src, dst in (("dfl", "metrica"), ("metrica", "dfl")):
        items = [(d["game"], d["X"], controlled(d["gt"], MIN_S), d["names"]) for d in train[src]]
        model = v2.fit(items)
        log_dur, log_trans = hsmm.count_durations([y for _, _, y, _ in items])
        best = max(HSMM_TEMPS, key=lambda T: np.mean([
            score(gt, hsmm_labels(p, gt, model["prior"], log_dur, log_trans, T))["spell_f1_mean"]
            for gt, p in model["oof"]]))
        print("{}->{}: v2 temp {} min_spell {}; hsmm temp {}".format(
            src, dst, model["temperature"], model["min_spell"], best), flush=True)
        for rate in TEST_RATES:
            acc = {"v2": [], "v2+hsmm": [], "n_v2": [], "n_hsmm": [], "n_gt": []}
            for d in test[dst]:
                if d["rate"] != rate:
                    continue
                gt = controlled(d["gt"], MIN_S)
                p = v2.symmetric_proba(model["clf"], d["X"], d["names"])
                a = score(gt, v2.decode(p, graded_blocks(gt), model["log_trans"], model["prior"],
                                        model["temperature"], model["min_spell"], len(gt)))
                b = score(gt, hsmm_labels(p, gt, model["prior"], log_dur, log_trans, best))
                acc["v2"].append(a["spell_f1_mean"])
                acc["v2+hsmm"].append(b["spell_f1_mean"])
                acc["n_v2"].append(a["n_pred_spells"])
                acc["n_hsmm"].append(b["n_pred_spells"])
                acc["n_gt"].append(a["n_gt_spells"])
            rows.append(dict(train=src, test=dst, live_missing=rate,
                             **{k: float(np.mean(v)) if k[0] == "v" else int(np.mean(v))
                                for k, v in acc.items()}))
    print()
    print(table(rows, list(rows[0].keys())))


if __name__ == "__main__":
    main()
