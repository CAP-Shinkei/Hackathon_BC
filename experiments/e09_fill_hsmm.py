"""E09: the four combinations of {no fill, straight-line ball fill} x {Viterbi + merge, HSMM}.

Same classifier per feature set; the HSMM's durations and transitions are counted on the training
labels and its temperature chosen on the training out-of-fold predictions. Fit on one dataset,
score once on the other, controlled-possession rule 2 s.

    python experiments/e09_fill_hsmm.py
"""

import numpy as np

from common import dfl_games, external_games, table
from controlled import controlled
from e07_ballfill import damaged, with_fill
from e08_hsmm import HSMM_TEMPS, hsmm_labels
from mtah import hsmm, v1, v2
from mtah.labels import graded_blocks
from mtah.scoring import score

MIN_S = 2.0
TEST_RATES = (0.35, 0.5)


def feats(d, fill):
    if not fill:
        return d["X"], d["names"]
    Xf, nf, _, _ = with_fill(d, None)
    return Xf, nf


def main():
    data = {"metrica": external_games(), "dfl": dfl_games()}
    train = {k: damaged(v, v1.GAP_RATES, 100) for k, v in data.items()}
    test = {k: damaged(v, TEST_RATES, 7) for k, v in data.items()}
    print("features ready", flush=True)
    rows = []
    for src, dst in (("dfl", "metrica"), ("metrica", "dfl")):
        labels = [controlled(d["gt"], MIN_S) for d in train[src]]
        log_dur, log_trans = hsmm.count_durations(labels)
        for fill in (False, True):
            items = [(d["game"], *feats(d, fill)[:1], y, feats(d, fill)[1])
                     for d, y in zip(train[src], labels)]
            model = v2.fit(items)
            t_h = max(HSMM_TEMPS, key=lambda T: np.mean([
                score(gt, hsmm_labels(p, gt, model["prior"], log_dur, log_trans, T))
                ["spell_f1_mean"] for gt, p in model["oof"]]))
            tag = "fill" if fill else "nofill"
            print("{}->{} {}: viterbi temp {} min_spell {}; hsmm temp {}".format(
                src, dst, tag, model["temperature"], model["min_spell"], t_h), flush=True)
            for rate in TEST_RATES:
                vit, hs, nv, nh, ng = [], [], [], [], []
                for d in test[dst]:
                    if d["rate"] != rate:
                        continue
                    gt = controlled(d["gt"], MIN_S)
                    Xg, nm = feats(d, fill)
                    p = v2.symmetric_proba(model["clf"], Xg, nm)
                    a = score(gt, v2.decode(p, graded_blocks(gt), model["log_trans"],
                                            model["prior"], model["temperature"],
                                            model["min_spell"], len(gt)))
                    b = score(gt, hsmm_labels(p, gt, model["prior"], log_dur, log_trans, t_h))
                    vit.append(a["spell_f1_mean"])
                    hs.append(b["spell_f1_mean"])
                    nv.append(a["n_pred_spells"])
                    nh.append(b["n_pred_spells"])
                    ng.append(a["n_gt_spells"])
                for dec, f, n in (("viterbi", vit, nv), ("hsmm", hs, nh)):
                    rows.append(dict(train=src, test=dst, fill=tag, decoder=dec,
                                     live_missing=rate, spellF1=float(np.mean(f)),
                                     pred_spells=int(np.mean(n)), gt_spells=int(np.mean(ng))))
    print()
    print(table(rows, list(rows[0].keys())))


if __name__ == "__main__":
    main()
