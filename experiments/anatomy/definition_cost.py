"""How much spellF1 does the 'controlled possession' threshold alone move? External labels only.

Truth = controlled(gt, a), prediction = controlled(gt, b): a perfect model that simply uses a
different threshold than the answer key. Also: perfect labels with every boundary shifted.

    python definition_cost.py <repo>
"""

import os
import sys

import numpy as np

repo = sys.argv[1]
sys.path.insert(0, os.path.join(repo, "experiments"))
import common  # noqa: E402
from controlled import controlled  # noqa: E402
from mtah.labels import runs  # noqa: E402
from mtah.scoring import score  # noqa: E402

NS = (0.0, 1.0, 2.0, 3.0, 5.0)
sets = {"metrica": common.external_games(), "dfl": common.dfl_games()}
for name, games in sets.items():
    gts = [g["gt"] for g in games]
    print("\n== {} ({} games): spellF1 of a perfect model using threshold b (columns) when the "
          "key uses a (rows)".format(name, len(gts)))
    print("  a\\b  " + "  ".join("{:>5.0f}s".format(b) for b in NS) + "   spells/game  mean_s  per_min_live")
    for a in NS:
        row = []
        for b in NS:
            row.append(np.mean([score(controlled(gt, a), controlled(gt, b))["spell_f1_mean"]
                                for gt in gts]))
        n_sp, dur, per_min = [], [], []
        for gt in gts:
            c = controlled(gt, a)
            sp = [(b_ - a_) for k, a_, b_ in runs(c) if k in (0, 1)]
            n_sp.append(len(sp))
            dur.append(np.mean(sp) / 25)
            per_min.append(len(sp) / (np.sum(c != 3) / 25 / 60))
        print("  {:>3.0f}s  ".format(a) + "  ".join("{:6.3f}".format(v) for v in row)
              + "   {:8.0f}  {:6.1f}  {:6.2f}".format(np.mean(n_sp), np.mean(dur), np.mean(per_min)))
    # touch-level spell length distribution: how many spells would each threshold remove
    short = []
    for gt in gts:
        L = np.array([(b_ - a_) / 25 for k, a_, b_ in runs(gt) if k in (0, 1)])
        short.append([np.mean(L < t) for t in (1, 2, 3, 5)])
    print("  share of touch-level spells shorter than 1/2/3/5 s: "
          + " / ".join("{:.2f}".format(v) for v in np.mean(short, axis=0)))
