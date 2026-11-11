"""Fit the current candidate (v2 + fill + HSMM, full features) in the E10 cross-dataset protocol
and dump per-frame probabilities, predictions and labels of the EXTERNAL test games, so the
errors can be analysed offline. External open data only; nothing from the competition match.

    python anatomy_fit.py <repo> <out_dir>
"""

import os
import sys

import numpy as np

repo, out_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(repo, "experiments"))
import common  # noqa: E402,F401  (sets sys.path for the repo and external/)
import e12_visibility_ablation as e12  # noqa: E402
from mtah import v1, v2  # noqa: E402
from mtah.labels import graded_blocks  # noqa: E402
from mtah.scoring import score  # noqa: E402

data = {"metrica": common.external_games(), "dfl": common.dfl_games()}
train = {k: e12.damaged(v, v1.GAP_RATES, 100) for k, v in data.items()}
test = {k: e12.damaged(v, e12.TEST_RATES, 7) + e12.damaged(v, (0.15,), 9) for k, v in data.items()}
del data
print("features ready", flush=True)

for src, dst in (("dfl", "metrica"), ("metrica", "dfl")):
    model = e12.fit(train[src], "full")
    print("{}->{}: hsmm temp {} oof {:.4f}".format(src, dst, model["hsmm_temperature"],
                                                   model["oof_spell_f1_hsmm"]), flush=True)
    dump = {}
    for i, d in enumerate(test[dst]):
        proba = v2.symmetric_proba(model["clf"], d["X"], d["names"])
        pred = v2.predict(model, d["X"], d["names"], graded_blocks(d["y"]), len(d["y"]))
        s = score(d["y"], pred)
        print("  {} {} spellF1 {:.4f}".format(d["game"], d["tag"], s["spell_f1_mean"]), flush=True)
        key = "{}|{}".format(d["game"], d["tag"])
        dump["y|" + key] = d["y"]
        dump["pred|" + key] = pred
        dump["proba|" + key] = proba.astype(np.float32)
        dump["seen|" + key] = d["X"][:, d["names"].index("ball_seen")] > 0.5
        dump["baseline|" + key] = d["baseline"]
    for j, (gt, p) in enumerate(model["oof"]):
        dump["oof_y|{}".format(j)] = gt
        dump["oof_proba|{}".format(j)] = p.astype(np.float32)
    dump["prior"] = model["prior"]
    dump["log_dur"] = model["log_dur"]
    dump["hsmm_log_trans"] = model["hsmm_log_trans"]
    dump["hsmm_temperature"] = model["hsmm_temperature"]
    np.savez_compressed(os.path.join(out_dir, "anatomy_{}_to_{}.npz".format(src, dst)), **dump)
    del model
print("done", flush=True)
