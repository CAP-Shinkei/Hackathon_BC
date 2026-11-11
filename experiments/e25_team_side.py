"""E25, the team's half: fit their pipeline (CAP-Shinkei/Hackathon_BC `main`) on the files written
by e25_team_export.py and score it on E25's test copies with their own decoder and the official
scorer. Run with their environment, pointing at a checkout of their repository that has the
official repository at <checkout>/official:

    <their python> experiments/e25_team_side.py <checkout> <folder> e17_old_train e18_v2 e24_v2_real

Their training regimes, features, classifier and decoder are used unchanged; only the split
(fit on one dataset, score on the other) and the test damage are ours. External open data only.
"""

import json
import os
import sys

import numpy as np

checkout, folder, names = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, checkout)
sys.path.insert(0, os.path.join(checkout, "scripts"))

from mtah import experiment, model  # noqa: E402
from mtah.schema import Match  # noqa: E402
from run_experiment import CONFIGS, feature_fn  # noqa: E402


def load(path):
    z = np.load(path)
    pitch = tuple(float(v) for v in z["pitch"])
    players = z["players"] * np.asarray(pitch, dtype=np.float32)
    y = z["y"]
    edges = np.flatnonzero(np.diff(np.concatenate(([0], (y != 3).astype(np.int8), [0]))))
    blocks = tuple((int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2]))
    # A slot can hold a home player in one half and an away player in the other (DFL), so each
    # team's array keeps every slot and is NaN where the slot is not that team's.
    side = lambda t: np.where((z["teams"] == t)[:, :, None], players, np.nan)  # noqa: E731
    return Match(os.path.basename(path)[:-4], side(0), side(1),
                 (z["ball"] * np.asarray(pitch, dtype=np.float32)).astype(np.float32), blocks,
                 pitch, y)


def files(role, dataset):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.startswith("{}__{}__".format(role, dataset)))


def main():
    train = {d: [load(p) for p in files("train", d)] for d in ("metrica", "dfl", "skillcorner")}
    results = os.path.join(folder, "team_results.json")
    out = {}
    if os.path.isfile(results):  # configurations finished in an earlier run
        with open(results) as handle:
            out = json.load(handle)
    for name in names:
        if name in out:
            continue
        cfg, featurise = CONFIGS[name], feature_fn(CONFIGS[name])
        real = train["skillcorner"][:cfg.real_games]
        runs = (("dfl", train["dfl"], files("test", "metrica")),
                ("metrica", train["metrica"], files("test", "dfl")),
                ("metrica+dfl", train["metrica"] + train["dfl"], files("test", "skillcorner")))
        rows = {}
        for src, matches, tests in runs:
            models, swap = experiment.fit_fold(matches, cfg, featurise, real)
            for path in tests:
                game = load(path)
                X, _ = experiment.featurise_cached(game, None, 0, cfg, featurise)
                proba = np.mean([model.predict_proba(m, X, swap) for m in models], axis=0)
                s = experiment.decode_and_score(proba, game, matches + list(real), cfg)
                rows[game.game_id] = {k: s[k] for k in ("spell_f1", "macro_f1", "d_f1",
                                                        "n_gt_spells", "n_pred_spells")}
            print("{} {}: fitted and scored {} test copies".format(name, src, len(tests)),
                  flush=True)
        summary = {}
        for tag in ("std35", "std50", "art10", "dark", "sc_real"):
            for dataset in ("metrica", "dfl", "skillcorner"):
                v = [r["spell_f1"] for k, r in rows.items()
                     if k.startswith("test__" + dataset) and k.endswith("__" + tag)]
                if v:
                    summary["{}|{}".format(dataset, tag)] = float(np.mean(v))
        out[name] = dict(summary=summary, rows=rows)
        print("== {}: {}".format(name, "  ".join("{} {:.4f}".format(k, v)
                                                 for k, v in summary.items())), flush=True)
        with open(results, "w") as handle:
            json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
