"""E07: does filling the missing ball help v2?

Variants, all with the same classifier, Viterbi and selection (v2):
  v2          features from the damaged tracking only
  v2+interp   plus features from the ball filled by straight-line interpolation
  v2+learned  plus features from the ball filled by interpolation + learned correction
The fill model is trained on the training dataset only. Fit on one dataset, score once on the
other, controlled-possession rule 2 s.

    python experiments/e07_ballfill.py
"""

import numpy as np

from common import dfl_games, external_games, table
from controlled import controlled
from gapsim import Params, damage
from mtah import ballfill, features, v1, v2
from mtah.evidence import nearest_team
from mtah.labels import graded_blocks
from mtah.scoring import score

MIN_S = 2.0
TEST_RATES = (0.35, 0.5)


def damaged(games, rates, seed0):
    out = []
    for g in games:
        blocks = graded_blocks(g["gt"])
        live = np.isin(g["gt"], (0, 1))
        for i, r in enumerate(rates):
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(live_missing=r), seed=seed0 + i)
            near = nearest_team(ball, players, g["teams"], g["pitch"])
            Xg, names = features.compute(ball, players, g["teams"], g["pitch"], near)
            out.append(dict(game=g["name"], rate=r, gt=g["gt"], ball=ball, players=players,
                            teams=g["teams"], pitch=g["pitch"], blocks=blocks, live=live,
                            clean=g["ball"], X=Xg, names=names))
    return out


def with_fill(d, models):
    filled, gap = ballfill.fill(d["ball"], d["players"], d["pitch"], d["blocks"], models)
    near_f = nearest_team(filled, d["players"], d["teams"], d["pitch"])
    Xf, nf = features.compute_filled(filled, gap, d["players"], d["teams"], d["pitch"], near_f)
    return np.hstack([d["X"], Xf]), d["names"] + nf, filled, gap


def fill_error(d, filled, gap):
    m = gap["missing"] & d["live"] & ~np.isnan(d["clean"]).any(axis=1) & \
        ~np.isnan(filled).any(axis=1)
    err = np.linalg.norm((filled[m] - d["clean"][m]) * np.asarray(d["pitch"]), axis=1)
    return float(np.median(err))


def main():
    data = {"metrica": external_games(), "dfl": dfl_games()}
    train = {k: damaged(v, v1.GAP_RATES, 100) for k, v in data.items()}
    test = {k: damaged(v, TEST_RATES, 7) for k, v in data.items()}
    print("features ready", flush=True)
    rows = []
    for src, dst in (("dfl", "metrica"), ("metrica", "dfl")):
        fill_models = ballfill.train([(d["ball"], d["players"], d["pitch"], d["blocks"],
                                       d["clean"], d["live"]) for d in train[src]])
        variants = {"v2": None, "v2+interp": "interp", "v2+learned": "learned"}
        for name, kind in variants.items():
            def feats(d):
                if kind is None:
                    return d["X"], d["names"], None
                Xf, nf, filled, gap = with_fill(d, fill_models if kind == "learned" else None)
                return Xf, nf, fill_error(d, filled, gap)
            items = []
            for d in train[src]:
                Xg, nm, _ = feats(d)
                items.append((d["game"], Xg, controlled(d["gt"], MIN_S), nm))
            model = v2.fit(items)
            for rate in TEST_RATES:
                f, errs = [], []
                for d in test[dst]:
                    if d["rate"] != rate:
                        continue
                    gt = controlled(d["gt"], MIN_S)
                    Xg, nm, e = feats(d)
                    s = score(gt, v2.predict(model, Xg, nm, graded_blocks(gt), len(gt)))
                    f.append(s["spell_f1_mean"])
                    if e is not None:
                        errs.append(e)
                rows.append(dict(train=src, test=dst, variant=name, live_missing=rate,
                                 spellF1=float(np.mean(f)),
                                 fill_err_median_m=float(np.mean(errs)) if errs else float("nan")))
            print("{}->{} {} done".format(src, dst, name), flush=True)
    print()
    print(table(rows, list(rows[0].keys())))


if __name__ == "__main__":
    main()
