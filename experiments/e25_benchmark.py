"""E25: one benchmark for every variant of the gradient-boosting pipeline (v2 + fill + HSMM + rule).

A variant is a set of training gap regimes, a set of feature families and whether SkillCorner
games with their real gaps join the training set:

  std        the E13 candidate: three gap rates, base + fill features
  mix        + two more training regimes: false detections (E13) and dark stoppages (E11)
  +ctx       + mtah.features.compute_context (after the team's E18)
  +turn      + mtah.features.compute_turnover (E26)
  +sc        + 12 SkillCorner games, as loaded, in every training set (after the team's E24)

Test sets, the same for every variant (labels: controlled rule 2 s):
  std35, std50   fit on one dataset, score on the other; E10's cells (seeds 7, 8). Their mean
                 over both directions is the selection number.
  art10          the same with 10 false-detection bursts a minute and the centre blind spot
                 (E13's hardest cell, seed 10)
  dark           7% of live frames missing, ball seen on a fifth of each stoppage (seed 11)
  sc_real        fit on Metrica + DFL, score on 8 SkillCorner games held out of every training set

Same fitting as mtah.v2.fit + add_hsmm (the Viterbi grid search, unused by the HSMM, is skipped,
and the HSMM decodes run in parallel): `std` reproduces E10 to the last digit.

    python experiments/e25_benchmark.py std mix mix+ctx ...
"""

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import REPO, dfl_games, external_games, skillcorner_games, table
from controlled import controlled
from gapsim import TRAIN_EXTRA, Params, damage
from mtah import ballfill, features, hsmm, v2
from mtah.evidence import drop_far_outside, nearest_team
from mtah.labels import graded_blocks
from mtah.scoring import score
from mtah.smooth import majority_vote

MIN_S = 2.0
STD = (("r15", dict(live_missing=0.15)), ("r35", dict(live_missing=0.35)),
       ("r50", dict(live_missing=0.5)))
# The two extra training regimes of gapsim.TRAIN_EXTRA: artefacts (E13) and dark stoppages (E11).
MIX = STD + (("art", TRAIN_EXTRA[0]), ("dark", TRAIN_EXTRA[1]))
TESTS = (("std35", dict(live_missing=0.35), 7), ("std50", dict(live_missing=0.5), 8),
         ("art10", dict(live_missing=0.35, false_per_min=10.0, centre_blind_m=3.0), 10),
         ("dark", dict(live_missing=0.07, dead_visible=0.2), 11))
SELECTION = ("std35", "std50")
SC_HELD_OUT = 8
OUT = os.path.join(REPO, "out", "e25")


def parse(variant):
    """'mix+ctx+turn+sc' -> (training regime tags, feature families, SkillCorner in training)."""
    parts = variant.split("+")
    regimes = {"std": STD, "mix": MIX}[parts[0]]
    families = ("base",) + tuple(p for p in parts[1:] if p in ("ctx", "turn"))
    return tuple(t for t, _ in regimes), families, "sc" in parts[1:]


def featurise(name, ball, players, teams, pitch, gt, tag):
    """Every feature family of one (damaged) game, the far-outside rule applied first."""
    ball = drop_far_outside(ball, pitch)
    blocks = graded_blocks(gt)
    near = nearest_team(ball, players, teams, pitch)
    filled, gap = ballfill.fill(ball, players, pitch, blocks)
    near_f = nearest_team(filled, players, teams, pitch)
    parts = [("base", features.compute(ball, players, teams, pitch, near)),
             ("base", features.compute_filled(filled, gap, players, teams, pitch, near_f)),
             ("ctx", features.compute_context(ball, players, teams, pitch, near_f)),
             ("turn", features.compute_turnover(ball, players, teams, pitch, near, near_f))]
    return dict(game=name, tag=tag, X=np.hstack([x for _, (x, _) in parts]),
                names=[k for _, (_, nm) in parts for k in nm],
                family=[f for f, (_, nm) in parts for _ in nm],
                y=controlled(gt, MIN_S), baseline=majority_vote(near, 50), gt=gt)


def damaged(games, profiles):
    """profiles: [(tag, Params fields, seed)]."""
    out = []
    for g in games:
        for tag, fields, seed in profiles:
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(**fields), seed=seed)
            out.append(featurise(g["name"], ball, players, g["teams"], g["pitch"], g["gt"], tag))
    return out


def select(d, families):
    keep = [i for i, f in enumerate(d["family"]) if f in families]
    if len(keep) == len(d["family"]):
        return d["X"], d["names"]
    return d["X"][:, keep], [d["names"][i] for i in keep]


def _decode_score(args):
    y, proba, prior, log_dur, log_trans, temperature = args
    pred = v2.hsmm_decode(proba, graded_blocks(y), prior, log_dur, log_trans, temperature, len(y))
    return pred, score(y, pred)


def fit(train, families, pool):
    """mtah.v2.fit + add_hsmm on the selected feature families."""
    items = [(d["game"],) + select(d, families) + (d["y"],) for d in train]
    ids = sorted({gid for gid, *_ in items})
    folds = [ids[i::3] for i in range(3)]
    oof, oof_keys = [], []
    for fold in folds:
        clf = v2.train_classifier([(X, y, nm) for gid, X, nm, y in items if gid not in fold])
        oof += [(y, v2.symmetric_proba(clf, X, nm)) for gid, X, nm, y in items if gid in fold]
        oof_keys += ["{}|{}".format(d["game"], d["tag"]) for d in train if d["game"] in fold]
    labels = [y for *_, y in items]
    prior = v2.class_prior(labels)
    log_dur, log_trans = hsmm.count_durations(labels)
    tasks = [(y, p, prior, log_dur, log_trans, T) for T in v2.HSMM_TEMPS for y, p in oof]
    f1 = np.array([s["spell_f1_mean"] for _, s in pool.map(_decode_score, tasks)])
    f1 = f1.reshape(len(v2.HSMM_TEMPS), len(oof)).mean(axis=1)
    best = int(np.argmax(f1))  # first maximum, as add_hsmm's strict comparison
    clf = v2.train_classifier([(X, y, nm) for _, X, nm, y in items])
    return dict(clf=clf, prior=prior, log_dur=log_dur, log_trans=log_trans,
                temperature=v2.HSMM_TEMPS[best], oof_f1=float(f1[best]),
                oof=dict(zip(oof_keys, oof)))


def evaluate(model, tests, families, pool):
    """[(test copy, probabilities, prediction, scores, baseline scores)]."""
    probas = [v2.symmetric_proba(model["clf"], *select(d, families)) for d in tests]
    tasks = [(d["y"], p, model["prior"], model["log_dur"], model["log_trans"],
              model["temperature"]) for d, p in zip(tests, probas)]
    done = list(pool.map(_decode_score, tasks))
    return [(d, p, pred, s, score(d["y"], d["baseline"]))
            for d, p, (pred, s) in zip(tests, probas, done)]


RUNS = (("dfl", "metrica"), ("metrica", "dfl"), ("metrica+dfl", "skillcorner"))


def dump_path(variant, src, dst):
    return os.path.join(OUT, "{}_{}_to_{}.npz".format(variant, src, dst))


def metrics(s):
    return dict(spell_f1=s["spell_f1_mean"], macro_f1=s["macro_f1"], n_pred=s["n_pred_spells"],
                n_gt=s["n_gt_spells"], d_f1=s["d_f1"], team_acc=s["team_acc_both_alive"])


def scores_of_dump(path):
    """{game|tag: metrics} of a finished run, from its saved predictions."""
    z = np.load(path)
    out = {}
    for key in [k[2:] for k in z.files if k.startswith("y|")]:
        out[key] = metrics(score(z["y|" + key], z["pred|" + key]))
        out[key]["baseline"] = score(z["y|" + key], z["baseline|" + key])["spell_f1_mean"] \
            if "baseline|" + key in z.files else float("nan")
    return out


def main(variants):
    os.makedirs(OUT, exist_ok=True)
    todo = [(v, src, dst) for v in variants for src, dst in RUNS
            if not os.path.isfile(dump_path(v, src, dst))]
    train, test, sc_test, sc_train, pool = {}, {}, [], [], None
    if todo:  # only the features the unfinished runs need
        pool = ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn"))
        data = {"metrica": external_games(), "dfl": dfl_games()}
        t0 = time.time()
        train = {k: damaged(v, [(t, f, 100 + i) for i, (t, f) in enumerate(MIX)])
                 for k, v in data.items()}
        if any(dst != "skillcorner" for _, _, dst in todo):
            test = {k: damaged(v, TESTS) for k, v in data.items()}
        del data
        if any(dst == "skillcorner" or parse(v)[2] for v, _, dst in todo):
            sc = [featurise(g["name"], g["ball"], g["players"], g["teams"], g["pitch"], g["gt"],
                            "sc_real") for g in skillcorner_games()]
            sc_test, sc_train = sc[:SC_HELD_OUT], sc[SC_HELD_OUT:]
        print("features ready in {:.0f} s".format(time.time() - t0), flush=True)

    rows = []
    for variant in variants:
        regimes, families, with_sc = parse(variant)
        extra = sc_train if with_sc else []
        pick = lambda copies: [d for d in copies if d["tag"] in regimes]  # noqa: E731
        per_game = {}
        for src, dst in RUNS:
            path = dump_path(variant, src, dst)
            if not os.path.isfile(path):  # else finished in an earlier run
                t0 = time.time()
                tr = sum((pick(train[k]) for k in src.split("+")), []) + extra
                te = sc_test if dst == "skillcorner" else test[dst]
                model = fit(tr, families, pool)
                dump = {}
                for d, p, pred, _, _ in evaluate(model, te, families, pool):
                    key = "{}|{}".format(d["game"], d["tag"])
                    dump["y|" + key], dump["pred|" + key] = d["y"], pred
                    dump["proba|" + key] = p.astype(np.float32)
                    dump["baseline|" + key] = d["baseline"]
                for key, (y, p) in model["oof"].items():  # out-of-fold, for stacking (E27)
                    dump["oof_y|" + key], dump["oof_proba|" + key] = y, p.astype(np.float32)
                for k in ("prior", "log_dur", "log_trans", "temperature"):
                    dump[k] = model[k]
                np.savez_compressed(path, **dump)
                print("{} {}->{}: {} columns, hsmm temp {}, oof {:.4f}, {:.0f} s".format(
                    variant, src, dst, len(select(te[0], families)[1]), model["temperature"],
                    model["oof_f1"], time.time() - t0), flush=True)
                del model, dump
            scores = scores_of_dump(path)
            per_game.update(scores)
            for tag in dict.fromkeys(k.split("|")[1] for k in scores):
                v = [m for k, m in scores.items() if k.split("|")[1] == tag]
                rows.append(dict(variant=variant, train=src, test=dst, gaps=tag,
                                 **{k: float(np.mean([m[k] for m in v])) for k in v[0]}))
        mine = [r for r in rows if r["variant"] == variant]
        summary = {tag: float(np.mean([r["spell_f1"] for r in mine if r["gaps"] == tag]))
                   for tag in ("std35", "std50", "art10", "dark", "sc_real")}
        summary["selection"] = float(np.mean([r["spell_f1"] for r in mine
                                              if r["gaps"] in SELECTION]))
        with open(os.path.join(OUT, variant + ".json"), "w") as handle:
            json.dump(dict(summary=summary, rows=mine, per_game=per_game), handle, indent=1)
        print("== {}: selection {:.4f} | art10 {:.4f} | dark {:.4f} | sc_real {:.4f}".format(
            variant, summary["selection"], summary["art10"], summary["dark"],
            summary["sc_real"]), flush=True)
    if pool is not None:
        pool.shutdown()
    print()
    print(table(rows, list(rows[0].keys())))


if __name__ == "__main__":
    main(sys.argv[1:])
