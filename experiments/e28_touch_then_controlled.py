"""E28: learn who touches the ball, decide what counts as possession after decoding.

Hypothesis (Álvaro, 2026-11-11). Today the 2 s controlled rule is applied to the external
labels BEFORE training (e25_benchmark.featurise: y = controlled(gt, MIN_S)), so the classifier
has to learn the ball contact and our possession convention at the same time, and a short
opposing touch relabels its frames as the other team's possession. The alternative tested here:
train the classifier and the HSMM on the touch-level labels, decode a touch-level sequence and
apply external/controlled.py to the PREDICTION. The HSMM's duration distributions and spell
transitions are counted on the touch labels, not reused from the controlled-label ones.

If the split works, the controlled threshold stops being a training constant: the
0/0.5/1/2/3 s sweep runs on the decoded sequence, so it costs one training run instead of one
per threshold. The sweep measures how much our result depends on that convention; it is not a
search for the challenge's private definition, and nothing here touches the competition match.

One bias to read the sweep with: every threshold is scored against answer keys built with the
SAME 2 s rule, so the table is stacked in favour of min_s = 2 s - of course the prediction
that was post-processed with 2 s matches labels built with 2 s. The honest reading is
sensitivity (how much each threshold costs against today's convention), not discovery (which
threshold is really right).

Open hypothesis, deliberately NOT implemented here (Álvaro, 2026-11-11): a fixed duration
cannot tell a deflection from a controlled touch - a 0.5 s contact can be a rebound or a clean
take followed by a pass. What separates them is what happens NEXT: a controlled touch ends in
an intentional exit (a pass to a teammate, ball played forward), a deflection ends loose or
with the other team. Nothing in controlled.py, the HSMM durations or the fixed rules uses that
signal today. A future variant could score each short spell with its aftermath (next-touch
team, time to next touch, change in ball velocity at the exit) instead of its length alone.
External data cannot validate that distinction - our only controlled labels are built by the
duration rule itself - so it stays a hypothesis until the team decides whether it is worth
designing against the challenge text.

Protocol, fixed before running (same goalposts as every entry since E10):
  - training regimes, feature families, folds, damage seeds and test sets: e25_benchmark's
    (candidate 1 is `mix+ctx+turn`; run that one first).
  - evaluation labels: controlled(gt, 2.0) everywhere, exactly as E25-E27. The sweep moves the
    rule applied to the prediction only; the answer keys it is scored against never move.
  - reported per cell and threshold: spellF1, macroF1, predicted/gt spell counts; per game in
    the json. Selection number = mean spellF1 of the std35/std50 cells at min_s = 2 s, to be
    compared with candidate 1's 0.6742.

Proposed decision rule (a proposal for the team, not a validated result): the split is worth
keeping only if, under this same protocol, it beats the E25 `mix+ctx+turn` row fitted and
scored the same way on the selection cells, without a clear loss on the sc_real cell (real
gaps). A win that shows up only on simulated gaps, or with fragmented possessions (n_pred well
above n_gt), argues for dropping it. Candidate 1's 0.6742 orients that comparison; it is not a
fixed threshold.

Status: written 2026-11-11, NOT RUN - this file is the run plan, there are no results yet.

    python experiments/e28_touch_then_controlled.py mix+ctx+turn
"""

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import e25_benchmark as e25
from common import REPO, dfl_games, external_games, skillcorner_games, table
from controlled import controlled
from gapsim import Params, damage
from mtah import hsmm, v2
from mtah.labels import graded_blocks
from mtah.scoring import score

MIN_S_EVAL = 2.0  # evaluation labels, fixed: controlled(gt, MIN_S_EVAL), the E25 goalposts
SWEEP_MIN_S = (0.0, 0.5, 1.0, 2.0, 3.0)  # 0.0 = the raw touch sequence, as reference
OUT = os.path.join(REPO, "out", "e28")


def featurise(name, ball, players, teams, pitch, gt, tag):
    """e25_benchmark.featurise with touch-level training labels; the evaluation labels stay
    controlled(gt, MIN_S_EVAL), bit-identical to what E25 trains on."""
    d = e25.featurise(name, ball, players, teams, pitch, gt, tag)
    d["y_eval"] = d["y"]  # controlled(gt, 2.0): the fixed goalposts, scored against
    d["y"] = gt           # touch-level: what the classifier and the HSMM learn here
    return d


def damaged(games, profiles):
    """e25_benchmark.damaged with the touch-label featurise above."""
    out = []
    for g in games:
        for tag, fields, seed in profiles:
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(**fields), seed=seed)
            out.append(featurise(g["name"], ball, players, g["teams"], g["pitch"], g["gt"], tag))
    return out


def _decode_score(args):
    """Decode one copy at touch level, apply the controlled rule to the prediction, score
    against the fixed evaluation labels."""
    y_eval, proba, prior, log_dur, log_trans, temperature, min_s = args
    pred_touch = v2.hsmm_decode(proba, graded_blocks(y_eval), prior, log_dur, log_trans,
                                temperature, len(y_eval))
    pred = controlled(pred_touch, min_s)
    return pred_touch, pred, score(y_eval, pred)


def folds_of(ids, n_folds=3):
    """e25_benchmark's fold split: the game id is the group, so every damaged copy of a game
    lands in one fold and no game leaks into another fold's training set."""
    ids = sorted(ids)
    return [ids[i::n_folds] for i in range(n_folds)]


def fit(train, families, pool):
    """e25_benchmark.fit on touch labels: same folds and classifier, but the HSMM durations
    and transitions are counted on the touch labels, and the temperature is chosen on
    out-of-fold predictions AFTER the controlled rule is applied, against the fixed
    evaluation labels."""
    items = [(d["game"],) + e25.select(d, families) + (d["y"], d["y_eval"]) for d in train]
    ids = sorted({gid for gid, *_ in items})
    folds = folds_of(ids)
    oof, oof_keys = [], []
    for fold in folds:
        clf = v2.train_classifier([(X, y, nm) for gid, X, nm, y, _ in items if gid not in fold])
        oof += [(yv, v2.symmetric_proba(clf, X, nm))
                for gid, X, nm, y, yv in items if gid in fold]
        oof_keys += ["{}|{}".format(d["game"], d["tag"]) for d in train if d["game"] in fold]
    labels = [y for *_, y, _ in items]  # touch-level, as decoded
    prior = v2.class_prior(labels)
    log_dur, log_trans = hsmm.count_durations(labels)
    tasks = [(yv, p, prior, log_dur, log_trans, T, MIN_S_EVAL)
             for T in v2.HSMM_TEMPS for yv, p in oof]
    f1 = np.array([s["spell_f1_mean"] for _, _, s in pool.map(_decode_score, tasks)])
    f1 = f1.reshape(len(v2.HSMM_TEMPS), len(oof)).mean(axis=1)
    best = int(np.argmax(f1))  # first maximum, as e25_benchmark.fit
    clf = v2.train_classifier([(X, y, nm) for _, X, nm, y, _ in items])
    return dict(clf=clf, prior=prior, log_dur=log_dur, log_trans=log_trans,
                temperature=v2.HSMM_TEMPS[best], oof_f1=float(f1[best]),
                oof=dict(zip(oof_keys, oof)))  # e25's key layout, for stacking (E27)


def evaluate(model, tests, families, pool):
    """Decode each test copy once at touch level, then sweep the controlled threshold on the
    decoded sequence: [(test copy, probabilities, touch prediction, {min_s: (pred, scores)},
    baseline scores)]."""
    probas = [v2.symmetric_proba(model["clf"], *e25.select(d, families)) for d in tests]
    tasks = [(d["y_eval"], p, model["prior"], model["log_dur"], model["log_trans"],
              model["temperature"], MIN_S_EVAL) for d, p in zip(tests, probas)]
    done = list(pool.map(_decode_score, tasks))
    rows = []
    for d, p, (pred_touch, pred, s) in zip(tests, probas, done):
        sweep = {}
        for ms in SWEEP_MIN_S:
            pred_c = pred if ms == MIN_S_EVAL else controlled(pred_touch, ms)
            sweep[ms] = (pred_c, s if ms == MIN_S_EVAL else score(d["y_eval"], pred_c))
        rows.append((d, p, pred_touch, sweep, score(d["y_eval"], d["baseline"])))
    return rows


RUNS = e25.RUNS


def dump_path(variant, src, dst):
    return os.path.join(OUT, "{}_{}_to_{}.npz".format(variant, src, dst))


def scores_of_dump(path):
    """{game|tag: {min_s: metrics}} of a finished run, from its saved touch predictions."""
    z = np.load(path)
    out = {}
    for key in [k[2:] for k in z.files if k.startswith("y|")]:
        y_eval, pred_touch = z["y|" + key], z["pred|" + key]
        out[key] = {ms: e25.metrics(score(y_eval, controlled(pred_touch, ms)))
                    for ms in SWEEP_MIN_S}
        out[key]["baseline"] = score(y_eval, z["baseline|" + key])["spell_f1_mean"] \
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
        train = {k: damaged(v, [(t, f, 100 + i) for i, (t, f) in enumerate(e25.MIX)])
                 for k, v in data.items()}
        if any(dst != "skillcorner" for _, _, dst in todo):
            test = {k: damaged(v, e25.TESTS) for k, v in data.items()}
        del data
        if any(dst == "skillcorner" or e25.parse(v)[2] for v, _, dst in todo):
            sc = [featurise(g["name"], g["ball"], g["players"], g["teams"], g["pitch"], g["gt"],
                            "sc_real") for g in skillcorner_games()]
            sc_test, sc_train = sc[:e25.SC_HELD_OUT], sc[e25.SC_HELD_OUT:]
        print("features ready in {:.0f} s".format(time.time() - t0), flush=True)

    rows = []
    for variant in variants:
        regimes, families, with_sc = e25.parse(variant)
        extra = sc_train if with_sc else []
        pick = lambda copies: [d for d in copies if d["tag"] in regimes]  # noqa: E731
        per_game = {}
        for src, dst in RUNS:
            path = dump_path(variant, src, dst)
            if not os.path.isfile(path):  # else finished in an earlier run
                t0 = time.time()
                tr = sum((pick(train[k]) for k in src.split("+")), []) + extra

Hypothesis (Álvaro, 2026-11-11). Today the 2 s controlled rule is applied to the external
labels BEFORE training (e25_benchmark.featurise: y = controlled(gt, MIN_S)), so the classifier
has to learn the ball contact and our possession convention at the same time, and a short
opposing touch relabels its frames as the other team's possession. The alternative tested here:
train the classifier and the HSMM on the touch-level labels, decode a touch-level sequence and
apply external/controlled.py to the PREDICTION. The HSMM's duration distributions and spell
transitions are counted on the touch labels, not reused from the controlled-label ones.

If the split works, the controlled threshold stops being a training constant: the
0/0.5/1/2/3 s sweep runs on the decoded sequence, so it costs one training run instead of one
per threshold. The sweep measures how much our result depends on that convention; it is not a
search for the challenge's private definition, and nothing here touches the competition match.

One bias to read the sweep with: every threshold is scored against answer keys built with the
SAME 2 s rule, so the table is stacked in favour of min_s = 2 s - of course the prediction
that was post-processed with 2 s matches labels built with 2 s. The honest reading is
sensitivity (how much each threshold costs against today's convention), not discovery (which
threshold is really right).

Open hypothesis, deliberately NOT implemented here (Álvaro, 2026-11-11): a fixed duration
cannot tell a deflection from a controlled touch - a 0.5 s contact can be a rebound or a clean
take followed by a pass. What separates them may be what happens NEXT: a controlled touch often
ends in a deliberate exit (a pass to a teammate), a deflection ends loose or with the other
team. The tracking features already compute next-sighting and velocity-break clues (E25/E26),
and ball speed alone does not prove intent - the new part would be a DECISION rule that
classifies each short spell as control or contact from its aftermath (next-touch team, time to
next touch, change in ball velocity at the exit), instead of its length alone. Nothing in
controlled.py, the HSMM durations or the fixed rules applies such a rule today. It cannot be
validated with our current controlled labels, which are built by the duration rule itself; it
would need an independent reference (hand annotations, an external review) before the team
trusts it. Until then it is a hypothesis, not a result.

Protocol, fixed before running (same goalposts as every entry since E10):
  - training regimes, feature families, folds, damage seeds and test sets: e25_benchmark's
    (candidate 1 is `mix+ctx+turn`; run that one first).
  - evaluation labels: controlled(gt, 2.0) everywhere, exactly as E25-E27. The sweep moves the
    rule applied to the prediction only; the answer keys it is scored against never move.
  - reported per cell and threshold: spellF1, macroF1, predicted/gt spell counts; per game in
    the json. Selection number = mean spellF1 of the std35/std50 cells at min_s = 2 s, to be
    compared with candidate 1's 0.6742.

Proposed decision rule (a proposal for the team, not a validated result): the split is worth
keeping only if, under this same protocol, it beats the E25 `mix+ctx+turn` row fitted and
scored the same way on the selection cells, without a clear loss on the sc_real cell (real
gaps). A win that shows up only on simulated gaps, or with fragmented possessions (n_pred well
above n_gt), argues for dropping it. Candidate 1's 0.6742 orients that comparison; it is not a
fixed threshold.

Status: written 2026-11-11, NOT RUN - this file is the run plan, there are no results yet.

    python experiments/e28_touch_then_controlled.py mix+ctx+turn
"""

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import e25_benchmark as e25
from common import REPO, dfl_games, external_games, skillcorner_games, table
from controlled import controlled
from gapsim import Params, damage
from mtah import hsmm, v2
from mtah.labels import graded_blocks
from mtah.scoring import score

MIN_S_EVAL = 2.0  # evaluation labels, fixed: controlled(gt, MIN_S_EVAL), the E25 goalposts
SWEEP_MIN_S = (0.0, 0.5, 1.0, 2.0, 3.0)  # 0.0 = the raw touch sequence, as reference
OUT = os.path.join(REPO, "out", "e28")


def featurise(name, ball, players, teams, pitch, gt, tag):
    """e25_benchmark.featurise with touch-level training labels; the evaluation labels stay
    controlled(gt, MIN_S_EVAL), bit-identical to what E25 trains on."""
    d = e25.featurise(name, ball, players, teams, pitch, gt, tag)
    d["y_eval"] = d["y"]  # controlled(gt, 2.0): the fixed goalposts, scored against
    d["y"] = gt           # touch-level: what the classifier and the HSMM learn here
    return d


def damaged(games, profiles):
    """e25_benchmark.damaged with the touch-label featurise above."""
    out = []
    for g in games:
        for tag, fields, seed in profiles:
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(**fields), seed=seed)
            out.append(featurise(g["name"], ball, players, g["teams"], g["pitch"], g["gt"], tag))
    return out


def _decode_score(args):
    """Decode one copy at touch level, apply the controlled rule to the prediction, score
    against the fixed evaluation labels."""
    y_eval, proba, prior, log_dur, log_trans, temperature, min_s = args
    pred_touch = v2.hsmm_decode(proba, graded_blocks(y_eval), prior, log_dur, log_trans,
                                temperature, len(y_eval))
    pred = controlled(pred_touch, min_s)
    return pred_touch, pred, score(y_eval, pred)


def folds_of(ids, n_folds=3):
    """e25_benchmark's fold split: the game id is the group, so every damaged copy of a game
    lands in one fold and no game leaks into another fold's training set."""
    ids = sorted(ids)
    return [ids[i::n_folds] for i in range(n_folds)]


def fit(train, families, pool):
    """e25_benchmark.fit on touch labels: same folds and classifier, but the HSMM durations
    and transitions are counted on the touch labels, and the temperature is chosen on
    out-of-fold predictions AFTER the controlled rule is applied, against the fixed
    evaluation labels."""
    items = [(d["game"],) + e25.select(d, families) + (d["y"], d["y_eval"]) for d in train]
    ids = sorted({gid for gid, *_ in items})
    folds = folds_of(ids)
    oof, oof_keys = [], []
    for fold in folds:
        clf = v2.train_classifier([(X, y, nm) for gid, X, nm, y, _ in items if gid not in fold])
        oof += [(yv, v2.symmetric_proba(clf, X, nm))
                for gid, X, nm, y, yv in items if gid in fold]
        oof_keys += ["{}|{}".format(d["game"], d["tag"]) for d in train if d["game"] in fold]
    labels = [y for *_, y, _ in items]  # touch-level, as decoded
    prior = v2.class_prior(labels)
    log_dur, log_trans = hsmm.count_durations(labels)
    tasks = [(yv, p, prior, log_dur, log_trans, T, MIN_S_EVAL)
             for T in v2.HSMM_TEMPS for yv, p in oof]
    f1 = np.array([s["spell_f1_mean"] for _, _, s in pool.map(_decode_score, tasks)])
    f1 = f1.reshape(len(v2.HSMM_TEMPS), len(oof)).mean(axis=1)
    best = int(np.argmax(f1))  # first maximum, as e25_benchmark.fit
    clf = v2.train_classifier([(X, y, nm) for _, X, nm, y, _ in items])
    return dict(clf=clf, prior=prior, log_dur=log_dur, log_trans=log_trans,
                temperature=v2.HSMM_TEMPS[best], oof_f1=float(f1[best]),
                oof=dict(zip(oof_keys, oof)))  # e25's key layout, for stacking (E27)


def evaluate(model, tests, families, pool):
    """Decode each test copy once at touch level, then sweep the controlled threshold on the
    decoded sequence: [(test copy, probabilities, touch prediction, {min_s: (pred, scores)},
    baseline scores)]."""
    probas = [v2.symmetric_proba(model["clf"], *e25.select(d, families)) for d in tests]
    tasks = [(d["y_eval"], p, model["prior"], model["log_dur"], model["log_trans"],
              model["temperature"], MIN_S_EVAL) for d, p in zip(tests, probas)]
    done = list(pool.map(_decode_score, tasks))
    rows = []
    for d, p, (pred_touch, pred, s) in zip(tests, probas, done):
        sweep = {}
        for ms in SWEEP_MIN_S:
            pred_c = pred if ms == MIN_S_EVAL else controlled(pred_touch, ms)
            sweep[ms] = (pred_c, s if ms == MIN_S_EVAL else score(d["y_eval"], pred_c))
        rows.append((d, p, pred_touch, sweep, score(d["y_eval"], d["baseline"])))
    return rows


RUNS = e25.RUNS


def dump_path(variant, src, dst):
    return os.path.join(OUT, "{}_{}_to_{}.npz".format(variant, src, dst))


def scores_of_dump(path):
    """{game|tag: {min_s: metrics}} of a finished run, from its saved touch predictions."""
    z = np.load(path)
    out = {}
    for key in [k[2:] for k in z.files if k.startswith("y|")]:
        y_eval, pred_touch = z["y|" + key], z["pred|" + key]
        out[key] = {ms: e25.metrics(score(y_eval, controlled(pred_touch, ms)))
                    for ms in SWEEP_MIN_S}
        out[key]["baseline"] = score(y_eval, z["baseline|" + key])["spell_f1_mean"] \
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
        train = {k: damaged(v, [(t, f, 100 + i) for i, (t, f) in enumerate(e25.MIX)])
                 for k, v in data.items()}
        if any(dst != "skillcorner" for _, _, dst in todo):
            test = {k: damaged(v, e25.TESTS) for k, v in data.items()}
        del data
        if any(dst == "skillcorner" or e25.parse(v)[2] for v, _, dst in todo):
            sc = [featurise(g["name"], g["ball"], g["players"], g["teams"], g["pitch"], g["gt"],
                            "sc_real") for g in skillcorner_games()]
            sc_test, sc_train = sc[:e25.SC_HELD_OUT], sc[e25.SC_HELD_OUT:]
        print("features ready in {:.0f} s".format(time.time() - t0), flush=True)

    rows = []
    for variant in variants:
        regimes, families, with_sc = e25.parse(variant)
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
                for d, p, pred_touch, _, _ in evaluate(model, te, families, pool):
                    key = "{}|{}".format(d["game"], d["tag"])
                    dump["y|" + key], dump["pred|" + key] = d["y_eval"], pred_touch
                    dump["proba|" + key] = p.astype(np.float32)
                    dump["baseline|" + key] = d["baseline"]
                for key, (yv, p) in model["oof"].items():  # out-of-fold, for stacking (E27)
                    dump["oof_y|" + key], dump["oof_proba|" + key] = yv, p.astype(np.float32)
                for k in ("prior", "log_dur", "log_trans", "temperature"):
                    dump[k] = model[k]
                np.savez_compressed(path, **dump)
                print("{} {}->{}: {} columns, hsmm temp {}, oof {:.4f}, {:.0f} s".format(
                    variant, src, dst, len(e25.select(te[0], families)[1]),
                    model["temperature"], model["oof_f1"], time.time() - t0), flush=True)
                del model, dump
            scores = scores_of_dump(path)
            per_game.update(scores)
            for tag in dict.fromkeys(k.split("|")[1] for k in scores):
                for ms in SWEEP_MIN_S:
                    v = [m[ms] for k, m in scores.items() if k.split("|")[1] == tag]
                    rows.append(dict(variant=variant, train=src, test=dst, gaps=tag, min_s=ms,
                                     **{k: float(np.mean([m[k] for m in v])) for k in v[0]}))
        mine = [r for r in rows if r["variant"] == variant]
        summary = {}
        for ms in SWEEP_MIN_S:
            at = [r for r in mine if r["min_s"] == ms]
            summary["selection_%gs" % ms] = float(np.mean(
                [r["spell_f1"] for r in at if r["gaps"] in e25.SELECTION]))
            for tag in ("std35", "std50", "art10", "dark", "sc_real"):
                summary["%s_%gs" % (tag, ms)] = float(np.mean(
                    [r["spell_f1"] for r in at if r["gaps"] == tag]))
        with open(os.path.join(OUT, variant + ".json"), "w") as handle:
            json.dump(dict(summary=summary, rows=mine, per_game=per_game), handle, indent=1,
                      default=float)
        print("== {}: selection {:.4f} (min_s 0/0.5/1/2/3: {}) | art10 {:.4f} | dark "
              "{:.4f} | sc_real {:.4f}".format(
                  variant, summary["selection_2s"],
                  " ".join("{:.4f}".format(summary["selection_%gs" % ms])
                           for ms in SWEEP_MIN_S),
                  summary["art10_2s"], summary["dark_2s"], summary["sc_real_2s"]), flush=True)
    if pool is not None:
        pool.shutdown()
    print()
    print(table(rows, list(rows[0].keys())))


if __name__ == "__main__":
    main(sys.argv[1:])
