"""v2: per-frame features -> gradient-boosted classifier -> hybrid HMM -> Viterbi -> merge short spells.

The classifier gives P(state | features) per frame; dividing by the class prior turns it into a
scaled likelihood for the same Viterbi decoder as v1 (a "hybrid" HMM). Training is
team-symmetric: every frame is also added with home and away exchanged.

Everything is fitted on gap-damaged external games. Temperature and minimum spell length are
chosen on out-of-fold predictions (games split into folds), so the choice never sees a
classifier trained on the same game.
"""

import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah import features
from mtah.hmm import count_tables, merge_short, _viterbi
from mtah.labels import A, D, H, X, graded_blocks
from mtah.scoring import score

TEMPERATURES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
MIN_SPELLS = (1, 25, 50, 75, 100)
SUBSAMPLE = 4
SWAP = np.array([A, H, D])


def train_classifier(items, seed=0):
    """items: [(X, y)] per damaged game; y has X (3) on ungraded frames, which are dropped."""
    Xs, ys = [], []
    for Xg, yg, names in items:
        keep = np.flatnonzero(yg != X)[::SUBSAMPLE]
        Xs += [Xg[keep], features.swap_teams(Xg[keep], names)]
        ys += [yg[keep], SWAP[yg[keep]]]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(np.concatenate(Xs), np.concatenate(ys))
    return clf


def symmetric_proba(clf, Xg, names):
    """Average of the prediction and the team-swapped prediction mapped back: exactly symmetric."""
    p = clf.predict_proba(Xg)
    q = clf.predict_proba(features.swap_teams(Xg, names))[:, SWAP]
    return 0.5 * (p + q)


def decode(proba, blocks, log_trans, prior, temperature, min_spell, n_frames):
    emis = temperature * (np.log(np.clip(proba, 1e-6, 1)) - np.log(prior))
    out = np.full(n_frames, X, dtype=np.int8)
    log_p0 = np.log(np.array([0.45, 0.45, 0.10]))
    for a, b in blocks:
        out[a:b] = _viterbi(np.ascontiguousarray(emis[a:b]), log_trans, log_p0)
    return merge_short(out, blocks, min_spell)


def choose(oof, log_trans, prior):
    """(temperature, min_spell, mean spellF1) maximising mean spellF1 over out-of-fold games."""
    best = None
    for temp, ms in itertools.product(TEMPERATURES, MIN_SPELLS):
        f = float(np.mean([score(gt, decode(p, graded_blocks(gt), log_trans, prior, temp, ms,
                                            len(gt)))["spell_f1_mean"] for gt, p in oof]))
        if best is None or f > best[2]:
            best = (temp, ms, f)
    return best


def class_prior(labels):
    y = np.concatenate([l[l != X] for l in labels])
    c = np.bincount(y, minlength=3).astype(float)
    c[H] = c[A] = 0.5 * (c[H] + c[A])
    return c / c.sum()


def fit(games_items, n_folds=3, seed=0):
    """games_items: [(game_id, X, y, names)] (several damaged copies per game share game_id).
    Returns a model dict with the classifier trained on everything."""
    ids = sorted({gid for gid, *_ in games_items})
    folds = [ids[i::n_folds] for i in range(n_folds)]
    oof = []
    for fold in folds:
        train = [(Xg, yg, nm) for gid, Xg, yg, nm in games_items if gid not in fold]
        clf = train_classifier(train, seed)
        for gid, Xg, yg, nm in games_items:
            if gid in fold:
                oof.append((yg, symmetric_proba(clf, Xg, nm)))
    _, log_trans = count_tables([(yg, np.full(len(yg), -1, np.int8))
                                 for _, _, yg, _ in games_items])
    prior = class_prior([yg for _, _, yg, _ in games_items])
    temp, ms, f = choose(oof, log_trans, prior)
    clf = train_classifier([(Xg, yg, nm) for _, Xg, yg, nm in games_items], seed)
    return {"clf": clf, "log_trans": log_trans, "prior": prior, "temperature": temp,
            "min_spell": ms, "oof_spell_f1": f, "oof": oof}


HSMM_TEMPS = (0.1, 0.2, 0.3, 0.5, 1.0)


def hsmm_decode(proba, blocks, prior, log_dur, log_trans, temperature, n_frames):
    from mtah import hsmm
    emis = temperature * (np.log(np.clip(proba, 1e-6, 1)) - np.log(prior))
    return hsmm.decode(emis, blocks, log_dur, log_trans, n_frames)


def add_hsmm(model, labels):
    """Switch a fitted model to the duration-aware decoder: durations and spell transitions
    counted on the training labels, temperature chosen on the out-of-fold predictions."""
    from mtah import hsmm
    log_dur, log_trans = hsmm.count_durations(labels)
    best = None
    for T in HSMM_TEMPS:
        f = float(np.mean([score(gt, hsmm_decode(p, graded_blocks(gt), model["prior"], log_dur,
                                                 log_trans, T, len(gt)))["spell_f1_mean"]
                           for gt, p in model["oof"]]))
        if best is None or f > best[1]:
            best = (T, f)
    model.update(decoder="hsmm", log_dur=log_dur, hsmm_log_trans=log_trans,
                 hsmm_temperature=best[0], oof_spell_f1_hsmm=best[1])
    return model


def predict(model, Xg, names, blocks, n_frames):
    if model.get("decoder") == "hsmm":
        p = symmetric_proba(model["clf"], Xg, names)
        return hsmm_decode(p, blocks, model["prior"], model["log_dur"], model["hsmm_log_trans"],
                           model["hsmm_temperature"], n_frames)
    p = symmetric_proba(model["clf"], Xg, names)
    return decode(p, blocks, model["log_trans"], model["prior"], model["temperature"],
                  model["min_spell"], n_frames)
