"""E15: would whistle boundaries help the candidate? A lawful external simulation.

Question (Alvaro, 2026-11-09): does adding the whistle help? He never tested it. No available
external dataset pairs match audio with possession labels, and checking anything on the
competition match is forbidden - so this experiment SIMULATES the whistle signal on the three
Metrica sample games, whose event data marks the whistled stoppages directly:

  whistle times = FAULT RECEIVED start frames (the whistle), plus events with OFFSIDE /
  FORCED / END HALF subtypes and SHOT-GOAL end frames (all whistled in football).

The integration tested is the only one compatible with the rules discussion of E14: decode
blocks are SPLIT at each (simulated detected) whistle so spells cannot cross it; nothing is
forced to dead by rule. Detector behaviour is simulated from the external-audio validation
(E14): recall sampled per whistle, false alarms uniform in graded time.

Protocol: leave-one-game-out over the three Metrica games. Train on the other two games
damaged at the submission's rates (v1.GAP_RATES, seed 100); test on the held-out game damaged
at 35%/50% live missing (seeds 7/8, the repo's test seeds). Controlled-possession rule 2 s.
NOTE: this first pass trains on Metrica games only (DFL adds long XML parsing); the full
submission mix is a one-line change (add dfl_games() to train) for the team's rerun.

    python experiments/e15_whistle_benefit.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external"))

import metrica_sample as ms  # noqa: E402
from common import load_game  # noqa: E402
from gapsim import Params, damage  # noqa: E402
from e12_visibility_ablation import featurise, fit  # noqa: E402
from mtah import v1, v2  # noqa: E402
from mtah.labels import graded_blocks  # noqa: E402
from mtah.scoring import score  # noqa: E402

TEST_RATES = (0.35, 0.5)
# (tag, recall, jitter_s, n_fa, structured) - oracle signal first, then degraded variants.
VARIANTS = (("oracle", 1.0, 0.0, 0, False),
            ("jitter1s", 1.0, 1.0, 0, False),
            ("real", 0.85, 1.0, 3, False),
            ("real_struct", 0.85, 1.0, 3, True))
FPS = 25


def exciting_frames(game):
    """[(start, end)] array-index windows around shots/goals: where crowd whistles cluster."""
    ev = ms.load_events(game)
    shots = ev[ev["type"] == "SHOT"]
    return [(int(s) - 1, int(e) - 1) for s, e in zip(shots["start"], shots["end"])]


def whistle_frames(game):
    """Array indices (frame - 1) of the whistled stoppages of a Metrica sample game."""
    ev = ms.load_events(game)
    w = list(ev.loc[ev["type"] == "FAULT RECEIVED", "start"])
    stop_sub = ev["subtype"].fillna("").str.contains("OFFSIDE|FORCED|END HALF", regex=True)
    w += list(ev.loc[stop_sub, "end"])
    goal = (ev["type"] == "SHOT") & ev["subtype"].fillna("").str.contains("GOAL")
    w += list(ev.loc[goal, "end"])
    return sorted(int(f) - 1 for f in w)  # events count frames from 1


def simulate(whistles, n, blocks, recall, jitter_s, n_fa, structured, exciting, seed):
    """Detected whistle indices from the true whistled stoppages: each kept with prob recall,
    displaced by uniform +/- jitter (event annotations are not the exact acoustic instant),
    plus n_fa false alarms - uniform in graded time, or structured near shots/goals (crowd
    whistles cluster at exciting moments; uniform alone would be optimistic)."""
    rng = np.random.default_rng(seed)
    j = int(jitter_s * FPS)
    det = [min(max(w + int(rng.integers(-j, j + 1)) if j else w, 0), n - 1)
           for w in whistles if rng.random() < recall]
    if n_fa:
        if structured and len(exciting):
            lo = np.concatenate([np.arange(max(a - 5 * FPS, 0), min(b + 15 * FPS, n))
                                 for a, b in exciting])
        else:
            lo = np.concatenate([np.arange(a, b) for a, b in blocks])
        cand = rng.choice(lo, size=min(n_fa * 10, len(lo)), replace=False)
        added = 0
        for c in cand:
            if all(abs(c - w) > 2 * FPS for w in whistles):
                det.append(int(c))
                added += 1
                if added == n_fa:
                    break
    return sorted(det)


def split_blocks(blocks, cuts):
    out = []
    for a, b in blocks:
        pts = [a] + [c for c in cuts if a < c < b] + [b]
        out += [(s, e) for s, e in zip(pts, pts[1:]) if e - s >= FPS]
    return out



FEAT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "cache", "e15")


def cached_featurise(g, ball, players, rate, seed, tag):
    """featurise() cached to disk: the featurisation dominates the runtime."""
    os.makedirs(FEAT_CACHE, exist_ok=True)
    path = os.path.join(FEAT_CACHE, "{}_{:.2f}_{}_{}.npz".format(g["name"], rate, seed, tag))
    if os.path.isfile(path):
        z = np.load(path, allow_pickle=False)
        return dict(game=g["name"], tag=tag, X=z["X"], names=list(z["names"]), y=z["y"],
                    baseline=z["baseline"])
    d = featurise(g["name"], ball, players, g["teams"], g["pitch"], g["gt"], tag)
    np.savez(path, X=d["X"].astype(np.float32), names=np.array(d["names"]), y=d["y"],
             baseline=d["baseline"])
    return d

MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "cache",
                           "e15_models")


def staged_fit(train, fold_tag):
    """v2.fit + add_hsmm with every stage cached to disk (one stage per run window).

    Same maths as e12's fit(): 3-fold out-of-fold classifier training, temperature/min-spell
    choice on OOF predictions, final classifier on everything, HSMM durations on the labels.
    """
    import pickle

    from mtah.hmm import count_tables
    from mtah.hsmm import count_durations

    os.makedirs(MODEL_CACHE, exist_ok=True)
    train = [(d["game"], d["X"], d["y"], d["names"]) for d in train]

    def stage(name, fn):
        path = os.path.join(MODEL_CACHE, "{}_{}.pkl".format(fold_tag, name))
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        out = fn()
        with open(path, "wb") as f:
            pickle.dump(out, f)
        print("stage {} done".format(name), flush=True)
        return out

    ids = sorted({gid for gid, *_ in train})
    folds = [ids[i::3] for i in range(3)]

    def fit_fold(i):
        return v2.train_classifier([(Xg, yg, nm) for gid, Xg, yg, nm in train
                                    if gid not in folds[i]])
    clfs = [stage("clf_fold{}".format(i), lambda i=i: fit_fold(i)) for i in range(3)]

    def build_oof():
        return [(yg, v2.symmetric_proba(clfs[ids.index(gid) // 1 if False else
                                        next(i for i, f in enumerate(folds) if gid in f)],
                                        Xg, nm))
                for gid, Xg, yg, nm in train if gid in
                folds[next(i for i, f in enumerate(folds) if gid in f)]]
    oof = stage("oof", build_oof)

    def build_tables():
        _, log_trans = count_tables([(yg, np.full(len(yg), -1, np.int8))
                                     for _, _, yg, _ in train])
        return {"log_trans": log_trans, "prior": v2.class_prior([yg for _, _, yg, _ in train])}
    tables = stage("tables", build_tables)

    def choose_combo(ti, mi):
        import itertools as _it
        T = v2.TEMPERATURES[ti]
        M = v2.MIN_SPELLS[mi]
        return float(np.mean([score(gt, v2.decode(pr, graded_blocks(gt), tables["log_trans"],
                                                   tables["prior"], T, M, len(gt)))[
                              "spell_f1_mean"] for gt, pr in oof]))

    def choose_all():
        best = None
        for ti in range(len(v2.TEMPERATURES)):
            for mi in range(len(v2.MIN_SPELLS)):
                f = stage("choose_{}_{}".format(ti, mi), lambda ti=ti, mi=mi:
                          choose_combo(ti, mi))
                T, M = v2.TEMPERATURES[ti], v2.MIN_SPELLS[mi]
                if best is None or f > best[2]:
                    best = (T, M, f)
        return best

    temp, ms_, f = choose_all()
    clf_full = stage("clf_full", lambda: v2.train_classifier([(Xg, yg, nm)
                                                              for _, Xg, yg, nm in train]))
    model = {"clf": clf_full, "log_trans": tables["log_trans"], "prior": tables["prior"],
             "temperature": temp, "min_spell": ms_, "oof_spell_f1": f, "oof": oof}

    def hsmm_temp(ti):
        from mtah.hsmm import count_durations as _cd
        log_dur = stage("hsmm_durations", lambda: _cd([yg for _, yg, *_ in
                                                       [(d,) for d in train]])[0] if False else
                        _cd([d_y for _, _, d_y, _ in train]))
        T = v2.HSMM_TEMPS[ti]
        from mtah.hsmm import count_durations  # noqa
        return None
    # simpler: cache durations separately, then the per-temperature scores together
    log_dur_lt = stage("hsmm_durations2", lambda: count_durations([d_y for _, _, d_y, _
                                                                   in train]))

    def hsmm_score_t(ti):
        T = v2.HSMM_TEMPS[ti]
        log_dur, log_trans = log_dur_lt
        return float(np.mean([score(gt, v2.hsmm_decode(pr, graded_blocks(gt), model["prior"],
                                                       log_dur, log_trans, T, len(gt)))[
                              "spell_f1_mean"] for gt, pr in model["oof"]]))

    def build_hsmm2():
        best = None
        for ti in range(len(v2.HSMM_TEMPS)):
            sc = stage("hsmm_T{}".format(ti), lambda ti=ti: hsmm_score_t(ti))
            if best is None or sc > best[1]:
                best = (v2.HSMM_TEMPS[ti], sc)
        return {"log_dur": log_dur_lt[0], "log_trans": log_dur_lt[1], "T": best[0],
                "f": best[1]}
    h = build_hsmm2()
    model.update(decoder="hsmm", log_dur=h["log_dur"], hsmm_log_trans=h["log_trans"],
                 hsmm_temperature=h["T"], oof_spell_f1_hsmm=h["f"])
    model["oof"] = None  # free the OOF probas once the model is assembled
    return model


def main():
    games = {g: load_game(g) for g in ms.GAMES}
    whistles = {g: whistle_frames(g) for g in ms.GAMES}
    for g in ms.GAMES:
        print("game {}: {} whistled stoppages in events".format(g, len(whistles[g])), flush=True)

    rows = []
    for test_g in ms.GAMES:
        train = []
        for g in ms.GAMES:
            if g == test_g:
                continue
            for i, r in enumerate(v1.GAP_RATES):
                ball, players = damage(games[g]["ball"], games[g]["players"], games[g]["teams"],
                                       games[g]["gt"], games[g]["pitch"],
                                       Params(live_missing=r), seed=100 + i)
                train.append(cached_featurise(games[g], ball, players, r, 100 + i,
                                              "train{:.0%}".format(r)))
        global train_raw
        train_raw = train
        model = staged_fit(train, "fold{}".format(test_g))
        del train
        print("game {}: model fitted".format(test_g), flush=True)

        for i, rate in enumerate(TEST_RATES):
            ball, players = damage(games[test_g]["ball"], games[test_g]["players"],
                                   games[test_g]["teams"], games[test_g]["gt"],
                                   games[test_g]["pitch"], Params(live_missing=rate),
                                   seed=7 + i)
            d = cached_featurise(games[test_g], ball, players, rate, 7 + i,
                                 "sim{:.0%}".format(rate))
            blocks = graded_blocks(d["y"])
            exciting = exciting_frames(test_g)
            variants = [("base", blocks)]
            for tag, rec, jit, n_fa, struct in VARIANTS:
                det = simulate(whistles[test_g], len(d["y"]), blocks, rec, jit, n_fa, struct,
                               exciting, seed=3)
                variants.append(("wh_" + tag, split_blocks(blocks, det)))
            for tag, bl in variants:
                import pickle as _pkl
                pcache = os.path.join(MODEL_CACHE, "pred_{}_{:.2f}_{}.pkl".format(test_g, rate,
                                                                                  tag))
                if os.path.isfile(pcache):
                    with open(pcache, "rb") as fh:
                        pred = _pkl.load(fh)
                else:
                    pred = v2.predict(model, d["X"], d["names"], bl, len(d["y"]))
                    with open(pcache, "wb") as fh:
                        _pkl.dump(pred, fh)
                s = score(d["y"], pred)
                rows.append((test_g, rate, tag, s["spell_f1_mean"], s["d_f1"],
                             s["n_pred_spells"], s["n_gt_spells"]))
                print("game {} rate {:.0%} {:9s} spellF1 {:.3f} dF1 {:.3f} spells {}/{}"
                      .format(test_g, rate, tag, *rows[-1][3:]), flush=True)
        del d

    print("\n== mean over games ==")
    for rate in TEST_RATES:
        for tag in ["base"] + ["wh_" + t for t, _, _, _, _ in VARIANTS]:
            vals = [r[3] for r in rows if r[1] == rate and r[2] == tag]
            if vals:
                print("rate {:.0%} {:9s} spellF1 {:.3f}".format(rate, tag, float(np.mean(vals))))


if __name__ == "__main__":
    main()
