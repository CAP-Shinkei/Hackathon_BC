"""E30: control acquisition by evidence - WHEN control starts vs WHEN it is confirmed.

Hypothesis (Álvaro, 2026-11-12): deciding possession changes by a fixed duration
(0.5/1/2/3 s) is the wrong axis. A 0.4 s contact can be a rebound back to the previous
team or a clean take ended by a first-time pass; what separates them is the episode's
evidence, not its length. The split tested here: candidates fire from the contact
sequence (so a sub-second control still gets proposed), a small regularized logistic
scores each episode from its aftermath clues (mtah/acquire.py), confirmation may arrive
late while the start is attributed back to the contact the evidence supports, and the
decision enters the HSMM as a per-boundary transition cost (mtah.hsmm trans_adjust).

What this experiment is allowed to conclude, in order:
  1. PLUMBING (this file, default mode): the proposal/feature/decision chain runs on the
     external games and the estimator can be fitted leave-one-game-out. Its scores are
     compared with a reference built by controlled(gt, 2 s) - that measures agreement with
     today's convention ONLY. It cannot validate the semantics: an estimator that imitates
     the 2 s rule scores perfectly against labels made by the 2 s rule, and that is the
     circle this experiment exists to break.
  2. SEMANTICS (needs people): --emit-worksheets writes one annotation worksheet per clean
     external game (external/control_annotation.py holds the contract: no duration rule,
     uncertain marked and excluded, nothing derived from the answer keys). --annotations
     evaluates the out-of-fold scores against completed worksheets, certain rows only.
     Until those worksheets exist and are filled, any spellF1-style number for E30 is
     plumbing, not evidence.

Rules position (deliberate): candidates, features and any fitting use the EXTERNAL games
only. Nothing here touches the competition match, its tracking, its video, its labels or
the public 5 minutes - the organizers' 2026-11-03 clarification (team inbox, "More Than A
Hack 2027 - Clarification of the rules for the Initial Challenge") requires training and
tuning on external data only, and the published RULES.md passage that might allow more is
unresolved, so this experiment stays on the strict reading.

Protocol, fixed before running:
  - games: the 10 external games (3 Metrica sample + 7 DFL), damaged at 35% and 50%
    live-missing; SkillCorner held out for a later real-gaps check.
  - candidates: mtah.acquire.propose on the damaged tracking (nearest team on the damaged
    ball; v1 does not gap-fill the ball for episode extraction - a listed limitation).
  - fitting: leave-one-GAME-out; every damaged copy of a game stays in one fold, as in
    e25_benchmark. C (and later weight/cap of transition_adjust) chosen on out-of-fold
    episodes only.
  - plumbing reference: for each candidate, acquired_ref = 1 if the majority of the
    frames in [t_start, t_start + 2 s) are the candidate's team under controlled(gt, 2.0).
    Printed next to every plumbing number as what it is: agreement with the duration rule.
  - annotation reference: certain rows only, matched by candidate_id, as control_annotation.

Proposed decision rule (a proposal for the team, not a validated result): E30 feeds the
submission decode only if, against the ANNOTATIONS (never against the duration-rule
reference), the episode score beats the fixed-rule baseline on precision/recall of
acquisitions AND, wired into candidate 1's decode on the external games, it does not
fragment possessions (n_pred near n_gt) or lose clearly on the selection cells. Until the
annotations exist, the answer is "not decided".

Status: code written 2026-11-12, NOT RUN - this file is the run plan, there are no
results yet. Unit tests: tests/test_acquire.py (synthetic tracking, no data needed).

    python experiments/e30_control_acquisition.py                    # plumbing vs 2 s rule
    python experiments/e30_control_acquisition.py --emit-worksheets out/e30/worksheets
    python experiments/e30_control_acquisition.py --annotations out/e30/worksheets
"""

import argparse
import os
import sys

import numpy as np

from common import REPO, dfl_games, external_games, table
from control_annotation import load_dir, match, summarize, write_worksheet
from controlled import controlled
from gapsim import Params, damage
from mtah import acquire
from mtah.evidence import nearest_team
from mtah.labels import A, H

OUT = os.path.join(REPO, "out", "e30")
MIN_S_REF = 2.0  # the duration-rule reference, labelled as such wherever it is used
DAMAGE = [("std35", dict(live_missing=0.35), 100), ("std50", dict(live_missing=0.50), 101)]


def game_episodes(ball, players, teams, pitch):
    """Candidates (with retro-attributed starts) and episode features for one game."""
    near = nearest_team(ball, players, teams, pitch)
    cands = acquire.propose(near, near)
    if not cands:
        return cands, np.zeros((0, len(acquire.FEATURES)), dtype=np.float32)
    X, _ = acquire.episode_features(ball, players, teams, pitch, near, near, cands)
    return cands, X


def reference_labels(cands, gt):
    """The PLUMBING reference: did this candidate survive the 2 s rule? This is agreement
    with the convention being replaced - useful to catch bugs, meaningless as validation."""
    ref = controlled(gt, MIN_S_REF)
    want = {acquire.HOME: H, acquire.AWAY: A}
    y = []
    for cand in cands:
        a = cand["t_start"]
        b = min(len(ref), a + int(acquire.AFTER_S * acquire.FPS))
        window = ref[a:b]
        y.append(int(len(window) and np.mean(window == want[cand["team"]]) > 0.5))
    return np.array(y)


def load_all(damaged_copies=True):
    """[(name, ball, players, teams, pitch, gt)] - clean games for worksheets, plus one
    damaged copy per profile per game for the plumbing fit."""
    games = []
    for g in external_games() + dfl_games():
        games.append((g["name"], g["ball"], g["players"], g["teams"], g["pitch"], g["gt"]))
        if damaged_copies:
            for tag, fields, seed in DAMAGE:
                ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"],
                                       g["pitch"], Params(**fields), seed=seed)
                games.append(("{}|{}".format(g["name"], tag), ball, players, g["teams"],
                              g["pitch"], g["gt"]))
    return games


def emit_worksheets(path):
    os.makedirs(path, exist_ok=True)
    for name, ball, players, teams, pitch, _ in load_all(damaged_copies=False):
        cands, _ = game_episodes(ball, players, teams, pitch)
        write_worksheet(os.path.join(path, name + ".csv"), name, cands)
        print("{}: {} candidates to annotate".format(name, len(cands)))
    print("\nContract: external/control_annotation.py - no duration rule, mark uncertain,",
          "never derive answers from the answer keys.")


def out_of_fold_scores(games):
    """Leave-one-game-out episode scores. Returns (prepared, {name: (cands, p_oof)}); every
    damaged copy of a base game stays in one fold."""
    base = sorted({name.split("|")[0] for name, *_ in games})
    prepared = {}
    for name, ball, players, teams, pitch, gt in games:
        cands, X = game_episodes(ball, players, teams, pitch)
        prepared[name] = (cands, X, reference_labels(cands, gt))
    scores = {}
    for held in base:
        tr_x = np.concatenate([prepared[n][1] for n in prepared
                               if n.split("|")[0] != held and len(prepared[n][1])])
        tr_y = np.concatenate([prepared[n][2] for n in prepared
                               if n.split("|")[0] != held and len(prepared[n][1])])
        pipe = acquire.fit_logistic(tr_x, tr_y)
        for n in prepared:
            if n.split("|")[0] == held and len(prepared[n][1]):
                scores[n] = (prepared[n][0], acquire.predict_acquire(pipe, prepared[n][1]))
    return prepared, scores


def plumbing(games):
    prepared, scores = out_of_fold_scores(games)
    rows = []
    for name in sorted(scores):
        cands, p = scores[name]
        y = prepared[name][2]
        s = summarize(y, p, np.zeros(len(y), dtype=int))
        rows.append(dict(game=name, n=s["n"], agree2s=s["accuracy"],
                         precision=s["precision"], recall=s["recall"]))
    print("PLUMBING ONLY - agreement with controlled(gt, {} s), the rule E30 is meant to "
          "replace. High numbers imitate the rule; they do not validate the "
          "semantics.\n".format(MIN_S_REF))
    print(table(rows, ["game", "n", "agree2s", "precision", "recall"]))


def annotated(games, path):
    answers = load_dir(path)
    _, scores = out_of_fold_scores(games)
    rows = []
    for name in sorted(set(answers) & set(scores)):
        cands, p = scores[name]
        pairs, missing, extra = match(cands, answers[name])
        if missing or extra:
            print("{}: worksheet drift - {} candidates unanswered, {} answers without a "
                  "candidate; scoring the matched {} only".format(
                      name, len(missing), len(extra), len(pairs)))
        by_id = {id(c): i for i, c in enumerate(cands)}
        y = np.array([a["acquired"] for _, a in pairs])
        unc = np.array([a["uncertain"] for _, a in pairs])
        pp = np.array([p[by_id[id(c)]] for c, _ in pairs])
        s = summarize(y, pp, unc)
        rows.append(dict(game=name, n=s["n"], n_uncertain=s["n_uncertain"],
                         accuracy=s["accuracy"], precision=s["precision"],
                         recall=s["recall"]))
    print("Against independent annotations (certain rows only). THIS is the reference that "
          "can accept or reject E30.\n")
    print(table(rows, ["game", "n", "n_uncertain", "accuracy", "precision", "recall"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-worksheets", metavar="DIR")
    ap.add_argument("--annotations", metavar="DIR")
    args = ap.parse_args()
    if args.emit_worksheets:
        emit_worksheets(args.emit_worksheets)
        return
    games = load_all()
    if args.annotations:
        annotated(games, args.annotations)
    else:
        plumbing(games)


if __name__ == "__main__":
    sys.exit(main())
