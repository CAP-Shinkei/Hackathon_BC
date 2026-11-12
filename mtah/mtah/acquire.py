"""Control acquisition at episode level: WHEN control starts is not WHEN it is confirmed.

The pipeline today decides possession changes in one of two ways, both by duration: the
2 s "controlled" rule on the labels (external/controlled.py), and the HSMM's duration
distributions on the decoded sequence. A fixed duration cannot tell a deflection from a
controlled touch: a 0.4 s contact can be a rebound that goes back to the previous team, or a
clean take followed by a first-time pass. What separates them is what happens NEXT, not how
long the contact lasts.

This module separates the two questions (Álvaro, 2026-11-12):

  t_start        when control was gained - attributed back to the contact that the evidence
                 supports, NOT automatically to the first brush of the ball;
  t_confirm      when enough evidence has arrived to believe it - later is fine, confirming
                 late does not force the possession to start late.

Mechanism (deliberately small, E30):
  1. Candidate acquisitions are proposed from the contact sequence itself (runs of which
     team is nearest to the ball), so a short control that the 2 s rule would erase still
     generates a candidate. Candidates can also come from a decoded sequence; any source
     that emits (t0, team) pairs works.
  2. Each candidate episode is scored by ONE regularized logistic regression (not a second
     per-frame classifier, not a transformer) on a handful of episode summaries built from
     the same tracking clues mtah.features already computes: who is nearest and by how much,
     how the ball's velocity breaks at the entry and the exit of the contact, whether the
     next contact belongs to the same team, and how much of the episode is even visible.
  3. A decision confirms or rejects the candidate. Confirmed: the start is attributed to
     the first raw contact of the acquiring team inside the episode (choose_start), which
     can precede the moment the candidate fired. Rejected: possession stays with the
     previous owner. "Contested" is an internal state of an unconfirmed candidate, never a
     fourth label.
  4. The decision enters decoding as a per-boundary log-odds adjustment on the HSMM's
     team-to-team transition (mtah.hsmm.decode(..., trans_adjust=...)), placed at t_start.
     Frames without a candidate carry no adjustment, so the decoder is unchanged wherever
     the episode model has nothing to say. The 2 s rule is never re-applied after this:
     a fixed duration must not erase decisions the evidence rule just made.

What this module is NOT:
  - Not trained, tuned or validated on the competition match, and not on labels built by
    the 2 s rule: that would teach it to imitate the rule it is meant to replace. The
    semantic reference is an independent annotation of candidate episodes (see
    external/control_annotation.py); until those exist, everything here is plumbing.
  - Not a claim about any player: nothing here assumes a track slot is a person, and no
    feature is keyed to an identity.
"""

import numpy as np

from mtah import features as F
from mtah.evidence import NOBODY

FPS = 25
# De-flicker vote for the candidate sequence: 3-frame half-width (~0.12 s), much shorter
# than turnover's 0.5 s feature vote on purpose - a candidate vote that smooths over a
# 0.4 s contact would erase exactly the episodes this module exists to score. Robustness
# to flicker is a feature's job (raw_contacts_1s counts it), not the proposal's.
VOTE_HALF = 3
LOOSE_CAP_S = 10.0   # gaps between contacts stop telling anything after this
RUN_CAP_S = 30.0
AFTER_S = 2.0        # aftermath window: what the contact turned into
NEXT_CAP_S = 3.0     # a next contact farther away than this is not part of the episode
ENTRY_HALF = F.PASS  # +-1 s around t_start / run end for the velocity-break clues


# Voted team codes (same convention as features.compute_turnover).
HOME, AWAY, NONE = 1, -1, 0


def voted_team(near_filled, half=VOTE_HALF):
    """(n,) HOME / AWAY / NONE per frame: a short rolling vote of the nearest team."""
    nh = np.where(near_filled == NOBODY, np.nan, (near_filled == 0).astype(float))
    share = F._rolling_mean(nh, half)
    return np.where(np.isnan(share) | (share == 0.5), NONE, np.where(share > 0.5, HOME, AWAY))


def candidates(near_filled, n_frames=None):
    """Candidate control acquisitions from the contact sequence.

    A candidate fires at the start of every voted-team run whose team differs from the
    previous voted owner, however short the run: a sub-second control that the 2 s rule
    would erase is exactly what must still be proposed. Returns a list of dicts with
    t0 (the voted boundary), team and prev_team (HOME/AWAY codes), run bounds, the previous
    owner's run bounds and the next run's start (for confirmation bookkeeping).
    """
    if n_frames is None:
        n_frames = len(near_filled)
    team = voted_team(near_filled)
    starts, stops, _ = F._runs_of(team)
    out = []
    prev = None  # (team, start, stop) of the last home/away run
    run_list = [(int(team[a]), int(a), int(b)) for a, b in zip(starts, stops)]
    for i, (c, a, b) in enumerate(run_list):
        nxt = run_list[i + 1][1] if i + 1 < len(run_list) else None
        if c in (HOME, AWAY):
            if prev is not None and prev[0] != c:
                out.append(dict(t0=a, team=c, prev_team=prev[0], run_start=a, run_stop=b,
                                prev_stop=prev[2], next_start=nxt))
            prev = (c, a, b)
    return out


def choose_start(near, cand, window=F.PASS):
    """Retro-attribution of the start: the first raw contact of the acquiring team inside
    the episode. The voted boundary t0 fires only once the team dominates the 0.5 s vote;
    the raw nearest-team run that made it fire usually starts earlier. We walk back from t0
    to the start of that raw run - and never further: the first brush of the ball is not
    automatically the start, only the contact the episode's evidence is about.
    """
    own = 0 if cand["team"] == HOME else 1
    t0 = cand["t0"]
    t = t0
    if near[t] != own:  # the vote can fire on a frame the raw sequence gives to nobody
        fwd = t0 + 1 + int(np.argmax(near[t0 + 1:t0 + 1 + window] == own)) \
            if (near[t0 + 1:t0 + 1 + window] == own).any() else t0
        t = fwd
    while t > 0 and near[t - 1] == own:
        t -= 1
    return t


def propose(near, near_filled):
    """Candidates with the start retro-attributed: the full episode proposal step."""
    cands = candidates(near_filled)
    for cand in cands:
        cand["t_start"] = choose_start(near, cand)
    return cands


def _episode_frames(ball, players, teams, pitch):
    """Per-frame arrays the episode summaries are built from (same maths as
    features.compute_turnover, kept per frame so each episode summarizes its own window)."""
    scale = np.asarray(pitch, dtype=float)
    bm = ball * scale
    n = len(ball)
    seen = ~np.isnan(ball).any(axis=1)
    d = [F._nearest_m(players, teams, bm, scale, t) for t in (0, 1)]
    before = np.full((n, 2), np.nan)
    after = np.full((n, 2), np.nan)
    before[F.KIN:] = (bm[F.KIN:] - bm[:-F.KIN]) / (F.KIN / FPS)
    after[:-F.KIN] = (bm[F.KIN:] - bm[:-F.KIN]) / (F.KIN / FPS)
    kink = np.linalg.norm(after - before, axis=1)
    speed = 0.5 * np.linalg.norm(after + before, axis=1)
    closing = [np.full(n, np.nan), np.full(n, np.nan)]
    for t in (0, 1):
        closing[t][F.KIN:-F.KIN] = (d[t][:-2 * F.KIN] - d[t][2 * F.KIN:]) / (2 * F.KIN / FPS)
    return seen, d, kink, speed, closing


def episode_features(ball, players, teams, pitch, near, near_filled, cands):
    """(n_candidates, F) float32 matrix and feature names, one row per candidate.

    Every feature is written from the ACQUIRING team's point of view ("own" vs "other"),
    so rows are team-symmetric by construction and the estimator has no team identity to
    learn. NaN means "the tracking does not say"; imputation is the model's job, not the
    features'.
    """
    n = len(ball)
    seen, d, kink, speed, closing = _episode_frames(ball, players, teams, pitch)
    team = voted_team(near_filled)
    own_near = lambda c: (near == (0 if c == HOME else 1))  # noqa: E731
    names = ["run_len_s", "margin_mean", "margin_start", "seen_run", "seen_after",
             "kink_entry", "kink_exit", "after_same_2s", "next_same", "t_next_s",
             "prev_gap_s", "closing_own", "closing_other", "raw_contacts_1s",
             "ball_speed_run"]
    rows = []
    for cand in cands:
        c = cand["team"]
        own, other = (0, 1) if c == HOME else (1, 0)
        t0, stop = cand["t0"], cand["run_stop"]
        t_start = cand.get("t_start", t0)
        run = np.arange(t_start, stop)
        entry = own_near(c)[max(0, t_start - ENTRY_HALF):t_start + ENTRY_HALF]
        exitw = own_near(c)[max(0, stop - ENTRY_HALF):stop + ENTRY_HALF]

        def kink_over(mask_idx):
            vals = kink[mask_idx]
            return np.nanmax(vals) if len(vals) else np.nan

        entry_idx = np.flatnonzero(entry) + max(0, t_start - ENTRY_HALF)
        exit_idx = np.flatnonzero(exitw) + max(0, stop - ENTRY_HALF)
        after = team[stop:min(n, stop + int(AFTER_S * FPS))]
        known = after != NONE
        nxt_same, t_next = np.nan, NEXT_CAP_S
        if cand.get("next_start") is not None:
            nxt = team[cand["next_start"]]
            if (cand["next_start"] - stop) / FPS <= NEXT_CAP_S:
                nxt_same = float(nxt == c)
                t_next = (cand["next_start"] - stop) / FPS
        raw_own = own_near(c)
        br = np.flatnonzero(raw_own[1:] & ~raw_own[:-1]) + 1
        raw_starts = br[(br >= t0 - FPS) & (br <= t0 + FPS)]
        rows.append([
            min((stop - t0) / FPS, RUN_CAP_S),
            np.clip(np.nanmean((d[other] - d[own])[run]), -10, 10),
            np.clip((d[other] - d[own])[t_start], -10, 10),
            np.mean(seen[t_start:stop]) if stop > t_start else np.nan,
            np.mean(seen[t0:min(n, t0 + int(AFTER_S * FPS))]),
            kink_over(entry_idx),
            kink_over(exit_idx),
            np.mean(after[known] == c) if known.any() else np.nan,
            nxt_same,
            t_next,
            min((t0 - cand["prev_stop"]) / FPS, LOOSE_CAP_S),
            closing[own][t0],
            closing[other][t0],
            float(len(raw_starts)),
            np.nanmean(speed[run]),
        ])
    X = np.array(rows, dtype=np.float32).reshape(len(cands), len(names))
    return X, names


FEATURES = ["run_len_s", "margin_mean", "margin_start", "seen_run", "seen_after",
            "kink_entry", "kink_exit", "after_same_2s", "next_same", "t_next_s",
            "prev_gap_s", "closing_own", "closing_other", "raw_contacts_1s",
            "ball_speed_run"]


def fit_logistic(X, y, C=1.0, seed=0):
    """One small regularized logistic regression over episode summaries.

    Deliberately not a sequence model: an episode has a handful of clues, not 25
    correlated posteriors per second to multiply into fictitious confidence. Median
    imputation + standardization + L2; C is a hyperparameter to choose on out-of-fold
    games only, and only once independent annotations exist to choose it against.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000, random_state=seed))
    pipe.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int))
    return pipe


def predict_acquire(pipe, X):
    return pipe.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def decide(cands, p_acquire, threshold=0.5):
    """One decision per candidate: confirmed or rejected, with both times kept apart.

    t_start comes from choose_start (retro-attribution); t_confirm is bookkeeping: the
    first frame by which every clue the score consumed has arrived - the pass that
    confirms a first-time control when it arrives, else the end of the aftermath window.
    A rejected candidate means possession stays with the previous owner; an episode that
    is neither confirmed nor rejected yet is "contested" internally and is never emitted
    as a label.
    """
    out = []
    for cand, p in zip(cands, p_acquire):
        stop = cand["run_stop"]
        nxt = cand.get("next_start")
        t_confirm = nxt if (nxt is not None and nxt - stop <= int(NEXT_CAP_S * FPS)) \
            else stop + int(AFTER_S * FPS)
        out.append(dict(team=cand["team"], t_start=int(cand.get("t_start", cand["t0"])),
                        t_confirm=int(t_confirm), p=float(p),
                        confirmed=bool(p >= threshold)))
    return out


def transition_adjust(decisions, n_frames, weight=2.0, cap=3.0, threshold=0.5):
    """(n_frames + 1,) log-odds adjustment for mtah.hsmm.decode(trans_adjust=...).

    Placed at t_start, not at t_confirm: confirming late does not move the start back.
    Positive where the episode model believes control changed hands (switching is
    cheaper), negative where it rejects the candidate (the deflection stays with the
    previous owner). Zero everywhere else: where there is no candidate, the decoder sees
    exactly the same costs as today. weight and cap bound how far the episode evidence may
    move the decode; they are hyperparameters to fit on external games only.
    """
    adj = np.zeros(n_frames + 1)
    z0 = _logit(threshold)
    for d in decisions:
        z = weight * (_logit(d["p"]) - z0)
        adj[min(max(d["t_start"], 0), n_frames)] += float(np.clip(z, -cap, cap))
    return adj
