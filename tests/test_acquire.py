"""Tests for the episode-level control-acquisition pieces of E30.

Synthetic tracking only: no external data, no trained model, no official scorer, so they
run anywhere (`python -m pytest tests/test_acquire.py`). They pin the semantics the
experiment relies on: short contacts still fire candidates, the start is attributed back
to the contact the evidence supports, confirmation can come later, and the HSMM's default
decode is untouched unless episode evidence is passed in.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "external"))
sys.path.insert(0, REPO)

from control_annotation import match, read_annotations, summarize, write_worksheet  # noqa: E402
from mtah import acquire, hsmm  # noqa: E402
from mtah.evidence import nearest_team  # noqa: E402
from mtah.labels import A, D, H  # noqa: E402

FPS = 25
PITCH = (105.0, 68.0)


def scenario(clean_take=True, deflection=True, n=750, brush=False):
    """Ball and 4 track slots (2 home, 2 away) for: home controls, away takes and plays a
    first-time pass to a teammate, then (optionally) a 0.4 s home deflection goes back to
    away. Returns (ball, players, teams, near)."""
    ball = np.full((n, 2), np.nan)
    players = np.full((n, 4, 2), np.nan)
    teams = np.tile(np.array([0, 0, 1, 1], dtype=np.int8), (n, 1))
    home0, home1 = np.array([0.40, 0.50]), np.array([0.30, 0.40])
    away0, away1 = np.array([0.80, 0.60]), np.array([0.85, 0.45])
    off = np.array([2.0 / PITCH[0], 0.0])  # controller stands 2 m from the ball
    take, played, land = 250, 260, 275
    defl, back = 500, 510
    for t in range(n):
        if t < take or (deflection and defl <= t < back):
            owner, pos = 0, home0        # home controls (or the short deflection)
        elif take <= t < played:
            owner, pos = 2, away0 if not clean_take else home0 + np.array([0.02, 0.0])
        elif played <= t < land:
            f = (t - played) / (land - played)   # the pass flight, ~19 m in 0.6 s
            pos = home0 + f * (np.array([0.60, 0.45]) - home0)
            owner = None
        else:
            owner, pos = 3, np.array([0.60, 0.45])   # away teammate controls
        ball[t] = pos
        players[t] = [home0, home1, away0, away1]
        if owner is not None:
            players[t, owner] = pos + off
        if brush and 240 <= t < 243:
            players[t, 2] = pos + off   # a brief away brush 7 frames before the real take
    near = nearest_team(ball, players, teams, PITCH)
    return ball, players, teams, near


def test_short_contacts_still_fire_candidates():
    ball, players, teams, near = scenario()
    cands = acquire.propose(near, near)
    teams_seen = [c["team"] for c in cands]
    assert acquire.AWAY in teams_seen          # the clean take
    home = [c for c in cands if c["team"] == acquire.HOME]
    assert home                                # the 0.4 s deflection is proposed too
    assert (home[0]["run_stop"] - home[0]["run_start"]) < FPS  # and it is short


def test_start_is_retro_attributed_to_the_raw_contact():
    ball, players, teams, near = scenario()
    cands = acquire.propose(near, near)
    take = next(c for c in cands if c["team"] == acquire.AWAY)
    assert take["t_start"] <= take["t0"]       # never later than the voted boundary
    assert near[take["t_start"]] == 1          # a real away contact, not the vote's frame
    assert abs(take["t_start"] - 250) <= 5


def test_start_selection_prefers_supported_contact_over_first_brush():
    # The agreed rule: the best-SUPPORTED start inside the episode wins, so a short brush
    # just before the real take must not pull the possession start back to it.
    ball, players, teams, near = scenario(brush=True)
    cands = acquire.propose(near, near)
    take = next(c for c in cands if c["team"] == acquire.AWAY)
    assert abs(take["t_start"] - 250) <= 3        # the take, not the brush at 240
    assert near[take["t_start"]] == 1


def test_episode_features_separate_take_from_deflection():
    ball, players, teams, near = scenario()
    cands = acquire.propose(near, near)
    X, names = acquire.episode_features(ball, players, teams, PITCH, near, near, cands)
    assert X.shape == (len(cands), len(acquire.FEATURES))
    idx = {k: i for i, k in enumerate(names)}
    take_row = X[[c["team"] for c in cands].index(acquire.AWAY)]
    defl_row = X[[c["team"] for c in cands].index(acquire.HOME)]
    # The take is followed by same-team evidence (the pass lands with a teammate); the
    # deflection is followed by the previous team again.
    assert take_row[idx["after_same_2s"]] > defl_row[idx["after_same_2s"]]
    assert np.isfinite(take_row[idx["kink_exit"]])  # the pass is a velocity break


def test_logistic_learns_the_synthetic_episode_types():
    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for i in range(6):
        ball, players, teams, near = scenario(deflection=bool(i % 2))
        cands = acquire.propose(near, near)
        if not cands:
            continue
        X, _ = acquire.episode_features(ball, players, teams, PITCH, near, near, cands)
        X = X + rng.normal(0, 0.01, X.shape)  # jitter so it is a fit, not a lookup
        y = np.array([c["team"] == acquire.AWAY for c in cands], dtype=int)
        Xs.append(X)
        ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    pipe = acquire.fit_logistic(X, y)
    p = acquire.predict_acquire(pipe, X)
    assert ((p >= 0.5) == y.astype(bool)).all()


def test_decide_keeps_start_and_confirmation_apart():
    ball, players, teams, near = scenario()
    cands = acquire.propose(near, near)
    p = np.array([0.9 if c["team"] == acquire.AWAY else 0.1 for c in cands])
    out = acquire.decide(cands, p)
    for d in out:
        assert d["t_start"] <= d["t_confirm"]  # confirming late never moves the start back
    take = next(d for d in out if d["team"] == acquire.AWAY and d["confirmed"])
    defl = next(d for d in out if d["team"] == acquire.HOME)
    assert not defl["confirmed"]               # a deflection stays with the previous owner


def test_transition_adjust_places_signed_evidence_at_t_start():
    n = 300
    decisions = [dict(team=acquire.AWAY, t_start=100, t_confirm=140, p=0.95, confirmed=True),
                 dict(team=acquire.HOME, t_start=200, t_confirm=230, p=0.05, confirmed=False)]
    adj = acquire.transition_adjust(decisions, n)
    assert adj.shape == (n + 1,)
    assert adj[100] > 0                        # confirmed: switching gets cheaper there
    assert adj[200] < 0                        # rejected: switching gets more expensive
    assert (np.delete(adj, [100, 200]) == 0).all()  # nowhere else


def _tiny_hsmm(switch_at, n=120, dmax=60, contrast=0.6):
    """Emissions that favour H before switch_at and A after it, by `contrast` nats."""
    emis = np.zeros((n, 3))
    emis[:, D] = -8.0
    emis[:switch_at, H], emis[:switch_at, A] = 0.0, -contrast
    emis[switch_at:, H], emis[switch_at:, A] = -contrast, 0.0
    hist = np.ones((3, dmax + 1))
    hist[:, 0] = 0.0
    with np.errstate(divide="ignore"):
        log_dur = np.log(hist / hist.sum(axis=1, keepdims=True))  # log_dur[:, 0] unused
    log_trans = np.log(np.full((3, 3), 1e-3) + np.eye(3))
    return emis, log_dur, log_trans


def test_hsmm_default_decode_is_unchanged_by_the_new_argument():
    emis, log_dur, log_trans = _tiny_hsmm(60)
    blocks = [(0, len(emis))]
    plain = hsmm.decode(emis, blocks, log_dur, log_trans, len(emis))
    zeros = hsmm.decode(emis, blocks, log_dur, log_trans, len(emis),
                        trans_adjust=np.zeros(len(emis) + 1))
    assert np.array_equal(plain, zeros)


def test_hsmm_episode_evidence_moves_the_switch():
    emis, log_dur, log_trans = _tiny_hsmm(60, contrast=0.05)  # nearly tied: costs decide
    blocks = [(0, len(emis))]
    adj = np.zeros(len(emis) + 1)
    adj[60] = -50.0
    blocked = hsmm.decode(emis, blocks, log_dur, log_trans, len(emis), trans_adjust=adj)
    assert (blocked == H).all()                # rejected episode: possession stays home
    adj[60] = 50.0
    switched = hsmm.decode(emis, blocks, log_dur, log_trans, len(emis), trans_adjust=adj)
    assert (switched[:60] == H).all() and (switched[60:] == A).all()


def test_hsmm_adjust_leaves_dead_ball_transitions_alone():
    # trans_adjust is evidence about TEAM-TO-TEAM switches only: a NEGATIVE adjustment
    # at an H->D boundary (a rejected episode there) must change nothing, because the
    # D-entering path never receives it and every alternative only gets worse.
    n = 120
    emis = np.zeros((n, 3))
    emis[:60, H], emis[:60, A], emis[:60, D] = 0.0, -0.5, -0.5
    emis[60:, H], emis[60:, A], emis[60:, D] = -0.5, -0.5, 0.0
    hist = np.ones((3, 61))
    hist[:, 0] = 0.0
    with np.errstate(divide="ignore"):
        log_dur = np.log(hist / hist.sum(axis=1, keepdims=True))
    log_trans = np.log(np.full((3, 3), 1e-3) + np.eye(3))
    adj = np.zeros(n + 1)
    adj[60] = -50.0
    plain = hsmm.decode(emis, [(0, n)], log_dur, log_trans, n)
    with_adj = hsmm.decode(emis, [(0, n)], log_dur, log_trans, n, trans_adjust=adj)
    assert np.array_equal(plain, with_adj)
    assert (plain[:60] == H).all() and (plain[60:] == D).all()


def test_annotation_worksheet_roundtrip_and_validation(tmp_path):
    ball, players, teams, near = scenario()
    cands = acquire.propose(near, near)[:3]
    path = str(tmp_path / "game1.csv")
    write_worksheet(path, "game1", cands)
    lines = open(path).read().splitlines()
    assert lines[0].startswith("game,candidate_id")
    # Unanswered worksheets are refused: blank acquired fails loudly.
    try:
        read_annotations(path)
        assert False, "a half-filled worksheet must not read as a reference"
    except ValueError:
        pass
    filled = []
    for line in lines[1:]:
        parts = line.split(",")
        parts[5] = "1" if (parts[3] == "A" and parts[1] == "0") else "0"  # the take only
        filled.append(",".join(parts))
    open(path, "w").write("\n".join([lines[0]] + filled) + "\n")
    answers = read_annotations(path)
    pairs, missing, extra = match(cands, answers)
    assert len(pairs) == 3 and not missing and not extra
    p = np.array([0.9, 0.2, 0.1])[:len(pairs)]
    s = summarize([a["acquired"] for _, a in pairs], p,
                  [a["uncertain"] for _, a in pairs])
    assert s["n"] == 3 and s["n_uncertain"] == 0 and s["recall"] == 1.0


def test_uncertain_annotations_are_excluded_not_negative():
    y = [1, 0, 1]
    p = [0.9, 0.8, 0.9]
    s = summarize(y, p, uncertain=[0, 0, 1])
    assert s["n"] == 2 and s["n_uncertain"] == 1
    assert s["precision"] == 0.5               # the certain rows are 1 correct, 1 wrong


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print("{} ok".format(name))
