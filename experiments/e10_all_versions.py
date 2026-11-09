"""E10: the versions that need no classifier, on exactly E09's test sets.

  baseline     official baseline, nearest player + majority vote +/-2 s
  alvaro_v02   Álvaro's v0.2 with his own constants (hard emission table, his switch rates,
               81-frame merge), WITHOUT his Smart Tagging bonus: the external games have no
               Smart Tagging
  alvaro_v03   Álvaro's v0.3 with his own constants (soft emission table, his gamma durations,
               transitions and segment penalty), also without Smart Tagging
  v1           Álvaro's v0.2 design refitted on the training dataset (tables counted,
               temperature and minimum spell chosen there)

Test sets: E07/E09's (damaged with seeds 7, 8 at 35% / 50% live missing), controlled rule 2 s.

    python experiments/e10_all_versions.py
"""

import numpy as np

from common import dfl_games, external_games, table
from controlled import controlled
from gapsim import Params, damage
from mtah import hsmm, v1
from mtah.evidence import nearest_team
from mtah.hmm import _viterbi, merge_short
from mtah.labels import X, graded_blocks
from mtah.scoring import score
from mtah.smooth import majority_vote

MIN_S = 2.0
TEST_RATES = (0.35, 0.5)


def observations(games, rates, seed0):
    out = []
    for g in games:
        for i, r in enumerate(rates):
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(live_missing=r), seed=seed0 + i)
            out.append((g, r, nearest_team(ball, players, g["teams"], g["pitch"])))
    return out


def obs_codes(near):
    """Álvaro's observation: the baseline_raw label, nearest team or D where unseen."""
    return np.where(near < 0, 2, near).astype(np.int64)


def alvaro_v02(near, blocks, n):
    E = np.log(np.array([[0.70, 0.05, 0.25], [0.05, 0.70, 0.25], [0.10, 0.10, 0.80]]))
    T_SWITCH, T_DEAD_IN, T_DEAD_OUT, MIN_FRAMES = 2e-4, 0.002, 0.01, 81
    log_t = np.log(np.array([[1 - T_SWITCH - T_DEAD_IN, T_SWITCH, T_DEAD_IN],
                             [T_SWITCH, 1 - T_SWITCH - T_DEAD_IN, T_DEAD_IN],
                             [T_DEAD_OUT, T_DEAD_OUT, 1 - 2 * T_DEAD_OUT]]))
    log_p0 = np.log(np.array([0.45, 0.45, 0.10]))
    emis = E[:, obs_codes(near)].T
    out = np.full(n, X, dtype=np.int8)
    for a, b in blocks:
        out[a:b] = _viterbi(np.ascontiguousarray(emis[a:b]), log_t, log_p0)
    return merge_short(out, blocks, MIN_FRAMES)


def _gamma(shape, scale, dmax, floor, width):
    d = np.arange(floor, dmax + 1, dtype=float)
    lp = (shape - 1) * np.log(d) - d / scale
    lp -= lp.max() + np.log(np.exp(lp - lp.max()).sum())
    out = np.full(width, -np.inf)
    out[floor:dmax + 1] = lp
    return out


def alvaro_v03(near, blocks, n):
    E3 = np.log(np.array([[0.40, 0.15, 0.45], [0.15, 0.40, 0.45], [0.20, 0.20, 0.60]]))
    width = 2201
    seg_penalty = 2.0
    log_dur = np.vstack([_gamma(12.0, 26.0, 2200, 50, width), _gamma(12.0, 22.0, 2200, 50, width),
                         _gamma(6.0, 45.0, 1800, 25, width)]) - seg_penalty
    with np.errstate(divide="ignore"):
        log_t = np.log(np.array([[0.0, 0.35, 0.65], [0.40, 0.0, 0.60], [0.55, 0.40, 0.05]]))
    log_p0 = np.log(np.array([0.45, 0.40, 0.15]))
    emis = E3[:, obs_codes(near)].T
    out = np.full(n, X, dtype=np.int8)
    for a, b in blocks:
        out[a:b] = hsmm._decode(np.ascontiguousarray(emis[a:b]), log_dur, log_t, log_p0)
    return out


def main():
    data = {"metrica": external_games(), "dfl": dfl_games()}
    rows = []
    for src, dst in (("dfl", "metrica"), ("metrica", "dfl")):
        train = observations(data[src], v1.GAP_RATES, 100)
        m1 = v1.fit([(controlled(g["gt"], MIN_S), near) for g, _, near in train])
        del train
        test = observations(data[dst], TEST_RATES, 7)
        for rate in TEST_RATES:
            acc = {k: [] for k in ("baseline", "alvaro_v02", "alvaro_v03", "v1")}
            spells = {k: [] for k in acc}
            n_gt = []
            for g, r, near in test:
                if r != rate:
                    continue
                gt = controlled(g["gt"], MIN_S)
                blocks = graded_blocks(gt)
                preds = {"baseline": majority_vote(near, 50),
                         "alvaro_v02": alvaro_v02(near, blocks, len(gt)),
                         "alvaro_v03": alvaro_v03(near, blocks, len(gt)),
                         "v1": v1.predict(near, blocks, m1)}
                for k, p in preds.items():
                    s = score(gt, p)
                    acc[k].append(s["spell_f1_mean"])
                    spells[k].append(s["n_pred_spells"])
                n_gt.append(s["n_gt_spells"])
            for k in acc:
                rows.append(dict(train=src, test=dst, version=k, live_missing=rate,
                                 spellF1=float(np.mean(acc[k])),
                                 pred_spells=int(np.mean(spells[k])),
                                 gt_spells=int(np.mean(n_gt))))
        print("{}->{} done".format(src, dst), flush=True)
    print()
    print(table(rows, list(rows[0].keys())))


if __name__ == "__main__":
    main()
