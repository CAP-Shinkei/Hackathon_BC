"""E12: how much does v2 + fill + HSMM rely on the ball's visibility pattern, and does it survive
real ball gaps?

Feature sets (same classifier, fill, HSMM and selection as the submission):
  full           every feature
  no_seen_share  without the rolling "ball seen" shares (0.5 s, 2 s, 5 s)
  no_visibility  also without "ball seen on this frame" and the gap-shape clues of the fill
                 (time since / until the ball is seen, gap length)
The dropped clues are the ones whose meaning comes from the gap simulator.

Test sets:
  sim cross      fit on one dataset (Metrica or DFL, simulated gaps), score on the other (E09's
                 test sets: 35% / 50% live missing, seeds 7 / 8)
  sc_real        fit on Metrica + DFL (as the submission), score on the 20 SkillCorner matches
                 with their real gaps; labels are SkillCorner's own automatic output
  sc_real+sim35  the same SkillCorner matches, further damaged by the simulator at 35% live missing

Controlled-possession rule 2 s everywhere. The baseline (+/-2 s vote) is scored for reference.

    python experiments/e12_visibility_ablation.py
"""

import numpy as np

from common import dfl_games, external_games, skillcorner_games, table
from controlled import controlled
from gapsim import Params, damage
from mtah import ballfill, features, v1, v2
from mtah.evidence import nearest_team
from mtah.labels import graded_blocks
from mtah.scoring import score
from mtah.smooth import majority_vote

MIN_S = 2.0
TEST_RATES = (0.35, 0.5)
DROP = {
    "full": (),
    "no_seen_share": ("ball_seen_share_13", "ball_seen_share_50", "ball_seen_share_125"),
    "no_visibility": ("ball_seen_share_13", "ball_seen_share_50", "ball_seen_share_125",
                      "ball_seen", "gap_since_s", "gap_until_s", "gap_len_s"),
}


def featurise(name, ball, players, teams, pitch, gt, tag):
    """Full feature matrix (v2 + straight-line fill) and what scoring needs; the tracking
    arrays are not kept, to bound memory."""
    blocks = graded_blocks(gt)
    near = nearest_team(ball, players, teams, pitch)
    Xg, names = features.compute(ball, players, teams, pitch, near)
    filled, gap = ballfill.fill(ball, players, pitch, blocks)
    near_f = nearest_team(filled, players, teams, pitch)
    Xf, nf = features.compute_filled(filled, gap, players, teams, pitch, near_f)
    return dict(game=name, tag=tag, X=np.hstack([Xg, Xf]), names=names + nf,
                y=controlled(gt, MIN_S), baseline=majority_vote(near, 50))


def damaged(games, rates, seed0):
    out = []
    for g in games:
        for i, r in enumerate(rates):
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(live_missing=r), seed=seed0 + i)
            out.append(featurise(g["name"], ball, players, g["teams"], g["pitch"], g["gt"],
                                 "sim{:.0%}".format(r)))
    return out


def select(d, variant):
    keep = [i for i, k in enumerate(d["names"]) if k not in DROP[variant]]
    return d["X"][:, keep], [d["names"][i] for i in keep]


def fit(train, variant):
    items = []
    for d in train:
        Xg, nm = select(d, variant)
        items.append((d["game"], Xg, d["y"], nm))
    model = v2.fit(items)
    del items
    v2.add_hsmm(model, [d["y"] for d in train])
    return model


def evaluate(model, tests, variant):
    acc = {}
    for d in tests:
        Xg, nm = select(d, variant)
        pred = v2.predict(model, Xg, nm, graded_blocks(d["y"]), len(d["y"]))
        s = score(d["y"], pred)
        b = score(d["y"], d["baseline"])
        graded = d["y"] != 3
        acc.setdefault(d["tag"], []).append(
            (s["spell_f1_mean"], b["spell_f1_mean"], s["n_pred_spells"], s["n_gt_spells"],
             float(np.mean(pred[graded] == 2)), float(np.mean(d["y"][graded] == 2)),
             s["d_f1"], s["team_acc_both_alive"]))
    return acc


def main():
    data = {"metrica": external_games(), "dfl": dfl_games()}
    train = {k: damaged(v, v1.GAP_RATES, 100) for k, v in data.items()}
    test = {k: damaged(v, TEST_RATES, 7) for k, v in data.items()}
    del data
    sc = []
    for g in skillcorner_games():
        sc.append(featurise(g["name"], g["ball"], g["players"], g["teams"], g["pitch"], g["gt"],
                            "sc_real"))
        ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                               Params(live_missing=0.35), seed=7)
        sc.append(featurise(g["name"], ball, players, g["teams"], g["pitch"], g["gt"],
                            "sc_real+sim35"))
    print("features ready", flush=True)

    rows = []
    for variant in DROP:
        runs = (("dfl", "metrica", train["dfl"], test["metrica"]),
                ("metrica", "dfl", train["metrica"], test["dfl"]),
                ("metrica+dfl", "skillcorner", train["metrica"] + train["dfl"], sc))
        for src, dst, tr, te in runs:
            model = fit(tr, variant)
            print("{} {}->{}: hsmm temp {}  oof {:.4f}".format(
                variant, src, dst, model["hsmm_temperature"], model["oof_spell_f1_hsmm"]),
                flush=True)
            for tag, v in evaluate(model, te, variant).items():
                v = np.array(v)
                rows.append(dict(variant=variant, train=src, test=dst, gaps=tag,
                                 spellF1=float(v[:, 0].mean()), baseline=float(v[:, 1].mean()),
                                 pred_spells=int(v[:, 2].mean()), gt_spells=int(v[:, 3].mean()),
                                 pred_D=float(v[:, 4].mean()), gt_D=float(v[:, 5].mean()),
                                 d_f1=float(v[:, 6].mean()), team_acc=float(v[:, 7].mean())))
            del model
    print()
    print(table(rows, list(rows[0].keys())))


if __name__ == "__main__":
    main()
