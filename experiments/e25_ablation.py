"""E25, ablation of the context features: which groups of clues carry the gain, and does any of
it come from clues that read how the simulator hides the ball?

One direction only, to keep it short: fit on Metrica (five training regimes), score on the seven
DFL games at 35% and 50% of live frames missing. Each row drops one group from `mix+ctx`, or keeps
only one group on top of the base features.

    python experiments/e25_ablation.py
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import e25_benchmark as e25
from common import dfl_games, external_games, table

GROUPS = {
    "near": ("f_near_home_share_past", "f_near_home_share_future"),   # who was / will be nearest
    "seen": ("ball_seen_share_past", "ball_seen_share_future"),       # visibility: simulator-made
    "pace": ("speed_all_", "still_share_"),                           # everyone slowing down
    "dmin": ("dmin_",),                                               # closest approach +/- 1 s
    "shape": ("fast_n_", "length_", "width_", "spread_rate_"),        # team shape
    "sight": ("d_last_", "d_next_"),                                  # teams vs last/next sighting
}


def group_of(name):
    for g, prefixes in GROUPS.items():
        if name.startswith(prefixes):
            return g
    raise ValueError(name)


def main():
    pool = ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn"))
    train = e25.damaged(external_games(), [(t, f, 100 + i) for i, (t, f) in enumerate(e25.MIX)])
    test = e25.damaged(dfl_games(), e25.TESTS[:2])
    family = train[0]["family"]
    groups = [group_of(k) if f == "ctx" else None for k, f in zip(train[0]["names"], family)]
    configs = [("mix (no context)", ()), ("mix+ctx (all)", tuple(GROUPS))]
    configs += [("without " + g, tuple(h for h in GROUPS if h != g)) for g in GROUPS]
    configs += [("only " + g, (g,)) for g in GROUPS]
    rows = []
    for label, keep in configs:
        relabelled = ["base" if f == "base" else ("ctx" if g in keep else "off")
                      for f, g in zip(family, groups)]
        for d in train + test:
            d["family"] = relabelled
        model = e25.fit(train, ("base", "ctx"), pool)
        scores = [s["spell_f1_mean"] for *_, s, _ in e25.evaluate(model, test, ("base", "ctx"),
                                                                  pool)]
        tags = [d["tag"] for d in test]
        row = dict(config=label, columns=relabelled.count("base") + relabelled.count("ctx"),
                   oof=model["oof_f1"],
                   **{t: float(np.mean([s for s, u in zip(scores, tags) if u == t]))
                      for t in ("std35", "std50")})
        row["mean"] = 0.5 * (row["std35"] + row["std50"])
        rows.append(row)
        print("{config}: {columns} columns, oof {oof:.4f}, std35 {std35:.4f}, std50 {std50:.4f}"
              .format(**row), flush=True)
    pool.shutdown()
    print()
    print(table(rows, list(rows[0].keys())))


if __name__ == "__main__":
    main()
