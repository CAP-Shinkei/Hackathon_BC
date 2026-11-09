"""E01: the official baseline method (nearest player + majority vote) on the clean external games.

    python experiments/e01_baseline_clean.py
"""

import numpy as np

from common import external_games, table
from mtah.evidence import nearest_team
from mtah.scoring import score
from mtah.smooth import majority_vote

WINDOWS = (0, 25, 50, 125)


def main():
    rows = []
    for g in external_games():
        near = nearest_team(g["ball"], g["players"], g["teams"], g["pitch"])
        for w in WINDOWS:
            s = score(g["gt"], majority_vote(near, w))
            rows.append(dict(game=g["name"], window=w, **s))
    keys = ["game", "window", "spell_f1_mean", "macro_f1", "d_f1", "n_gt_spells",
            "n_pred_spells", "mean_spell_duration_s", "gt_mean_spell_duration_s"]
    print(table(rows, keys))
    print("\nmean over games")
    for w in WINDOWS:
        sel = [r for r in rows if r["window"] == w]
        print("  window {:>3}: spellF1 {:.4f}  macroF1 {:.4f}".format(
            w, np.mean([r["spell_f1_mean"] for r in sel]), np.mean([r["macro_f1"] for r in sel])))


if __name__ == "__main__":
    main()
