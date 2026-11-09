"""Scores through the official metric code, imported from a clone of the challenge repository.

Set MTAH_OFFICIAL_REPO to the clone (default: ../official_repo next to this repository). The
official code is used unchanged, so a score here is the score run_evaluation.py would print.
"""

import os
import sys

DEFAULT_OFFICIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "official_repo")
OFFICIAL = os.environ.get("MTAH_OFFICIAL_REPO", DEFAULT_OFFICIAL)
sys.path.insert(0, os.path.join(OFFICIAL, "evaluation"))
import possession_metrics as pm  # noqa: E402

HEADLINE = ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells",
            "mean_spell_duration_s", "gt_mean_spell_duration_s", "alive_dead_f1",
            "team_acc_both_alive")


def score(gt_codes, pred_codes):
    """Headline metrics as a dict (spell_f1_mean ranks, macro_f1 breaks ties)."""
    result = pm.evaluate(gt_codes, pred_codes)
    out = {k: result[k] for k in HEADLINE}
    out["d_f1"] = result["per_class"]["D"]["f1"]
    return out
