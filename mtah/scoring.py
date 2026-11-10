"""The only scorer in this repository: a thin call into the official metric library.

Nothing here re-implements a metric. `official/evaluation/possession_metrics.py` decides every
number, so a score measured on an external game means what the leaderboard means.
"""
import os
import sys

import numpy as np

from mtah.schema import X

OFFICIAL_EVALUATION = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "official", "evaluation")
if OFFICIAL_EVALUATION not in sys.path:
    sys.path.insert(0, OFFICIAL_EVALUATION)

import possession_metrics as pm  # noqa: E402  (needs the path above)


def grade_mask(labels, blocks):
    """Ground truth as the scorer wants it: real labels inside the halves, X everywhere else."""
    graded = np.full(len(labels), X, dtype=np.int8)
    for start, stop in blocks:
        graded[start:stop] = labels[start:stop]
    return graded


def score(gt_codes, pred_codes, blocks=None):
    """{'spell_f1', 'macro_f1', 'spell_f1_per_iou', 'n_gt_spells', 'n_pred_spells', 'd_f1'}."""
    gt = np.asarray(gt_codes, dtype=np.int8)
    if blocks is not None:
        gt = grade_mask(gt, blocks)
    result = pm.evaluate(gt, np.asarray(pred_codes, dtype=np.int8))
    return {
        "spell_f1": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "spell_f1_per_iou": {iou: row["f1"] for iou, row in result["spell_f1"].items()},
        "n_gt_spells": result["n_gt_spells"],
        "n_pred_spells": result["n_pred_spells"],
        "d_f1": result["per_class"]["D"]["f1"],
    }
