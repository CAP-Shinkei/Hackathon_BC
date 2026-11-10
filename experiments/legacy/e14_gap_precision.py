"""E14: can a ball gap of >= X seconds be a hard dead-ball cut? Measured on EXTERNAL data.

The 31-10 handoff proposed tagging-confirmed ball gaps (>= 2 s) as hard dead-ball boundaries.
No external game has Smart Tagging, so the tag part cannot be validated externally - but the
underlying constant can: on a damaged external game, how often is a long ball gap actually a
stoppage? If P(dead | gap >= X s) is low, the cut splits live possessions and the idea dies.

Metric: damage the Metrica sample Game 1 with gapsim (35% / 50% live missing, seeds 7/8), then
measure, over graded frames inside ball gaps of at least X seconds, the share that is truly dead
(answer key from external/build_gt.py). Base rate P(dead) ~ 0.41.

Result (2026-11-09, this script): see EXPERIMENTS.md E14. gap >= 2 s: 0.51-0.66 - barely above
the base rate, UNSAFE as a hard cut. gap >= 20 s: 0.81-0.97 - still not clean. Decision: REJECT
gap-based hard cuts (tagged or not). Note the measurement is optimistic for the competition: in
the sample data the ball is hidden during most dead time, while broadcast tracking (SkillCorner,
E11) still sees it in ~21% of dead frames, which lowers real precision further.

    python experiments/e14_gap_precision.py   # needs the Metrica sample data + build_gt keys
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external"))
sys.path.insert(0, os.path.join(HERE, ".."))

import metrica_sample as ms  # noqa: E402
from gapsim import Params, damage  # noqa: E402
from mtah.labels import read_labels, graded_blocks  # noqa: E402

FPS = 25


def gap_runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def main():
    gt = read_labels(os.path.join(
        os.environ.get("MTAH_LABELS", os.path.join(HERE, "..", "..", "external_data",
                                                   "labels", "metrica_sample")),
        "Sample_Game_1_gt.csv"))
    ball, players, teams = ms.load_tracking(1)
    blocks = graded_blocks(gt)
    graded = np.zeros(len(gt), bool)
    for a, b in blocks:
        graded[a:b] = True
    print("game 1: {} frames, base P(dead) on graded = {:.3f}".format(
        len(gt), float((gt[graded] == 2).mean())))
    for rate in (0.35, 0.5):
        for seed in (7, 8):
            b2, _ = damage(ball, players, teams, gt, ms.PITCH,
                           Params(live_missing=rate), seed=seed)
            missing = np.isnan(b2[:, 0])
            line = []
            for xs in (1, 2, 5, 10, 20):
                tot = dead = 0
                for a, b in gap_runs(missing):
                    if b - a >= xs * FPS and graded[a:b].any():
                        g = gt[a:b][graded[a:b]]
                        tot += len(g)
                        dead += int((g == 2).sum())
                line.append(">={:2d}s {:.3f}".format(xs, dead / tot if tot else float("nan")))
            print("rate {:.2f} seed {}: ".format(rate, seed) + "  ".join(line))


if __name__ == "__main__":
    main()
