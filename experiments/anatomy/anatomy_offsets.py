"""Boundary timing of matched spells on the EXTERNAL test games: predicted minus true, frames.
Also: spellF1 if the matched spells had perfect boundaries (detection-only ceiling).

    python anatomy_offsets.py <repo> <dump_dir>
"""

import os
import sys
from collections import defaultdict

import numpy as np

repo, dump_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from mtah.scoring import pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3
off = defaultdict(list)
snap = defaultdict(list)


def kind(codes, i):
    """What is on the other side of boundary index i (frame before a start / at a stop)."""
    if i < 0 or i >= len(codes) or codes[i] in (D, X):
        return "D"
    return "T"


for fn in sorted(os.listdir(dump_dir)):
    if not (fn.startswith("anatomy_") and fn.endswith(".npz")):
        continue
    z = np.load(os.path.join(dump_dir, fn))
    direction = fn[len("anatomy_"):-4]
    for key in [k[2:] for k in z.files if k.startswith("y|")]:
        game, tag = key.split("|")
        y, p = z["y|" + key], z["pred|" + key]
        runs = pm.valid_runs(y)
        gt_sp = pm.possession_spells(y, runs)
        pr_sp = pm.possession_spells(p, runs)
        ov = pm.spell_overlaps(gt_sp, pr_sp)
        g, q = set(), set()
        fixed = p.copy()
        cell = (direction, tag)
        for iou, i, j in ov:
            if iou < 0.3:
                break
            if i in g or j in q:
                continue
            g.add(i)
            q.add(j)
            (a, b, k), (c, d, _) = gt_sp[i], pr_sp[j]
            off[(cell, "start", kind(y, a - 1))].append(c - a)
            off[(cell, "end", kind(y, b))].append(d - b)
        # detection-only ceiling: count matches at IoU 0.3 as matches at every threshold
        tp = len(g)
        prec, rec = tp / max(1, len(pr_sp)), tp / max(1, len(gt_sp))
        snap[cell].append(2 * prec * rec / max(1e-9, prec + rec))

print("boundary offsets of spells matched at IoU>=0.3, predicted - true, in frames (25 = 1 s)")
print("{:<26} {:<6} {:<3} {:>6} {:>7} {:>7} {:>7} {:>7} {:>9} {:>9}".format(
    "cell", "edge", "to", "n", "median", "mean", "p25", "p75", "|x|<=12", "|x|<=25"))
for key in sorted(off):
    cell, edge, k = key
    v = np.array(off[key])
    print("{:<26} {:<6} {:<3} {:>6} {:>7.1f} {:>7.1f} {:>7.1f} {:>7.1f} {:>9.2f} {:>9.2f}".format(
        cell[0] + " " + cell[1], edge, k, len(v), np.median(v), v.mean(), np.percentile(v, 25),
        np.percentile(v, 75), np.mean(np.abs(v) <= 12), np.mean(np.abs(v) <= 25)))
print("\ndetection-only ceiling (every IoU>=0.3 match counted at all thresholds):")
for cell in sorted(snap):
    print("  {:<26} {:.3f}".format(cell[0] + " " + cell[1], np.mean(snap[cell])))
