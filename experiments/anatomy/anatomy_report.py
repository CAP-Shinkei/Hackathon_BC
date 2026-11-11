"""Where does the candidate lose spellF1 on the EXTERNAL test games? Reads anatomy_*.npz.

    python anatomy_report.py <repo> <dump_dir>
"""

import os
import sys
from collections import defaultdict

import numpy as np

repo, dump_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from mtah.scoring import pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3


def spell_f1(y, p):
    r = pm.spell_scores(y, pm.fill_abstentions(p, pm.valid_runs(y)), pm.valid_runs(y))
    return r


def fill_team(team, live):
    """team: array of H/A or -1 (unknown) ; inside each True run of `live`, forward- then
    back-fill the unknowns; a run with no known value gets H."""
    out = team.copy()
    edges = np.flatnonzero(np.diff(np.concatenate(([0], live.astype(np.int8), [0]))))
    for a, b in zip(edges[0::2], edges[1::2]):
        w = out[a:b]
        known = np.flatnonzero(w >= 0)
        if not len(known):
            w[:] = H
            continue
        carry = np.maximum.accumulate(np.where(w >= 0, np.arange(len(w)), -1))
        carry[carry < 0] = known[0]
        out[a:b] = w[carry]
    return out


def oracle_dead(y, p):
    """True dead/alive mask, predicted team inside it."""
    live = np.isin(y, (H, A))
    team = np.where(np.isin(p, (H, A)), p, -1).astype(np.int8)
    out = y.copy()
    out[live] = fill_team(team, live)[live]
    return out


def oracle_team(y, p):
    """Predicted dead/alive mask, true team inside it."""
    live = np.isin(p, (H, A))
    team = np.where(np.isin(y, (H, A)), y, -1).astype(np.int8)
    out = p.copy()
    graded = y != X
    out[live & graded] = fill_team(team, live & graded)[live & graded]
    return out


def classify(spells, other, idx_matched, seen):
    """For each unmatched spell (start, stop, team) the dominant label of `other` over it."""
    out = defaultdict(int)
    for i, (a, b, k) in enumerate(spells):
        if idx_matched[i]:
            out["matched"] += 1
            continue
        w = other[a:b]
        same, opp, dead = np.mean(w == k), np.mean(w == 1 - k), np.mean(w == D)
        if opp >= max(same, dead):
            out["other_team"] += 1
        elif dead >= max(same, opp):
            out["dead"] += 1
        else:
            out["same_team_bad_overlap"] += 1
    return out


def matched_flags(gt_sp, pr_sp, thr):
    ov = pm.spell_overlaps(gt_sp, pr_sp)
    g = np.zeros(len(gt_sp), bool)
    p = np.zeros(len(pr_sp), bool)
    for iou, i, j in ov:
        if iou < thr:
            break
        if g[i] or p[j]:
            continue
        g[i] = p[j] = True
    return g, p


rows = defaultdict(list)
dur_bins = [(0, 3), (3, 6), (6, 12), (12, 25), (25, 1e9)]
hid_bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
by_dur = defaultdict(lambda: [0, 0])
by_hid = defaultdict(lambda: [0, 0])
by_start = defaultdict(lambda: [0, 0])
pred_dur = defaultdict(lambda: [0, 0])
miss_kind = defaultdict(lambda: defaultdict(int))
spur_kind = defaultdict(lambda: defaultdict(int))

for fn in sorted(os.listdir(dump_dir)):
    if not (fn.startswith("anatomy_") and fn.endswith(".npz")):
        continue
    z = np.load(os.path.join(dump_dir, fn))
    direction = fn[len("anatomy_"):-4]
    for key in [k[2:] for k in z.files if k.startswith("y|")]:
        game, tag = key.split("|")
        y, p, seen, base = z["y|" + key], z["pred|" + key], z["seen|" + key], z["baseline|" + key]
        proba = z["proba|" + key]
        runs = pm.valid_runs(y)
        cell = (direction, tag)
        r = spell_f1(y, p)
        graded = y != X
        live = np.isin(y, (H, A))
        argmax = np.full(len(y), X, np.int8)
        argmax[graded] = proba[graded].argmax(axis=1)
        both = live & np.isin(p, (H, A))
        rows[cell].append(dict(
            f1=r["mean_f1"], f1_03=r["per_iou"][0.3]["f1"], f1_05=r["per_iou"][0.5]["f1"],
            f1_07=r["per_iou"][0.7]["f1"], prec05=r["per_iou"][0.5]["precision"],
            rec05=r["per_iou"][0.5]["recall"], n_gt=r["n_gt"], n_pred=r["n_pred"],
            or_dead=spell_f1(y, oracle_dead(y, p))["mean_f1"],
            or_team=spell_f1(y, oracle_team(y, p))["mean_f1"],
            base=spell_f1(y, base)["mean_f1"],
            frame_acc_argmax=float(np.mean(argmax[graded] == y[graded])),
            team_acc_dec=float(np.mean(p[both] == y[both])),
            team_acc_argmax_seen=float(np.mean((proba[:, H] > proba[:, A])[live & seen] == (y[live & seen] == H))),
            team_acc_argmax_hidden=float(np.mean((proba[:, H] > proba[:, A])[live & ~seen] == (y[live & ~seen] == H))),
            hidden_live=float(np.mean(~seen[live])),
        ))
        gt_sp = pm.possession_spells(y, runs)
        pr_sp = pm.possession_spells(pm.fill_abstentions(p, runs), runs)
        g, q = matched_flags(gt_sp, pr_sp, 0.5)
        for k_, v in classify(gt_sp, p, g, seen).items():
            miss_kind[cell][k_] += v
        for k_, v in classify(pr_sp, y, q, seen).items():
            spur_kind[cell][k_] += v
        for i, (a, b, k) in enumerate(gt_sp):
            L = (b - a) / 25.0
            for lo, hi in dur_bins:
                if lo <= L < hi:
                    by_dur[(cell, lo)][0] += g[i]
                    by_dur[(cell, lo)][1] += 1
            hshare = float(np.mean(~seen[a:b]))
            for lo, hi in hid_bins:
                if lo <= hshare < hi:
                    by_hid[(cell, lo)][0] += g[i]
                    by_hid[(cell, lo)][1] += 1
            prev = y[a - 1] if a > 0 else X
            nxt = y[b] if b < len(y) else X
            kind = ("after_stoppage" if prev in (D, X) else "after_turnover") + "/" + \
                   ("ends_stoppage" if nxt in (D, X) else "ends_turnover")
            by_start[(cell, kind)][0] += g[i]
            by_start[(cell, kind)][1] += 1
        for j, (a, b, k) in enumerate(pr_sp):
            L = (b - a) / 25.0
            for lo, hi in dur_bins:
                if lo <= L < hi:
                    pred_dur[(cell, lo)][0] += q[j]
                    pred_dur[(cell, lo)][1] += 1

cells = sorted(rows)
print("== headline per cell (mean over games) ==")
keys = ["f1", "f1_03", "f1_05", "f1_07", "prec05", "rec05", "n_gt", "n_pred", "or_dead", "or_team",
        "base", "frame_acc_argmax", "team_acc_dec", "team_acc_argmax_seen",
        "team_acc_argmax_hidden", "hidden_live"]
print("{:<30}".format("cell") + " ".join("{:>9}".format(k[:9]) for k in keys))
for c in cells:
    print("{:<30}".format(c[0] + " " + c[1]) + " ".join(
        "{:9.3f}".format(np.mean([r[k] for r in rows[c]])) for k in keys))

print("\n== unmatched TRUE spells at IoU 0.5: what the prediction says over them (share of all true spells) ==")
for c in cells:
    t = sum(miss_kind[c].values())
    print("{:<30}".format(c[0] + " " + c[1]) + "  ".join(
        "{}={:.3f}".format(k, miss_kind[c][k] / t) for k in ("matched", "other_team", "dead", "same_team_bad_overlap")))
print("\n== unmatched PREDICTED spells at IoU 0.5: what the truth says over them (share of all predicted spells) ==")
for c in cells:
    t = sum(spur_kind[c].values())
    print("{:<30}".format(c[0] + " " + c[1]) + "  ".join(
        "{}={:.3f}".format(k, spur_kind[c][k] / t) for k in ("matched", "other_team", "dead", "same_team_bad_overlap")))

print("\n== recall at IoU 0.5 by TRUE spell duration (s): recall (share of true spells) ==")
for c in cells:
    tot = sum(by_dur[(c, lo)][1] for lo, _ in dur_bins)
    print("{:<30}".format(c[0] + " " + c[1]) + "  ".join(
        "{}+:{:.2f}({:.2f})".format(int(lo), by_dur[(c, lo)][0] / max(1, by_dur[(c, lo)][1]),
                                    by_dur[(c, lo)][1] / tot) for lo, _ in dur_bins))
print("\n== precision at IoU 0.5 by PREDICTED spell duration (s): precision (share of predicted spells) ==")
for c in cells:
    tot = sum(pred_dur[(c, lo)][1] for lo, _ in dur_bins)
    print("{:<30}".format(c[0] + " " + c[1]) + "  ".join(
        "{}+:{:.2f}({:.2f})".format(int(lo), pred_dur[(c, lo)][0] / max(1, pred_dur[(c, lo)][1]),
                                    pred_dur[(c, lo)][1] / tot) for lo, _ in dur_bins))
print("\n== recall at IoU 0.5 by share of the true spell with the ball hidden: recall (share of spells) ==")
for c in cells:
    tot = sum(by_hid[(c, lo)][1] for lo, _ in hid_bins)
    print("{:<30}".format(c[0] + " " + c[1]) + "  ".join(
        "{:.1f}+:{:.2f}({:.2f})".format(lo, by_hid[(c, lo)][0] / max(1, by_hid[(c, lo)][1]),
                                        by_hid[(c, lo)][1] / tot) for lo, _ in hid_bins))
print("\n== recall at IoU 0.5 by how the true spell starts/ends: recall (share of spells) ==")
kinds = ["after_stoppage/ends_stoppage", "after_stoppage/ends_turnover",
         "after_turnover/ends_stoppage", "after_turnover/ends_turnover"]
for c in cells:
    tot = sum(by_start[(c, k)][1] for k in kinds)
    print("{:<30}".format(c[0] + " " + c[1]) + "  ".join(
        "{}:{:.2f}({:.2f})".format(k.replace("after_", "").replace("ends_", "").replace("stoppage", "D").replace("turnover", "T"),
                                   by_start[(c, k)][0] / max(1, by_start[(c, k)][1]),
                                   by_start[(c, k)][1] / tot) for k in kinds))
