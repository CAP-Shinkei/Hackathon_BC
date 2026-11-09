"""Regenerate our submission CSV for the competition match.

    python make_submission.py                          # v1 (out/submissions/v1.csv)
    python make_submission.py --model v2 --min-s 2     # v2 (out/submissions/v2_min2s.csv)
    python make_submission.py --model v2 --min-s 2 --fill --decoder hsmm
                                                       # out/submissions/v2_min2s_fill_hsmm.csv
    python make_submission.py --model v2 --min-s 2 --fill --decoder hsmm --drop-outside
                                                       # ..._hsmm_rule.csv (E13)

Needs the Metrica sample data (external_data/metrica_sample_data, or METRICA_SAMPLE_DIR), its
answer keys (python external/build_gt.py), the competition data (MTAH_DATA_DIR) and a clone of
the official repository for the scorer (MTAH_OFFICIAL_REPO).

Prints aggregates only: the competition data must not be sent to hosted AI services, and per-frame
output is never printed.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "external"))
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import dfl_games, external_games  # noqa: E402
from controlled import controlled  # noqa: E402
from gapsim import Params, damage  # noqa: E402
from mtah import competition as comp  # noqa: E402
from mtah import ballfill, features, kickoff, v1, v2  # noqa: E402
from mtah.evidence import drop_far_outside, nearest_team  # noqa: E402
from mtah.labels import D, H, X, graded_blocks, runs, write_labels  # noqa: E402

PUBLISHED_KICKOFF_2 = 113742  # FAQ.md; used only as a pipeline check, never as an input


def second_half_kickoff(ball, instances):
    lo, hi = kickoff.tagged_windows(instances)["2nd half"]
    lo, hi = lo - 250, hi + 250  # the tag brackets the restart by a few seconds
    found = kickoff.detect(ball, comp.PITCH, lo, hi)
    return (found, "detected") if found is not None else (lo + 250, "tag start (fallback)")


def v2_features(ball, players, teams, pitch, blocks, fill):
    """v2's feature matrix, plus the straight-line ball-fill features when fill is set."""
    near = nearest_team(ball, players, teams, pitch)
    Xg, names = features.compute(ball, players, teams, pitch, near)
    if fill:
        filled, gap = ballfill.fill(ball, players, pitch, blocks)
        near_f = nearest_team(filled, players, teams, pitch)
        Xf, nf = features.compute_filled(filled, gap, players, teams, pitch, near_f)
        Xg, names = np.hstack([Xg, Xf]), names + nf
    return Xg, names


def v2_training_items(min_s, fill=False, rule=False):
    """[(game_id, X, y, names)]: every external game damaged at each training gap rate."""
    items = []
    for g in external_games() + dfl_games():
        for i, rate in enumerate(v1.GAP_RATES):
            ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                   Params(live_missing=rate), seed=100 + i)
            if rule:
                ball = drop_far_outside(ball, g["pitch"])
            Xg, names = v2_features(ball, players, g["teams"], g["pitch"],
                                    graded_blocks(g["gt"]), fill)
            items.append((g["name"], Xg, controlled(g["gt"], min_s), names))
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("v1", "v2"), default="v1")
    parser.add_argument("--min-s", type=float, default=0.0,
                        help="controlled-possession rule for the external labels (v2 only)")
    parser.add_argument("--fill", action="store_true",
                        help="add straight-line ball-fill features (v2 only)")
    parser.add_argument("--drop-outside", action="store_true",
                        help="treat a ball more than 10 m beyond the lines as not seen (E13)")
    parser.add_argument("--decoder", choices=("viterbi", "hsmm"), default="viterbi",
                        help="final decoding step (v2 only)")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.out is None:
        name = "v1" if args.model == "v1" else "v2_min{:g}s{}{}{}".format(
            args.min_s, "_fill" if args.fill else "",
            "_hsmm" if args.decoder == "hsmm" else "",
            "_rule" if args.drop_outside else "")
        args.out = os.path.join(HERE, "out", "submissions", name + ".csv")

    if args.model == "v1":
        print("fitting v1 on the damaged Metrica games ...", flush=True)
        model = v1.fit(v1.damaged_observations(external_games(), damage, Params))
        print("  temperature {}  min_spell {} frames  external spellF1 {:.4f}".format(
            model["temperature"], model["min_spell"], model["external_spell_f1"]))
    else:
        print("fitting v2 on the damaged Metrica + DFL games, controlled rule {} s ...".format(
            args.min_s), flush=True)
        items = v2_training_items(args.min_s, args.fill, args.drop_outside)
        model = v2.fit(items)
        print("  temperature {}  min_spell {} frames  out-of-fold external spellF1 {:.4f}".format(
            model["temperature"], model["min_spell"], model["oof_spell_f1"]))
        if args.decoder == "hsmm":
            v2.add_hsmm(model, [y for _, _, y, _ in items])
            print("  hsmm temperature {}  out-of-fold external spellF1 {:.4f}".format(
                model["hsmm_temperature"], model["oof_spell_f1_hsmm"]))

    print("loading the competition tracking ...", flush=True)
    ball, players, teams = comp.load_tracking()
    if args.drop_outside:
        kept = ball
        ball = drop_far_outside(ball, comp.PITCH)
        print("  far-outside rule: {} ball frames treated as not seen".format(
            int(np.sum(np.isnan(ball[:, 0]) & ~np.isnan(kept[:, 0])))))
    k2, how = second_half_kickoff(ball, comp.load_smart_tagging())
    print("  2nd-half kick-off: frame {} ({}); published {} -> difference {} frames".format(
        k2, how, PUBLISHED_KICKOFF_2, k2 - PUBLISHED_KICKOFF_2))

    (a1, b1), (a2, b2) = comp.GRADED
    blocks = [(a1, b1), (k2, b2)]
    near = nearest_team(ball, players, teams, comp.PITCH)
    if args.model == "v1":
        labels = v1.predict(near, blocks, model)
    else:
        Xc, names = v2_features(ball, players, teams, comp.PITCH, comp.GRADED, args.fill)
        labels = v2.predict(model, Xc, names, blocks, comp.N_FRAMES)
    labels[a2:k2] = D

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_labels(args.out, labels)

    graded = np.zeros(comp.N_FRAMES, bool)
    for a, b in comp.GRADED:
        graded[a:b] = True
    spells = [(c, stop - start) for a, b in comp.GRADED
              for c, start, stop in runs(labels[a:b]) if c in (0, 1)]
    home = sum(n for c, n in spells if c == H)
    print("wrote {}".format(os.path.relpath(args.out, HERE)))
    print("  graded frames: H {:.3f}  A {:.3f}  D {:.3f}  X {}".format(
        *[float(np.mean(labels[graded] == c)) for c in range(3)],
        int(np.sum(labels[graded] == X))))
    print("  possessions {}  mean {:.1f} s  home share of possession time {:.3f}".format(
        len(spells), np.mean([n for _, n in spells]) / comp.FPS,
        home / sum(n for _, n in spells)))


if __name__ == "__main__":
    main()
