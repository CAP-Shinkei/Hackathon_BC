"""Predict the tactical video's possession labels and write the validated submission CSV.

    python scripts/make_submission.py --data-dir /path/to/mtah2027 --bundle model/bundle.joblib

Prints only aggregate facts (path, md5, label counts, possession count, and the shares and
warnings of mtah/diagnostics.py); it never prints frame-level data.
"""
import argparse
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "official")
for folder in (ROOT, OFFICIAL, os.path.join(OFFICIAL, "evaluation")):
    if folder not in sys.path:
        sys.path.insert(0, folder)

import games as official_games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from mtah import diagnostics, io_atd, submission  # noqa: E402

DATA_DIR_ENV = "MTAH_DATA_DIR"
DEFAULT_BUNDLE = os.path.join(ROOT, "model", "bundle.joblib")
DEFAULT_OUT = os.path.join(ROOT, "out", "submissions", "submission.csv")
HASH_CHUNK_BYTES = 1 << 20
COUNT_NAMES = ("H", "A", "D", "X")


def _video_blocks(games_module, game):
    """Half-open video-frame spans of the live halves, from inclusive (half, first, last) triples."""
    return [(first, last + 1) for _, first, last in games_module.half_video_blocks(game)]


def _kickoff_frames(games_module, game_config, blocks):
    """One kick-off per block: the first block starts at its own start, the second at its anchor."""
    second_anchor = games_module.anchor_video_frame(game_config["kickoff_2nd"])
    return [blocks[0][0], second_anchor]


def _tracking_paths(games_module, game, data_dir):
    paths = games_module.asset_paths(game, data_dir)
    meta, raw = paths["atd_meta"], paths["atd_raw"]
    missing = [path for path in (meta, raw) if not os.path.isfile(path)]
    if missing:
        raise IOError("tracking files not found ({}): {}".format(
            DATA_DIR_ENV, ", ".join(missing)))
    return meta, raw


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


DRAFT_SUFFIX = ".unvalidated"  # a file only takes its final name once it has validated


def _aggregates(table, match, out_path):
    codes = pm.label_codes(table)
    counts = {name: int((codes == index).sum()) for index, name in enumerate(COUNT_NAMES)}
    possessions = len(pm.possession_spells(codes, pm.valid_runs(codes)))
    tracking = diagnostics.input_summary(match)
    labels = diagnostics.output_summary(codes, match.blocks, possessions)
    warnings = diagnostics.input_warnings(tracking) + diagnostics.output_warnings(labels)
    return {"output": out_path, "md5": _md5(out_path), "counts": counts,
            "possessions": possessions, "rows": match.n_frames,
            "tracking": tracking, "labels": labels, "warnings": warnings}


def build(data_dir, bundle_path, out_path, game, games_module):
    """Predict, write and validate the submission for one game. Raises ValueError if invalid."""
    config = games_module.GAMES[game]
    n_frames = config["n_video_frames"]
    blocks = _video_blocks(games_module, game)
    meta, raw = _tracking_paths(games_module, game, data_dir)
    match = io_atd.load_atd(meta, raw, n_frames, tuple(config["pitch"]), blocks, game_id=game)
    bundle = submission.load_bundle(bundle_path)
    codes = submission.predict_codes(bundle, match,
                                     kickoff_frames=_kickoff_frames(games_module, config, blocks))
    draft_path = out_path + DRAFT_SUFFIX
    submission.write_csv(codes, draft_path)
    table = pm.read_label_csv(draft_path)
    reason = pm.validate_submission(table, n_frames)
    if reason:
        raise ValueError("submission is invalid ({}); left at {}".format(reason, draft_path))
    os.replace(draft_path, out_path)
    return _aggregates(table, match, out_path)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="data bundle folder (default: $MTAH_DATA_DIR)")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE, help="trained bundle to predict with")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the CSV")
    parser.add_argument("--game", default=official_games.GAME, help="game id in games.GAMES")
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir or os.environ.get(DATA_DIR_ENV)
    if not args.data_dir:
        parser.error("give --data-dir or set {}".format(DATA_DIR_ENV))
    return args


def _rounded(value):
    return round(value, 3) if isinstance(value, float) else value


def _report(summary):
    counts = summary["counts"]
    print("output: {}".format(summary["output"]))
    print("md5: {}".format(summary["md5"]))
    print("rows: {}".format(summary["rows"]))
    print("labels: " + "  ".join("{} {}".format(name, counts[name]) for name in COUNT_NAMES))
    print("possessions: {}".format(summary["possessions"]))
    for section in ("tracking", "labels"):
        print("{}: {}".format(section, "  ".join(
            "{} {}".format(name, _rounded(value)) for name, value in summary[section].items())))
    for text in summary["warnings"]:
        print("WARNING: " + text)
    if not summary["warnings"]:
        print("sanity checks: no warnings")


def main(argv=None):
    args = _parse_args(argv)
    summary = build(args.data_dir, args.bundle, args.out, args.game, official_games)
    _report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
