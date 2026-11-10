"""End to end on a synthetic match in the competition file format.

The match is built here from random walks and a ball that alternates between following a home
player, following an away player and lying still outside the pitch. It is trained on, damaged,
exported as an ATD pair, then turned into a submission by make_submission.build with a fake games
module. The real match data is never touched.
"""
import dataclasses
import hashlib
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "official")
SCRIPTS = os.path.join(ROOT, "scripts")
for folder in (OFFICIAL, os.path.join(OFFICIAL, "evaluation"), SCRIPTS):
    if folder not in sys.path:
        sys.path.insert(0, folder)

import games as official_games  # noqa: E402
import possession_metrics as pm  # noqa: E402
import make_submission  # noqa: E402
from mtah import atd_export, gapsim, io_atd, model, submission  # noqa: E402
from mtah.schema import A, D, H, Match  # noqa: E402
from mtah.scoring import score  # noqa: E402

PITCH = (105.0, 68.0)
HALF_FRAMES = 3000
PAUSE_FRAMES = 300
TOTAL_FRAMES = 2 * HALF_FRAMES + PAUSE_FRAMES
BLOCKS = ((0, HALF_FRAMES), (HALF_FRAMES + PAUSE_FRAMES, TOTAL_FRAMES))
KICKOFF_SECOND_HALF = 3350  # 2 min 14 s into the second block's frame count, at 25 fps
N_SLOTS = 11
PLAYER_STEP_SD_M = 0.05
SEGMENT_MIN_FRAMES = 75
SEGMENT_MAX_FRAMES = 250
BALL_NOISE_M = 0.3
DEAD_BALL_XY = (-30.0, 34.0)
STATES = (H, A, D)
TEST_GAME = "TEST"
TEST_STEM = "TEST_tactical"
MIN_MACRO_F1 = 0.6
TRAIN_PARAMS = model.Params(max_iter=30, stride=2)
TEMPERATURE = 4.0
STEP = 5


def _walk(rng, n_frames):
    start = rng.uniform(low=(5.0, 5.0), high=(PITCH[0] - 5.0, PITCH[1] - 5.0),
                        size=(N_SLOTS, 2))
    steps = rng.normal(0.0, PLAYER_STEP_SD_M, size=(n_frames, N_SLOTS, 2))
    path = start + np.cumsum(steps, axis=0)
    path[..., 0] = np.clip(path[..., 0], 0.0, PITCH[0])
    path[..., 1] = np.clip(path[..., 1], 0.0, PITCH[1])
    return path.astype(np.float32)


def _next_state(rng, previous):
    choices = [state for state in STATES if state != previous]
    return int(choices[rng.integers(len(choices))])


def _ball_track(state, rng, home, away):
    n = len(home)
    if state == D:
        return np.tile(np.asarray(DEAD_BALL_XY), (n, 1))
    players = home if state == H else away
    slot = int(rng.integers(N_SLOTS))
    return players[:, slot] + rng.normal(0.0, BALL_NOISE_M, size=(n, 2))


def _half(rng, home, away):
    n = len(home)
    labels = np.empty(n, dtype=np.int8)
    ball = np.empty((n, 2))
    start, previous = 0, None
    while start < n:
        state = _next_state(rng, previous)
        length = int(rng.integers(SEGMENT_MIN_FRAMES, SEGMENT_MAX_FRAMES + 1))
        stop = min(n, start + length)
        labels[start:stop] = state
        ball[start:stop] = _ball_track(state, rng, home[start:stop], away[start:stop])
        start, previous = stop, state
    return labels, ball


def make_match(game_id, seed):
    """A two-half synthetic match with labels, plus a pause that belongs to no block."""
    rng = np.random.default_rng(seed)
    home, away = _walk(rng, TOTAL_FRAMES), _walk(rng, TOTAL_FRAMES)
    labels = np.full(TOTAL_FRAMES, D, dtype=np.int8)
    ball = np.tile(np.asarray(DEAD_BALL_XY), (TOTAL_FRAMES, 1))
    for start, stop in BLOCKS:
        labels[start:stop], ball[start:stop] = _half(rng, home[start:stop], away[start:stop])
    return Match(game_id=game_id, home_xy=home, away_xy=away,
                 ball_xy=ball.astype(np.float32), blocks=BLOCKS, pitch=PITCH, labels=labels)


def _asset_paths(data_dir):
    folder = os.path.join(data_dir, "tactical")
    return {"atd_meta": os.path.join(folder, TEST_STEM + "_atd.xml"),
            "atd_raw": os.path.join(folder, TEST_STEM + "_atd.txt")}


def make_games_module():
    """A stand-in for official/games.py that knows only the synthetic game."""
    kickoff = {"minute": 2, "second": 14, "frame": 0}
    assert official_games.anchor_video_frame(kickoff) == KICKOFF_SECOND_HALF
    return types.SimpleNamespace(
        GAMES={TEST_GAME: {"n_video_frames": TOTAL_FRAMES, "pitch": PITCH,
                           "kickoff_2nd": kickoff}},
        asset_paths=lambda game, data_dir=None, source="tactical": _asset_paths(data_dir),
        half_video_blocks=lambda game, adjust=0: [
            (1, 0, HALF_FRAMES - 1), (2, BLOCKS[1][0], TOTAL_FRAMES - 1)],
        anchor_video_frame=official_games.anchor_video_frame,
    )


def _md5(path):
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    train = [make_match("S1", 11), make_match("S2", 12)]
    held_out = make_match(TEST_GAME, 13)
    bundle = submission.train_bundle(train, ("random35",), 1, (0,), TRAIN_PARAMS,
                                     TEMPERATURE, STEP, 0)
    bundle_path = submission.save_bundle(bundle, str(root / "model" / "bundle.joblib"))
    damaged = gapsim.damage(held_out, gapsim.REGIMES["random35"], 5)
    atd_export.export_atd(damaged, str(root / "data" / "tactical"), TEST_STEM)
    games_module = make_games_module()
    first = make_submission.build(str(root / "data"), bundle_path,
                                  str(root / "out" / "run1.csv"), TEST_GAME, games_module)
    return types.SimpleNamespace(root=root, data_dir=str(root / "data"), bundle=bundle,
                                 bundle_path=bundle_path, truth=held_out.labels,
                                 games=games_module, summary=first,
                                 csv_path=first["output"])


def _codes(path):
    return pm.label_codes(pm.read_label_csv(path))


def test_submission_passes_official_validation(pipeline):
    table = pm.read_label_csv(pipeline.csv_path)

    reason = pm.validate_submission(table, TOTAL_FRAMES)

    assert reason is None


def test_csv_has_one_row_per_frame_with_header(pipeline):
    with open(pipeline.csv_path, encoding="ascii") as handle:
        lines = handle.read().split("\n")

    assert lines[0] == "frame,label"
    assert len(lines) == TOTAL_FRAMES + 2  # header, one row per frame, trailing newline
    assert lines[-1] == ""


def test_frames_outside_blocks_and_before_second_kickoff_are_dead(pipeline):
    codes = _codes(pipeline.csv_path)

    assert np.all(codes[HALF_FRAMES:BLOCKS[1][0]] == D)
    assert np.all(codes[BLOCKS[1][0]:KICKOFF_SECOND_HALF] == D)


def test_macro_f1_against_true_labels_exceeds_threshold(pipeline):
    codes = _codes(pipeline.csv_path)

    result = score(pipeline.truth, codes, blocks=BLOCKS)

    assert result["macro_f1"] > MIN_MACRO_F1


def test_two_builds_give_byte_identical_files(pipeline):
    second = make_submission.build(pipeline.data_dir, pipeline.bundle_path,
                                   str(pipeline.root / "out" / "run2.csv"), TEST_GAME,
                                   pipeline.games)

    assert second["md5"] == pipeline.summary["md5"]
    assert _md5(pipeline.csv_path) == _md5(second["output"])


def test_bundle_round_trip_predicts_exactly(pipeline):
    loaded = submission.load_bundle(pipeline.bundle_path)
    meta, raw = _asset_paths(pipeline.data_dir).values()
    match = io_atd.load_atd(meta, raw, TOTAL_FRAMES, PITCH, BLOCKS, game_id=TEST_GAME)
    kickoffs = (BLOCKS[0][0], KICKOFF_SECOND_HALF)

    before = submission.predict_codes(pipeline.bundle, match, kickoff_frames=kickoffs)
    after = submission.predict_codes(loaded, match, kickoff_frames=kickoffs)

    assert loaded.names == pipeline.bundle.names
    assert np.array_equal(before, after)


def test_write_csv_refuses_abstention_code(tmp_path):
    with pytest.raises(ValueError, match="never written"):
        submission.write_csv(np.array([H, 3, D], dtype=np.int8), str(tmp_path / "x.csv"))


def test_predict_rejects_bundle_with_different_feature_names(pipeline):
    wrong = dataclasses.replace(pipeline.bundle, names=("not_a_feature",))
    meta, raw = _asset_paths(pipeline.data_dir).values()
    match = io_atd.load_atd(meta, raw, TOTAL_FRAMES, PITCH, BLOCKS, game_id=TEST_GAME)

    with pytest.raises(ValueError, match="feature names"):
        submission.predict_codes(wrong, match)


def test_train_refuses_matches_without_labels():
    unlabelled = dataclasses.replace(make_match("U", 21), labels=None)

    with pytest.raises(ValueError, match="labels"):
        submission.train_bundle([unlabelled], ("random35",), 1, (0,), TRAIN_PARAMS,
                                TEMPERATURE, STEP, 0)


def test_kickoff_frames_must_match_block_count(pipeline):
    meta, raw = _asset_paths(pipeline.data_dir).values()
    match = io_atd.load_atd(meta, raw, TOTAL_FRAMES, PITCH, BLOCKS, game_id=TEST_GAME)

    with pytest.raises(ValueError, match="kickoff"):
        submission.predict_codes(pipeline.bundle, match, kickoff_frames=(0,))
