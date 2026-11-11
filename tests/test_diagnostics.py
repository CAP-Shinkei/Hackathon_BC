"""The sanity report: aggregates of a loaded match and of a predicted label sequence."""
import numpy as np
import pytest

from mtah import diagnostics
from mtah.schema import A, D, FPS, H, Match

PITCH = (105.0, 68.0)
N_FRAMES = 3000
BLOCKS = ((0, 1500), (1500, 3000))
PLAYERS = 11


def make_match(seed=0):
    rng = np.random.default_rng(seed)
    home = rng.uniform(0.0, 1.0, (N_FRAMES, PLAYERS, 2)) * np.array(PITCH)
    away = rng.uniform(0.0, 1.0, (N_FRAMES, PLAYERS, 2)) * np.array(PITCH)
    ball = rng.uniform(0.0, 1.0, (N_FRAMES, 2)) * np.array(PITCH)
    return Match(game_id="synthetic", home_xy=home.astype(np.float32),
                 away_xy=away.astype(np.float32), ball_xy=ball.astype(np.float32),
                 blocks=BLOCKS, pitch=PITCH)


def _replace(match, **changes):
    fields = {"game_id": match.game_id, "home_xy": match.home_xy, "away_xy": match.away_xy,
              "ball_xy": match.ball_xy, "blocks": match.blocks, "pitch": match.pitch}
    return Match(**{**fields, **changes})


def alternating_codes(spell_frames):
    """H and A spells of the given length over every frame."""
    spell = np.arange(N_FRAMES) // spell_frames
    return np.where(spell % 2 == 0, H, A).astype(np.int8)


def test_healthy_match_reports_plain_aggregates_and_no_warnings():
    summary = diagnostics.input_summary(make_match())

    assert summary["graded_frames"] == N_FRAMES
    assert summary["ball_missing_share"] == 0.0
    assert summary["home_players_mean"] == PLAYERS
    assert summary["away_players_mean"] == PLAYERS
    assert summary["ball_off_pitch_share"] == 0.0
    assert summary["players_off_pitch_share"] == 0.0
    assert diagnostics.input_warnings(summary) == []


def test_only_frames_inside_the_blocks_are_counted():
    match = make_match()
    ball = match.ball_xy.copy()
    ball[1000:] = np.nan
    half = _replace(match, ball_xy=ball, blocks=((0, 1000),))

    summary = diagnostics.input_summary(half)

    assert summary["graded_frames"] == 1000
    assert summary["ball_missing_share"] == 0.0


def test_ball_missing_share_counts_frames_without_a_ball():
    match = make_match()
    ball = match.ball_xy.copy()
    ball[:1230] = np.nan

    summary = diagnostics.input_summary(_replace(match, ball_xy=ball))

    assert summary["ball_missing_share"] == pytest.approx(1230 / N_FRAMES)
    assert diagnostics.input_warnings(summary) == []


def test_warns_when_the_ball_is_almost_never_tracked():
    match = make_match()
    ball = match.ball_xy.copy()
    ball[100:] = np.nan

    warnings = diagnostics.input_warnings(diagnostics.input_summary(_replace(match, ball_xy=ball)))

    assert len(warnings) == 1 and "ball" in warnings[0]


def test_warns_when_a_team_has_few_tracked_players():
    match = make_match()
    away = match.away_xy.copy()
    away[:, 2:] = np.nan

    summary = diagnostics.input_summary(_replace(match, away_xy=away))
    warnings = diagnostics.input_warnings(summary)

    assert summary["away_players_mean"] == 2
    assert len(warnings) == 1 and "away" in warnings[0]


def test_warns_when_a_team_has_no_slots_at_all():
    match = make_match()
    empty = np.empty((N_FRAMES, 0, 2), dtype=np.float32)

    summary = diagnostics.input_summary(_replace(match, home_xy=empty))
    warnings = diagnostics.input_warnings(summary)

    assert summary["home_players_mean"] == 0
    assert any("home" in text for text in warnings)


def test_warns_when_coordinates_do_not_fit_the_pitch():
    match = make_match()
    scaled = _replace(match, ball_xy=match.ball_xy * 100.0, home_xy=match.home_xy * 100.0)

    summary = diagnostics.input_summary(scaled)
    warnings = diagnostics.input_warnings(summary)

    assert summary["ball_off_pitch_share"] > 0.9
    assert any("ball" in text and "pitch" in text for text in warnings)
    assert any("player" in text and "pitch" in text for text in warnings)


def test_a_match_without_graded_frames_is_refused():
    match = _replace(make_match(), blocks=())

    with pytest.raises(ValueError, match="no graded frames"):
        diagnostics.input_summary(match)


def test_plausible_labels_report_shares_and_no_warnings():
    codes = alternating_codes(spell_frames=15 * FPS)

    summary = diagnostics.output_summary(codes, BLOCKS, possessions=8)

    assert summary["dead_share"] == 0.0
    assert summary["home_share_of_live"] == pytest.approx(0.5)
    assert summary["possessions_per_minute"] == pytest.approx(4.0)
    assert diagnostics.output_warnings(summary) == []


def test_frames_outside_the_blocks_do_not_count_as_dead():
    codes = alternating_codes(spell_frames=15 * FPS)
    codes[1500:] = D

    summary = diagnostics.output_summary(codes, ((0, 1500),), possessions=4)

    assert summary["dead_share"] == 0.0


def test_warns_when_almost_everything_is_dead():
    codes = np.full(N_FRAMES, D, dtype=np.int8)
    codes[:100] = H

    warnings = diagnostics.output_warnings(diagnostics.output_summary(codes, BLOCKS, possessions=1))

    assert any("dead" in text for text in warnings)


def test_warns_when_one_team_never_has_the_ball():
    codes = np.full(N_FRAMES, H, dtype=np.int8)

    warnings = diagnostics.output_warnings(diagnostics.output_summary(codes, BLOCKS, possessions=5))

    assert any("one team" in text for text in warnings)


@pytest.mark.parametrize("possessions", [1, 40])
def test_warns_when_the_possession_rate_is_implausible(possessions):
    codes = alternating_codes(spell_frames=15 * FPS)

    summary = diagnostics.output_summary(codes, BLOCKS, possessions=possessions)

    assert any("possessions per minute" in text for text in diagnostics.output_warnings(summary))


def test_all_dead_labels_have_no_home_share():
    codes = np.full(N_FRAMES, D, dtype=np.int8)

    summary = diagnostics.output_summary(codes, BLOCKS, possessions=0)

    assert summary["home_share_of_live"] is None
    assert any("dead" in text for text in diagnostics.output_warnings(summary))
