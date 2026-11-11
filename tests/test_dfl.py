import numpy as np
import pytest

from mtah.external import dfl
from mtah.schema import A, D, H, X

SYNTHETIC_POSITIONS = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
<Positions>
<FrameSet TeamId="T_HOME" PersonId="P2" GameSection="firstHalf">
<Frame N="10000" X="-10.0" Y="5.0" S="1.0"/>
<Frame N="10001" X="-9.0" Y="5.5" S="1.0"/>
</FrameSet>
<FrameSet TeamId="T_HOME" PersonId="P1" GameSection="firstHalf">
<Frame N="10001" X="0.0" Y="0.0" S="1.0"/>
</FrameSet>
<FrameSet TeamId="T_AWAY" PersonId="P9" GameSection="firstHalf">
<Frame N="10000" X="20.0" Y="-3.0" S="1.0"/>
</FrameSet>
<FrameSet TeamId="BALL" PersonId="" GameSection="firstHalf">
<Frame N="10000" X="0.0" Y="0.0" Z="0.1" S="0" BallStatus="1" BallPossession="1"/>
<Frame N="10001" X="1.0" Y="0.0" Z="0.1" S="0" BallStatus="0" BallPossession="2"/>
</FrameSet>
<FrameSet TeamId="T_HOME" PersonId="P2" GameSection="secondHalf">
<Frame N="100000" X="0.0" Y="0.0" S="1.0"/>
</FrameSet>
<FrameSet TeamId="BALL" PersonId="" GameSection="secondHalf">
<Frame N="100000" X="2.0" Y="0.0" Z="0.1" S="0" BallStatus="1" BallPossession="2"/>
<Frame N="100001" X="3.0" Y="1.0" Z="0.1" S="0" BallStatus="1" BallPossession="2"/>
</FrameSet>
</Positions>
</PutDataRequest>
"""

PITCH = (105.0, 68.0)


def _write_xml(tmp_path, text, name="positions.xml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _parse_synthetic(tmp_path):
    path = _write_xml(tmp_path, SYNTHETIC_POSITIONS)
    return dfl.parse_positions(path, "T_HOME", "T_AWAY", PITCH)


def test_parse_positions_concatenates_halves_into_one_timeline(tmp_path):
    # Arrange / Act
    parsed = _parse_synthetic(tmp_path)

    # Assert: 2 frames per half, half 2 starts after half 1
    assert parsed["home_xy"].shape == (4, 2, 2)
    assert parsed["away_xy"].shape == (4, 1, 2)
    assert parsed["ball_xy"].shape == (4, 2)
    assert parsed["blocks"] == ((0, 2), (2, 4))


def test_parse_positions_shifts_centred_coordinates_to_pitch_corner(tmp_path):
    parsed = _parse_synthetic(tmp_path)

    # P2 at (-10, 5) centred -> (42.5, 39.0) with a 105 x 68 pitch
    np.testing.assert_allclose(parsed["home_xy"][0, 1], [42.5, 39.0], atol=1e-4)
    np.testing.assert_allclose(parsed["ball_xy"][0], [52.5, 34.0], atol=1e-4)


def test_parse_positions_uses_nan_for_players_off_the_pitch(tmp_path):
    parsed = _parse_synthetic(tmp_path)

    # P1 only appears in frame 1 of the first half; P2 is absent in frame 3
    assert np.isnan(parsed["home_xy"][0, 0]).all()
    assert np.isnan(parsed["home_xy"][3, 1]).all()
    assert not np.isnan(parsed["home_xy"][1, 0]).any()


def test_parse_positions_assigns_slots_by_sorted_player_id(tmp_path):
    parsed = _parse_synthetic(tmp_path)

    assert list(parsed["home_ids"]) == ["P1", "P2"]
    assert list(parsed["away_ids"]) == ["P9"]


def test_parse_positions_reads_ball_status_and_possession_per_frame(tmp_path):
    parsed = _parse_synthetic(tmp_path)

    np.testing.assert_array_equal(parsed["ball_status"], [1, 0, 1, 1])
    np.testing.assert_array_equal(parsed["ball_possession"], [1, 2, 2, 2])


def test_ball_labels_mark_dead_ball_as_dead_and_live_ball_by_possession(tmp_path):
    parsed = _parse_synthetic(tmp_path)

    labels = dfl.ball_labels(parsed["ball_status"], parsed["ball_possession"])

    np.testing.assert_array_equal(labels, [H, D, A, A])


def test_ball_labels_are_unknown_when_status_is_missing():
    status = np.array([-1, 1], dtype=np.int8)
    possession = np.array([1, 0], dtype=np.int8)

    labels = dfl.ball_labels(status, possession)

    np.testing.assert_array_equal(labels, [X, X])


def test_parse_positions_rejects_non_positive_pitch(tmp_path):
    path = _write_xml(tmp_path, SYNTHETIC_POSITIONS)

    with pytest.raises(ValueError, match="pitch"):
        dfl.parse_positions(path, "T_HOME", "T_AWAY", (0.0, 68.0))


def test_parse_positions_rejects_unknown_game_section(tmp_path):
    text = SYNTHETIC_POSITIONS.replace('GameSection="secondHalf"', 'GameSection="firstHalfExtra"')
    path = _write_xml(tmp_path, text)

    with pytest.raises(ValueError, match="GameSection"):
        dfl.parse_positions(path, "T_HOME", "T_AWAY", PITCH)


def test_parse_positions_rejects_file_missing_a_half(tmp_path):
    first_half_only = SYNTHETIC_POSITIONS.split('<FrameSet TeamId="T_HOME" PersonId="P2" GameSection="secondHalf">')[0]
    path = _write_xml(tmp_path, first_half_only + "</Positions>\n</PutDataRequest>\n")

    with pytest.raises(ValueError, match="half 2"):
        dfl.parse_positions(path, "T_HOME", "T_AWAY", PITCH)


def test_match_id_rejects_unknown_game():
    with pytest.raises(ValueError, match="unknown DFL game id"):
        dfl.load("dfl_NOPE00")


def test_game_ids_are_seven_prefixed_ids():
    assert len(dfl.GAME_IDS) == 7
    assert all(game_id.startswith("dfl_") for game_id in dfl.GAME_IDS)


def _real_game_available(game_id):
    cache = dfl.DEFAULT_DATA_DIR / (game_id + ".npz")
    positions = dfl.DEFAULT_DATA_DIR / (game_id + "_positions.xml")
    return cache.exists() or positions.exists()


@pytest.mark.skipif(not _real_game_available("dfl_J03WMX"), reason="external_data/dfl not downloaded")
def test_load_real_game_has_expected_shapes_and_label_shares():
    match = dfl.load("dfl_J03WMX")

    n = match.n_frames
    shares = np.bincount(match.labels, minlength=4) / n
    assert n > 100_000
    assert match.home_xy.shape[0] == n and match.away_xy.shape[0] == n
    assert match.home_xy.shape[2] == 2 and match.ball_xy.shape == (n, 2)
    assert shares[H] > 0 and shares[A] > 0 and shares[D] > 0
    assert np.isclose(shares.sum(), 1.0)


META_XML = """<?xml version="1.0" encoding="UTF-8"?>
<MatchInformation>
<Environment PitchX="105.0" PitchY="68.0"/>
<Teams>
<Team Role="home" TeamId="T_HOME"/>
<Team Role="guest" TeamId="T_AWAY"/>
</Teams>
</MatchInformation>
"""

HOME_ONLY_POSITIONS = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
<Positions>
<FrameSet TeamId="T_HOME" PersonId="P1" GameSection="firstHalf">
<Frame N="1" X="0.0" Y="0.0" S="1.0"/>
</FrameSet>
<FrameSet TeamId="BALL" PersonId="" GameSection="firstHalf">
<Frame N="1" X="0.0" Y="0.0" S="0" BallStatus="1" BallPossession="1"/>
</FrameSet>
<FrameSet TeamId="BALL" PersonId="" GameSection="secondHalf">
<Frame N="5" X="0.0" Y="0.0" S="0" BallStatus="1" BallPossession="1"/>
</FrameSet>
<FrameSet TeamId="T_HOME" PersonId="P1" GameSection="secondHalf">
<Frame N="5" X="0.0" Y="0.0" S="1.0"/>
</FrameSet>
</Positions>
</PutDataRequest>
"""


class _FakeResponse:
    def __init__(self, status_code, chunks):
        self.status_code = status_code
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def iter_content(self, chunk_size):
        yield from self._chunks


def _patch_http(monkeypatch, status_code, chunks):
    monkeypatch.setattr(dfl.requests, "get",
                        lambda url, stream, timeout: _FakeResponse(status_code, chunks))


def test_xml_path_names_file_after_game_and_kind(tmp_path):
    # Arrange / Act
    path = dfl._xml_path(tmp_path, "dfl_J03WMX", "meta")

    # Assert
    assert path == tmp_path / "dfl_J03WMX_meta.xml"


def test_stream_to_file_writes_all_chunks_and_leaves_no_partial(tmp_path, monkeypatch):
    # Arrange
    _patch_http(monkeypatch, 200, [b"ab", b"cd"])
    target = tmp_path / "positions.xml"

    # Act
    dfl._stream_to_file("https://example.test/positions", target)

    # Assert
    assert target.read_bytes() == b"abcd"
    assert not (tmp_path / "positions.xml.part").exists()


def test_stream_to_file_rejects_non_200_status_without_writing(tmp_path, monkeypatch):
    # Arrange
    _patch_http(monkeypatch, 404, [b"not found"])
    target = tmp_path / "positions.xml"

    # Act / Assert
    with pytest.raises(OSError, match="HTTP 404"):
        dfl._stream_to_file("https://example.test/positions", target)
    assert not target.exists()


def test_stream_to_file_removes_empty_partial_download(tmp_path, monkeypatch):
    # Arrange
    _patch_http(monkeypatch, 200, [])
    target = tmp_path / "positions.xml"

    # Act / Assert
    with pytest.raises(OSError, match="empty body"):
        dfl._stream_to_file("https://example.test/positions", target)
    assert not target.exists()
    assert not (tmp_path / "positions.xml.part").exists()


def test_download_fetches_only_the_files_not_already_present(tmp_path, monkeypatch):
    # Arrange
    fetched = []
    monkeypatch.setattr(dfl, "_stream_to_file", lambda url, target: fetched.append(target))
    existing = tmp_path / "dfl_J03WMX_meta.xml"
    existing.write_text("present", encoding="utf-8")
    total_files = 2 * len(dfl.FILE_IDS)

    # Act
    dfl.download(tmp_path)

    # Assert
    assert existing not in fetched
    assert len(fetched) == total_files - 1


def test_read_match_info_returns_pitch_and_home_and_away_team_ids(tmp_path):
    # Arrange
    path = _write_xml(tmp_path, META_XML, name="meta.xml")

    # Act
    info = dfl.read_match_info(path)

    # Assert
    assert info == {"pitch": (105.0, 68.0), "home_team_id": "T_HOME", "away_team_id": "T_AWAY"}


def test_read_match_info_rejects_file_without_environment(tmp_path):
    # Arrange
    path = _write_xml(tmp_path, META_XML.replace('<Environment PitchX="105.0" PitchY="68.0"/>', ""),
                      name="meta.xml")

    # Act / Assert
    with pytest.raises(ValueError, match="Environment"):
        dfl.read_match_info(path)


def test_read_match_info_rejects_file_without_guest_team(tmp_path):
    # Arrange
    path = _write_xml(tmp_path, META_XML.replace('<Team Role="guest" TeamId="T_AWAY"/>', ""),
                      name="meta.xml")

    # Act / Assert
    with pytest.raises(ValueError, match="home and guest"):
        dfl.read_match_info(path)


def test_frame_number_rejects_frame_without_n_attribute():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="without N"):
        dfl._frame_number({"X": "1.0"})


def test_parse_positions_ignores_frameset_with_no_frames(tmp_path):
    # Arrange
    text = SYNTHETIC_POSITIONS.replace(
        "</Positions>",
        '<FrameSet TeamId="T_HOME" PersonId="P5" GameSection="firstHalf">\n</FrameSet>\n</Positions>')
    path = _write_xml(tmp_path, text)

    # Act
    parsed = dfl.parse_positions(path, "T_HOME", "T_AWAY", PITCH)

    # Assert
    assert list(parsed["home_ids"]) == ["P1", "P2"]
    assert parsed["home_xy"].shape == (4, 2, 2)


def test_parse_positions_rejects_team_with_no_player_frames(tmp_path):
    # Arrange
    path = _write_xml(tmp_path, HOME_ONLY_POSITIONS)

    # Act / Assert
    with pytest.raises(ValueError, match="away team"):
        dfl.parse_positions(path, "T_HOME", "T_AWAY", PITCH)


def test_parse_positions_rejects_identical_home_and_away_team_ids(tmp_path):
    # Arrange
    path = _write_xml(tmp_path, SYNTHETIC_POSITIONS)

    # Act / Assert
    with pytest.raises(ValueError, match="must differ"):
        dfl.parse_positions(path, "T_HOME", "T_HOME", PITCH)


def test_load_parses_positions_then_reads_identical_match_from_cache(tmp_path):
    # Arrange
    _write_xml(tmp_path, META_XML, name="dfl_J03WMX_meta.xml")
    _write_xml(tmp_path, SYNTHETIC_POSITIONS, name="dfl_J03WMX_positions.xml")

    # Act
    parsed = dfl.load("dfl_J03WMX", tmp_path)
    cached = dfl.load("dfl_J03WMX", tmp_path)

    # Assert
    assert (tmp_path / "dfl_J03WMX.npz").exists()
    assert parsed.blocks == ((0, 2), (2, 4))
    assert parsed.pitch == (105.0, 68.0)
    assert parsed.labels.tolist() == [H, D, A, A]
    assert cached.blocks == parsed.blocks
    assert cached.pitch == parsed.pitch
    assert np.array_equal(cached.labels, parsed.labels)
    assert np.array_equal(cached.home_xy, parsed.home_xy, equal_nan=True)


def test_load_raises_when_raw_positions_file_is_missing(tmp_path):
    # Arrange
    _write_xml(tmp_path, META_XML, name="dfl_J03WMX_meta.xml")

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="download"):
        dfl.load("dfl_J03WMX", tmp_path)
