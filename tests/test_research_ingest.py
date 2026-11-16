"""tools/research_ingest.py: name normalisation, blob hash checks and pin loading. No network, no data."""

import hashlib
import json
import subprocess

import pytest

from tools import research_ingest as I

GAME = "11.01.2015.AAA.at.BBB"            # made-up ids: never a real test game


def test_prefixed_names_normalise_to_the_game_the_frozen_regex_accepts():
    assert I.prefixed_name(GAME) == "2016.NBA.Raw.SportVU.Game.Logs11.01.2015.AAA.at.BBB.7z"
    assert I.normal_name(I.prefixed_name(GAME)) == GAME
    assert I.normal_name(GAME + ".7z") == GAME


def test_the_frozen_regex_still_rejects_the_prefixed_name():
    from sportvu.manifest import parse_name
    assert parse_name(I.prefixed_name(GAME)) is None


@pytest.mark.parametrize("name", ["2016.NBA.Raw.SportVU.Game.Logs", "readme.md", "x" + GAME + ".7z",
                                  I.PREFIX + GAME + ".zip", I.PREFIX + "11.01.2015.aaa.at.BBB.7z"])
def test_other_names_do_not_normalise(name):
    assert I.normal_name(name) is None


def test_git_blob_sha1_matches_git():
    assert I.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert I.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_git_blob_sha1_matches_git_hash_object(tmp_path):
    p = tmp_path / "a.7z"
    data = bytes(range(256)) * 40
    p.write_bytes(data)
    out = subprocess.run(["git", "hash-object", "--no-filters", str(p)], capture_output=True, text=True, check=True)
    assert I.git_blob_sha1(data) == out.stdout.strip() == I.git_hash_object(p)
    assert I.git_blob_sha1(data) != hashlib.sha1(data).hexdigest()      # the blob header is part of the hash


def test_check_blob_fails_loudly_on_a_mismatch():
    data = b"archive bytes"
    assert I.check_blob(GAME, data, I.git_blob_sha1(data)) == I.git_blob_sha1(data)
    with pytest.raises(SystemExit, match="does not match"):
        I.check_blob(GAME, data + b"!", I.git_blob_sha1(data))


def test_checked_pins_needs_a_blob_pin_for_every_game():
    other = "11.02.2015.CCC.at.DDD"
    pins = {GAME: "a" * 40, other: "b" * 40}
    assert I.checked_pins(pins) == pins
    with pytest.raises(SystemExit, match="without an archive blob pin"):
        I.checked_pins({GAME: None})                     # what the door returns for a source without pins
    with pytest.raises(SystemExit, match="without an archive blob pin"):
        I.checked_pins({GAME: "xyz"})
    with pytest.raises(SystemExit, match="not a game name"):
        I.checked_pins({"junk": "c" * 40})
    with pytest.raises(SystemExit, match="no pinned archive"):
        I.checked_pins({})


def test_check_listing_requires_exactly_the_pinned_prefixed_entries():
    pins = {GAME: "a" * 40}
    good = [{"name": I.prefixed_name(GAME), "sha": "a" * 40}, {"name": "11.03.2015.EEE.at.FFF.7z", "sha": "d" * 40}]
    assert I.check_listing(good, pins) == 1
    with pytest.raises(SystemExit, match="differs from the pin"):
        I.check_listing([{"name": I.prefixed_name(GAME), "sha": "e" * 40}], pins)
    with pytest.raises(SystemExit, match="not the pinned games"):
        I.check_listing(good + [{"name": I.prefixed_name("11.04.2015.GGG.at.HHH"), "sha": "f" * 40}], pins)


def test_the_output_root_may_not_sit_under_the_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    with pytest.raises(SystemExit):
        I.guard_root(tmp_path / "data" / "sportvu_untouched")
    assert I.guard_root(tmp_path / "elsewhere") == (tmp_path / "elsewhere").resolve()


def test_every_blocked_lane_step_is_described():
    steps = set(I.BLOCKED["steps"])
    for lane, needs in I.BLOCKED["lanes_needing_them"].items():
        assert set(needs) <= steps, lane
    assert set(I.BLOCKED["pages_per_game"]) == {"play_by_play", "shot_chart"}


def test_shot_counts_are_row_counts_only_and_never_read_an_outcome():
    import pandas as pd

    other = "11.02.2015.CCC.at.DDD"                      # no outcome column anywhere: reading one would fail
    shots = pd.DataFrame({"sportvu_game": [GAME, GAME, other]})
    sf = pd.DataFrame({"sportvu_game": [GAME, other, other], "confidence_flag": ["high", "low", "high"]})
    feats = pd.DataFrame({"sportvu_game": [GAME, GAME, other], "has_contest": [1, 0, 1]})
    out = I.shot_counts(shots, sf, feats, {GAME: 2})
    assert out[GAME] == {"shot_chart_rows": 2, "pbp_field_goal_attempts": 2, "shot_frames_rows": 1,
                         "shot_frames_high": 1, "assembled_rows": 2, "has_contest": 1}
    assert out[other]["pbp_field_goal_attempts"] is None and out[other]["shot_frames_rows"] == 2
    assert not any("made" in k for c in out.values() for k in c)


def test_the_bref_stage_stays_under_the_site_limit():
    assert I.BREF_RPM <= 15
