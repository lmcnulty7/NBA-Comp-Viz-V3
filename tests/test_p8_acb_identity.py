"""tools.p8_acb_identity: SkillCorner events to a running score, the final-score check and the adapted D130 verdict,
on synthetic data. No OCR, no network, no SkillCorner download."""

import json

import pytest

from tools import p8_acb_identity as A
from tools import p8_identity as I

HOME, AWAY = 10, 20
DYN = {
    "shots": [
        {"period": 1, "startGameClock": 590.0, "endGameClock": 588.5, "offTeamId": HOME, "outcome": True, "three": False},
        {"period": 1, "startGameClock": 570.0, "endGameClock": 569.0, "offTeamId": AWAY, "outcome": True, "three": True},
        {"period": 1, "startGameClock": 550.0, "endGameClock": 549.0, "offTeamId": HOME, "outcome": False, "three": True},
        {"period": 2, "startGameClock": 300.0, "endGameClock": None, "offTeamId": HOME, "outcome": True, "three": True},
    ],
    "free_throws": [
        {"period": 1, "gameClock": 500.0, "offTeamId": AWAY, "outcome": True},
        {"period": 1, "gameClock": 500.0, "offTeamId": AWAY, "outcome": False},
    ],
}


def test_events_score_made_shots_and_free_throws_for_the_right_team():
    ev = A.events_from(DYN, HOME, AWAY)
    assert ev == [(1, 588.5, "0-2"), (1, 569.0, "3-2"), (1, 500.0, "4-2"), (2, 300.0, "4-5")]


def test_events_reject_an_unknown_team():
    bad = {"shots": [{"period": 1, "endGameClock": 1.0, "offTeamId": 99, "outcome": True, "three": False}]}
    with pytest.raises(ValueError):
        A.events_from(bad, HOME, AWAY)


def test_a_frame_matches_skillcorner_state_in_either_order_within_the_slack():
    ev = A.events_from(DYN, HOME, AWAY)
    # first quarter, 8:00 left (480 s): away 4, home 2 after the free throw at 500 s
    assert I.frame_matches(I.parse_tokens(["8:00", "2", "4"]), ev)
    assert I.frame_matches(I.parse_tokens(["8:00", "4", "2"]), ev)
    assert not I.frame_matches(I.parse_tokens(["8:00", "4", "3"]), ev)


def test_the_verdict_confirms_only_one_game_matched_at_all_three_moments():
    games = ["1", "2", "3"]
    assert I.verdict([["1"], ["1"], ["1"]], games) == {"status": "confirmed", "game": "1"}
    assert I.verdict([["1", "2"], ["1", "2"], ["1", "2"]], games)["status"] == "ambiguous"
    assert I.verdict([["1"], [], ["1"]], games)["status"] == "rejected"


def test_final_scores_compare_the_rebuild_with_matches_json(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "matches" / "7").mkdir(parents=True)
    (root / "data" / "matches.json").write_text(json.dumps([
        {"id": 7, "home_team": {"id": HOME}, "away_team": {"id": AWAY}, "home_score": 5, "away_score": 4}]))
    (root / "data" / "matches" / "7" / "7_dynamic_events.json").write_text(json.dumps(DYN))
    assert A.final_scores(root) == [{"game": 7, "built": "5-4", "official": "5-4", "equal": True, "events": 4}]
    (root / "data" / "matches.json").write_text(json.dumps([
        {"id": 7, "home_team": {"id": HOME}, "away_team": {"id": AWAY}, "home_score": 6, "away_score": 4}]))
    assert A.final_scores(root)[0]["equal"] is False


def test_the_acb_reading_adds_the_combined_game_and_shot_clock():
    for tok, sec in (("09.40/24", 580), ("09.40124", 580), ("08.4524", 525), ("08*45/24", 525)):
        assert sec in A.parse_acb([tok])["clocks"], tok
        assert sec not in I.parse_tokens([tok])["clocks"], tok
    assert 556 in A.parse_acb(["09.16"])["clocks"] and 556 in I.parse_tokens(["09.16"])["clocks"]
    assert A.parse_acb(["11.40/24"])["clocks"] == []          # more than a FIBA quarter


def test_a_moment_keeps_the_first_matching_section_per_reading():
    miss = {"games": [], "frames": 8, "frames_with_clock": 0, "frames_with_two_ints": 0}
    hit = dict(miss, games=["7"])
    tries = [{"start": 0, "downloaded": True, "unchanged": miss, "acb_clock": hit},
             {"start": 90, "downloaded": True, "unchanged": miss, "acb_clock": miss}]
    assert A.moment_result(tries) == {"unchanged": [], "acb_clock": ["7"]}
    assert A.moment_result([{"start": 0, "downloaded": False}]) == {"unchanged": [], "acb_clock": []}


def test_the_halftime_retry_fires_only_when_two_sections_read_no_clock():
    blank = {"games": [], "frames": 8, "frames_with_clock": 0, "frames_with_two_ints": 0}
    clock = dict(blank, frames_with_clock=3)
    t = lambda u, a: {"start": 0, "downloaded": True, "unchanged": u, "acb_clock": a}
    assert A.needs_halftime_retry([t(blank, blank), t(blank, blank)])
    assert not A.needs_halftime_retry([t(blank, blank)])                       # the 90 s retry comes first
    assert not A.needs_halftime_retry([t(blank, clock), t(blank, blank)])      # a clock was read: not half-time
    assert A.needs_halftime_retry([{"start": 0, "downloaded": False}, t(blank, blank)])
    assert A.HALFTIME_RETRY_S == 900


def test_the_calibration_set_is_fixed_and_keeps_out_the_candidate_game():
    assert len(A.CALIBRATION_POSITIVES) == 3 and len(A.CALIBRATION_NEGATIVES) == 2
    assert "179612" not in {g for _, g in A.CALIBRATION_POSITIVES}
