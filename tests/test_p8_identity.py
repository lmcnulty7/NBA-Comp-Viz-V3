"""tools.p8_identity: scorebug parsing and the D130 verdict rule, on synthetic tokens and play-by-play. No OCR, no
network."""

import pytest

from tools import p8_identity as I

# events as tools.p8_scorebug.pbp returns them: (period, clock_s, "away-home"), sorted period up, clock down
EV_A = [(1, 700.0, "2-0"), (1, 650.0, "2-3"), (2, 600.0, "30-28"), (3, 400.0, "60-55"), (4, 100.0, "95-90")]
EV_B = [(1, 690.0, "0-2"), (2, 610.0, "25-31"), (3, 410.0, "50-62"), (4, 120.0, "88-97")]


def test_parse_reads_clock_period_and_scores():
    p = I.parse_tokens(["BOS 30", "NYK 28", "2ND", "9:45", ":24"])
    assert p["periods"] == [2] and 585 in p["clocks"] and {30, 28} <= set(p["ints"])


def test_parse_reads_sub_minute_clocks_and_quarter_words():
    p = I.parse_tokens(["4th Qtr", "45.3", "95", "90"])
    assert p["periods"] == [4] and 45.3 in p["clocks"] and p["ints"][:2] == [95, 90]


def test_a_frame_matches_the_score_at_its_clock_in_either_order():
    assert I.frame_matches(I.parse_tokens(["2ND", "9:45", "30", "28"]), EV_A)
    assert I.frame_matches(I.parse_tokens(["2ND", "9:45", "28", "30"]), EV_A)
    assert not I.frame_matches(I.parse_tokens(["2ND", "9:45", "30", "29"]), EV_A)


def test_a_frame_without_a_clock_or_two_scores_never_matches():
    assert not I.frame_matches(I.parse_tokens(["2ND", "30", "28"]), EV_A)
    assert not I.frame_matches(I.parse_tokens(["9:45", "30"]), EV_A)


def test_a_moment_needs_two_matching_frames():
    one = [I.parse_tokens(["2ND", "9:45", "30", "28"]), I.parse_tokens(["AD", "x"])]
    two = one + [I.parse_tokens(["2ND", "9:44", "30", "28"])]
    assert I.moment_games(one, {"A": EV_A}) == []
    assert I.moment_games(two, {"A": EV_A}) == ["A"]


def test_confirmed_only_when_one_game_matches_every_moment():
    assert I.verdict([["A"], ["A"], ["A"]], ["A", "B"]) == {"status": "confirmed", "game": "A"}
    assert I.verdict([["A"], ["A"], []], ["A", "B"])["status"] == "rejected"
    assert I.verdict([["A", "B"], ["A", "B"], ["A", "B"]], ["A", "B"])["status"] == "ambiguous"


def test_one_matching_moment_is_never_enough():
    assert I.verdict([["A"]], ["A"])["status"] == "rejected"


def test_check_refuses_a_spent_or_test_video_before_anything():
    with pytest.raises(SystemExit):
        I.check("youtube", "f8lAcHg6kk0", 5000, ["12.16.2015.PHX.at.GSW"])     # the spent held-out game
    with pytest.raises(SystemExit):
        I.check("youtube", "x", 5000, ["12.25.2015.NOP.at.MIA"])                # a D85 test game
