"""tools.p8_nba2526_search: title parsing, classes and game matching on synthetic titles. No network."""

import datetime as dt

from tools import p8_nba2526_search as S

GAMES = [
    {"game_id": "0022500100", "date": "2025-10-24", "home": "LAL", "away": "MIN", "kind": "regular"},
    {"game_id": "0022500900", "date": "2026-02-27", "home": "OKC", "away": "DEN", "kind": "regular"},
    {"game_id": "0022500950", "date": "2026-03-09", "home": "OKC", "away": "DEN", "kind": "regular"},
    {"game_id": "0022500300", "date": "2025-11-20", "home": "CHA", "away": "BKN", "kind": "regular"},
]
IDX = S.pair_index(GAMES)


def test_teams_are_word_bounded_and_include_chinese_forms():
    assert S.teams_in("TIMBERWOLVES at LAKERS | FULL GAME") == {"MIN", "LAL"}
    assert S.teams_in("Hornets vs Nets") == {"CHA", "BKN"}
    assert S.teams_in("a heated game") == set()
    assert S.teams_in("雷霆 vs 掘金 全场回放") == {"OKC", "DEN"}


def test_dates_in_many_forms():
    d = dt.date(2025, 10, 24)
    for t in ("October 24, 2025", "Oct. 24, 2025", "24 Oct 2025", "10.24.2025", "10/24/25", "(10.24.25)",
              "2025-10-24", "2025年10月24日", "Oct 24"):
        assert S.dates_in(t) == [d], t
    assert S.dates_in("Feb 27") == [dt.date(2026, 2, 27)]


def test_classes():
    assert S.video_class("TIMBERWOLVES at LAKERS | FULL GAME | October 24, 2025", "x", 6386) == "full"
    assert S.video_class("Lakers vs Wolves FULL GAME Highlights", "x", 2600) is None
    assert S.video_class("EXTENDED: NUGGETS at THUNDER | FULL GAME HIGHLIGHTS | February 27, 2026", "NBA", 3123) == "extended"
    assert S.video_class("EXTENDED: NUGGETS at THUNDER | FULL GAME HIGHLIGHTS", "someone", 3123) is None
    assert S.video_class("Nuggets vs Thunder full game", "x", 1800) is None
    assert S.video_class("Nuggets vs Thunder full game", "x", None) is None


def test_match_by_title_date_or_by_upload_window():
    assert S.match_game("TIMBERWOLVES at LAKERS | FULL GAME | October 24, 2025", None, IDX) == "0022500100"
    assert S.match_game("NUGGETS at THUNDER | FULL GAME | March 9, 2026", None, IDX) == "0022500950"
    assert S.match_game("NUGGETS at THUNDER | FULL GAME", "20260228", IDX) == "0022500900"
    assert S.match_game("NUGGETS at THUNDER | FULL GAME", "20260320", IDX) is None
    assert S.match_game("NUGGETS at THUNDER | FULL GAME | April 1, 2026", None, IDX) is None
    assert S.match_game("Lakers vs Thunder vs Nuggets", None, IDX) is None


def test_simulation_and_live_stream_titles_are_rejected():
    assert S.video_class("NBA LIVE: Denver Nuggets vs Orlando Magic | December 27, 2025 | NBA Regular Season", "x", 4073) is None
    assert S.video_class("LIVE - Cavaliers vs Hornets - Play-By-Play & Reactions (2/20/26)", "x", 9328) is None
    assert S.SIM_RE.search("NBA LIVE Houston Rockets vs New Orleans Pelicans | NBA Regular Season 2k26")
    assert not S.SIM_RE.search("TIMBERWOLVES at LAKERS | FULL GAME | October 24, 2025")


def test_league_highlights_and_playoff_extended():
    assert S.video_class("THUNDER at CLIPPERS | FULL GAME HIGHLIGHTS | March 23, 2025", "@NBA", 600) == "league_highlights"
    assert S.video_class("EXTENDED: #5 RAPTORS at #4 CAVALIERS | FULL GAME 5 HIGHLIGHTS | April 29, 2026", "@NBA", 2743) == "extended"
    assert S.video_class("THUNDER at CLIPPERS | FULL GAME HIGHLIGHTS | March 23, 2025", "fan", 600) is None


def test_quarter_uploads():
    assert S.video_class("Lakers vs Rockets NBA  | FULL GAME  4TH-QTR  | APRIL 26 ,2026 | NBA PLAYOFFS", "x", 542) == "quarter"
    assert S.video_class("Lakers vs Rockets NBA  | FULL 1ST-QTR | APRIL 26 ,2026 | NBA PLAYOFFS", "x", 200) is None
