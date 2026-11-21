"""tools.p8_search: query forms, text rules and game classes (PLAN.md D128 to D136). No network."""

import os

import pytest

from tools import p8_search as S

G = {"game": "01.12.2016.BOS.at.NYK", "date": "2016-01-12", "away": "BOS", "home": "NYK"}


def test_youtube_forms_keep_the_frozen_three_and_add_city_channel_and_chinese_forms():
    from sportvu.manifest import query_forms
    qs = S.queries(G, "youtube")
    assert qs[:3] == query_forms(G)
    assert "Boston Celtics vs New York Knicks January 12 2016" in qs
    assert "Nba HD Full Game Celtics at Knicks 2015 / 2016" in qs
    assert "凯尔特人 vs 尼克斯 2016年1月12日 常规赛 全场" in qs
    assert "2016.01.12 凯尔特人 尼克斯 全场录像" in qs


def test_bilibili_uses_the_first_chinese_form_only():
    assert S.queries(G, "bilibili") == ["凯尔特人 vs 尼克斯 2016年1月12日 常规赛 全场"]


@pytest.mark.parametrize("title,why", [
    ("NBA 2K16 Knicks vs Celtics Jan 12 2016", "video-game simulation"),
    ("凯尔特人vs尼克斯 2K16 模拟", "video-game simulation"),
    ("Celtics vs Knicks Game 7 playoffs", "postseason game"),
    ("凯尔特人vs尼克斯 季后赛 全场", "postseason game"),
    ("Celtics vs Knicks 2019 full game", "year 2019"),
])
def test_rejections_in_english_and_chinese(title, why):
    assert S.rejection(title, None, G["date"]) == why


def test_a_plain_regular_season_title_is_not_rejected():
    assert S.rejection("Celtics vs Knicks full game 2015 / 2016", None, G["date"]) is None
    assert S.rejection("凯尔特人vs尼克斯 2016年1月12日 常规赛 全场录像", None, G["date"]) is None


def test_evidence_reads_chinese_dates_and_seasons():
    assert S.evidence(G["date"], "凯尔特人vs尼克斯 2016年1月12日").startswith("date:")
    assert S.evidence(G["date"], "2016.01.12 凯尔特人 尼克斯").startswith("date:")
    assert S.evidence(G["date"], "凯尔特人 尼克斯 2015-16赛季").startswith("season:")
    assert S.evidence(G["date"], "凯尔特人 尼克斯") is None


def test_mentions_chinese_names_and_the_old_mavericks_name():
    assert S.mentions("凯尔特人vs尼克斯", "BOS") and S.mentions("凯尔特人vs尼克斯", "NYK")
    assert S.mentions("小牛vs马刺", "DAL") and S.mentions("独行侠vs马刺", "DAL")
    assert not S.mentions("勇士vs骑士", "BOS")


def test_length_classes():
    assert S.length_class(5000, "Celtics vs Knicks") == "full"
    assert S.length_class(1100, "Celtics vs Knicks Condensed Game") == "condensed"
    assert S.length_class(1100, "Celtics vs Knicks Highlights") is None
    assert S.length_class(300, "Condensed Game") is None


@pytest.mark.skipif(not os.environ.get("DATA_ROOT"), reason="needs the SportVU contract manifest")
def test_game_classes_follow_the_existing_guards():
    games = S.contract_games()
    cls = {n: S.game_class(n, g) for n, g in games.items()}
    assert cls["12.16.2015.PHX.at.GSW"] == "spent"
    assert all(c == "spent" for n, c in cls.items() if n.endswith(".at.CLE") and games[n]["status"] == "ok")
    assert cls["12.25.2015.NOP.at.MIA"] == "d85_test" and cls["12.09.2015.SAS.at.TOR"] == "d85_test"
    assert all(c == "d85_test" for n, c in cls.items()
               if n.split(".")[-1] in ("MIA", "TOR", "LAC", "LAL") and games[n]["status"] == "ok")
    assert cls["01.12.2016.BOS.at.NYK"] == "training"
    assert not any(n.startswith("12.05.2015.") for n in games)        # the sealed Set A is not in the listing
