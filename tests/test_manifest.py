"""sportvu/manifest.py decision rules: names, gaps, team and season evidence, verdicts, priority."""
from sportvu.manifest import (classify, evidence_kind, finalize, gap_flags, mentions, overall, parse_name,
                              priority, season_evidence)


def _moments(q, clocks):
    return [{"q": q, "clock": c, "players": [0] * 10} for c in clocks]


def test_parse_name():
    assert parse_name("12.25.2015.CLE.at.GSW.7z") == {"game": "12.25.2015.CLE.at.GSW", "date": "2015-12-25", "away": "CLE", "home": "GSW"}
    assert parse_name("2016.NBA.Raw.SportVU.Game.Logs12.05.2015.LAL.at.TOR.7z") is None


def test_gap_flags_clean_and_gappy():
    full = sum((_moments(q, [720 - 0.04 * i for i in range(18001)]) for q in (1, 2, 3, 4)), [])
    assert gap_flags(full)["flags"] == []
    gappy = _moments(1, [720, 700, 650, 640]) + _moments(2, [720, 0]) + _moments(3, [700, 5])
    f = gap_flags(gappy)["flags"]
    assert "Q1 gap 700..650 s" in f and "Q1 ends at 640 s" in f and "Q4 missing" in f
    assert "Q3 starts at 700 s" in f and not any(x.startswith("Q2 ends") for x in f)


def test_mentions_teams():
    assert mentions("Cavs vs Warriors Xmas Game", "CLE") and mentions("Cavs vs Warriors Xmas Game", "GSW")
    assert not mentions("Lakers at Celtics", "LAC")          # 'LA' alone never matches the Clippers
    assert mentions("Golden State vs OKC full game", "OKC")


def test_season_evidence():
    assert season_evidence("2015-12-25", "Cavaliers vs Warriors December 25, 2015", "", None).startswith("date")
    assert season_evidence("2015-12-25", "Cavs vs Warriors 12/25/15", "", None).startswith("date")
    assert season_evidence("2015-12-23", "2015-2016 NBA Season Cavaliers vs Knicks", "December 23 2015", None) == "date: December 23 2015"
    assert season_evidence("2015-12-25", "Cavs vs Warriors Xmas Game", "", "20151226").startswith("uploaded")
    assert season_evidence("2015-12-25", "Warriors vs Cavaliers Christmas Day", "2016-17 season", "20210101") is None
    assert season_evidence("2016-01-14", "Lakers vs Warriors 2015-16 season", "", None) == "season: 2015-16"
    assert [evidence_kind(e) for e in ("date: Dec 25, 2015", "season: 2015-16", "text: 2015-2016", "text: December 23 2015",
                                       "uploaded 1 days after the game", None)] == ["date", "season", "season", "date", "upload", None]


def test_classify_candidate():
    g = {"away": "CLE", "home": "GSW", "date": "2015-12-25"}
    base = {"title": "Cavaliers vs Warriors", "description": "", "vcodec": "avc1.4d401f", "height": 720, "duration": 5000}
    assert classify({**base, "season_evidence": "uploaded 1 days after the game"}, g) == "ok_unverified"
    assert classify({**base, "season_evidence": None}, g) == "ambiguous"
    assert classify({**base, "duration": 3000, "season_evidence": "x"}, g) == "short_broadcast"
    assert classify({**base, "height": 480, "season_evidence": "x"}, g) == "low_res"
    assert classify({**base, "title": "Warriors highlights", "season_evidence": "x"}, g) == "no_broadcast"
    assert classify({"error": "ERROR: Requested format is not available", "no_format": True}, g) == "no_avc1"
    assert classify({"error": "Sign in to confirm you're not a bot", "no_format": False}, g) == "probe_error"


def test_overall_precedence():
    ok_bc = {"verdict": "ok_unverified"}
    assert overall({"status": "ok"}, ok_bc, False) == "ok_unverified"
    assert overall({"status": "sportvu_gaps"}, ok_bc, False) == "sportvu_gaps"
    assert overall({"status": "sportvu_broken"}, ok_bc, False) == "sportvu_broken"
    assert overall({"status": "sportvu_gaps"}, {"verdict": "no_broadcast"}, False) == "no_broadcast"
    assert overall({"status": "ok"}, ok_bc, True) == "held_out"


def test_priority_order():
    rows = [{"game": "01.02.2016.DEN.at.GSW", "home": "GSW", "date": "2016-01-02", "verdict": "no_broadcast"},
            {"game": "11.04.2015.NYK.at.CLE", "home": "CLE", "date": "2015-11-04", "verdict": "ok_unverified"},
            {"game": "12.25.2015.CLE.at.GSW", "home": "GSW", "date": "2015-12-25", "verdict": "no_broadcast"},
            {"game": "11.02.2015.MEM.at.GSW", "home": "GSW", "date": "2015-11-02", "verdict": "ok_unverified"}]
    order = [r["game"] for r in priority(rows)]
    assert order == ["12.25.2015.CLE.at.GSW", "11.02.2015.MEM.at.GSW", "01.02.2016.DEN.at.GSW", "11.04.2015.NYK.at.CLE"]


def test_finalize_weak_evidence_and_duplicates():
    sched = [{"game": "11.04.2015.NYK.at.CLE", "date": "2015-11-04", "away": "NYK", "home": "CLE"},
             {"game": "11.13.2015.CLE.at.NYK", "date": "2015-11-13", "away": "CLE", "home": "NYK"},
             {"game": "12.23.2015.NYK.at.CLE", "date": "2015-12-23", "away": "NYK", "home": "CLE"},
             {"game": "12.25.2015.CLE.at.GSW", "date": "2015-12-25", "away": "CLE", "home": "GSW"}]
    vid = {"id": "7Tu", "verdict": "ok_unverified", "upload_date": "20260308"}
    parts = {"11.04.2015.NYK.at.CLE": {"broadcast": {"probed": [{**vid, "season_evidence": "season: 2015-2016"}]}},
             "11.13.2015.CLE.at.NYK": {"broadcast": {"probed": [{**vid, "season_evidence": "date: November 13, 2015"}]}},
             "12.23.2015.NYK.at.CLE": {"broadcast": {"probed": [{**vid, "season_evidence": "date: December 23 2015"}]}},
             "12.25.2015.CLE.at.GSW": {"broadcast": {"probed": [{"id": "QJL", "verdict": "ok_unverified", "upload_date": "20151226",
                                                                  "season_evidence": "uploaded 1 days after the game"}]}}}
    out = finalize(sched, parts, sched)
    assert out["11.04.2015.NYK.at.CLE"]["verdict"] == "no_broadcast"       # the video is dated for other games
    assert out["11.13.2015.CLE.at.NYK"]["verdict"] == "ambiguous"          # same video, two dated games
    assert out["12.23.2015.NYK.at.CLE"]["verdict"] == "ambiguous"
    assert out["12.25.2015.CLE.at.GSW"]["verdict"] == "ok_unverified"      # no other CLE-GSW game in the window
    sched2 = sched + [{"game": "12.10.2015.GSW.at.CLE", "date": "2015-12-10", "away": "GSW", "home": "CLE"}]
    assert finalize(sched[3:], parts, sched2)["12.25.2015.CLE.at.GSW"]["verdict"] == "ambiguous"


def test_rejects_simulations_and_other_games():
    from sportvu.manifest import candidate_rejection
    d = "2015-12-25"
    assert candidate_rejection("NBA2K16-(NBA Rewind 12/25/15) Cavaliers Vs Warriors", "Klassic Rah", d) == "video-game simulation"
    assert candidate_rejection("Cavaliers vs Warriors | Full Game | December 25, 2015", "NBA 2K SIMULATION", d) == "video-game simulation"
    assert candidate_rejection("Cavaliers vs Warriors 2016 NBA Finals Game 7", None, d) == "postseason game"
    assert candidate_rejection("Chicago Bulls vs Golden State Warriors November 20 1991", None, d) == "year 1991"
    assert candidate_rejection("Cleveland Cavaliers VS Utah Jazz 1/10/17", None, d) == "another date"
    assert candidate_rejection("Atlanta Hawks vs Cleveland Cavaliers - Live Stream [23/05/2015]", None, d) == "another date"
    assert candidate_rejection("Cavs vs Warriors Xmas Game", "Greg Smith", d) is None
    assert candidate_rejection("Cavaliers vs Warriors Christmas 2015 NBA Finals Rematch", None, d) is None
    assert candidate_rejection("2016 NBA Western Conference Semifinals: Thunder vs. Spurs", None, d) == "postseason game"
    assert candidate_rejection("2015-2016 NBA Season Golden State Warriors vs Cleveland Cavaliers", None, d) is None
    assert candidate_rejection("Golden State Warriors, Cleveland Cavaliers (12/25/2015)", None, d) is None
    assert candidate_rejection("Warriors vs Cavaliers 25-12-2015 full game", None, d) is None


def test_human_override():
    sched = [{"game": "12.28.2015.SAC.at.GSW", "date": "2015-12-28", "away": "SAC", "home": "GSW"}]
    parts = {"12.28.2015.SAC.at.GSW": {"broadcast": {"probed": [{"id": "qz", "verdict": "ok_unverified", "upload_date": "20250103",
                                                                  "season_evidence": "season: 2015 / 2016"}]}}}
    assert finalize(sched, parts, sched)["12.28.2015.SAC.at.GSW"]["verdict"] == "ambiguous"
    ov = {"12.28.2015.SAC.at.GSW": {"video_id": "qz", "by": "Lucien", "date": "2026-10-04", "note": "watched",
                                    "probe": {"vcodec": "avc1.640020", "height": 720, "duration": 6252}}}
    out = finalize(sched, parts, sched, ov)["12.28.2015.SAC.at.GSW"]
    assert out["verdict"] == "ok_unverified" and evidence_kind(out["best"]["season_evidence"]) == "human"
    ov["12.28.2015.SAC.at.GSW"]["probe"]["height"] = 360
    assert finalize(sched, parts, sched, ov)["12.28.2015.SAC.at.GSW"]["verdict"] == "low_res"


def test_override_reject():
    sched = [{"game": "12.25.2015.CLE.at.GSW", "date": "2015-12-25", "away": "CLE", "home": "GSW"}]
    parts = {"12.25.2015.CLE.at.GSW": {"broadcast": {"probed": [{"id": "QJL", "verdict": "ok_unverified", "upload_date": "20151226",
                                                                  "season_evidence": "uploaded 1 days after the game"}]}}}
    ov = {"12.25.2015.CLE.at.GSW": {"reject": "video-game simulation", "video_id": "QJL"}}
    out = finalize(sched, parts, sched, ov)["12.25.2015.CLE.at.GSW"]
    assert out["verdict"] == "no_broadcast" and out["note"].startswith("rejected")
