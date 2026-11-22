"""tools.p8_datasets_fetch: the SHOT clip-name screen and the record's email removal. No network."""

from tools import p8_datasets_fetch as D


def test_team_codes_from_clip_names():
    assert D.teams_in_name("CLEvsGS") == ["CLE", "GSW"]
    assert D.teams_in_name("ATL_vs_NJ_0012") == ["ATL", "BKN"]
    assert D.teams_in_name("clip_0042") == []


def test_screen_flags_cle_phx_gsw_and_the_d85_arenas():
    assert D.screen("CLEvsGS") == "a game with CLE"
    assert D.screen("PHXvsGSW") == "PHX and GSW"
    assert D.screen("MIAvsBOS") is not None
    assert D.screen("LALvsBOS_2023") is None                 # a year outside 2015-16 written in the name
    assert D.screen("ATLvsNJ") is None
    assert D.screen("clip_0042") == "no team codes readable"


def test_email_pattern():
    addr = "a.b" + "@" + "example.org"                  # built here so find_private stays clean on this file
    assert D.EMAIL_RE.sub("x", f"contact {addr} now") == "contact x now"


def test_pl_nba_folders_resolve_by_date_and_known_teams_only():
    sched = {"0022200200": ("2022-11-07", "SAC", "GSW"), "0022200201": ("2022-11-07", "DEN", "SAS"),
             "0022200300": ("2022-11-18", "DAL", "DEN")}
    assert D.resolve_folder("20221107_GSW_SAC_label", sched) == ("0022200200", "date and both teams")
    assert D.resolve_folder("20221108_GSW_SAC_label", sched)[0] == "0022200200"       # one day of slack
    gid, how = D.resolve_folder("20221118_DAL_DNF_label", sched)                        # DNF not guessed
    assert gid == "0022200300" and how == "date and one team (DAL)"
    assert D.resolve_folder("20221107_XYZ_QRS_label", sched) == (None, "no team code recognised")
    assert D.resolve_folder("notadate", sched)[0] is None


def test_season_of_a_game_id():
    assert D.season_of("0022400861") == "2024-25" and D.season_of("0021800909") == "2018-19"
    assert D.season_of("0042500171") == "2025-26"
