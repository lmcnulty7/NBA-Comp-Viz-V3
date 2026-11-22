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
    assert D.EMAIL_RE.sub("x", "contact a.b@example.org now") == "contact x now"
