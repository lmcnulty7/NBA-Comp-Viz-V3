"""tools.p8_nba_identity: clock parsing, score changes in the tools.p8_scorebug event form, the fixed calibration.
No OCR, no network, no mirror download."""

from tools import p8_identity as I
from tools import p8_nba_identity as N


def test_iso_clock():
    assert N.clock_s("PT11M45.00S") == 705.0 and N.clock_s("PT00M04.30S") == 4.3 and N.clock_s("bad") is None


def test_score_changes_keep_changes_only_in_away_home_order():
    rows = [(1, 720.0, 0, 0), (1, 700.0, 0, 0), (1, 690.0, 2, 0), (1, 680.0, 2, 3), (1, 670.0, 2, 3),
            (2, 720.0, 2, 3), (2, 600.0, float("nan"), 3), (2, 500.0, 4, 3)]
    ev = N.score_changes(rows)
    assert ev == [(1, 720.0, "0-0"), (1, 690.0, "0-2"), (1, 680.0, "3-2"), (2, 500.0, "3-4")]
    assert I.frame_matches(I.parse_tokens(["2ND", "8:00", "3", "4"]), ev)
    assert not I.frame_matches(I.parse_tokens(["2ND", "8:30", "3", "4"]), ev)


def test_calibration_is_fixed():
    assert len(N.CALIBRATION) == 5 and len({g for _, g in N.CALIBRATION}) == 5
    assert all(g.startswith("00225") for _, g in N.CALIBRATION)
