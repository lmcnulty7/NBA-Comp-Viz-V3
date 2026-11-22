"""tools.p8_nba2526_fetch: the download check. No network."""

from tools import p8_nba2526_fetch as F


def test_check_needs_h264_720_and_duration_within_two_seconds():
    good = {"ok": True, "duration_s": 3456.4}
    assert F.check_facts(good, 3456)
    assert F.check_facts(good, 3458.4)
    assert not F.check_facts(good, 3459)
    assert not F.check_facts({"ok": False, "duration_s": 3456.0}, 3456)
    assert not F.check_facts(good, None)
