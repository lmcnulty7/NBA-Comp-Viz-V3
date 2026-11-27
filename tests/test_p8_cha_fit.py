"""tools.p8_cha_fit: the pure parts of the stage 1 report (docs/research/P8_cha_fit.md). No video, no model."""

from tools import p8_cha_fit as C


def test_first_reset_and_side():
    assert C.first_reset(None) is None and C.first_reset({"resets": []}) is None
    log = {"resets": [{"frame": 6534, "stripped": 2}, {"frame": 312, "stripped": 2}, {"frame": None, "stripped": 2}]}
    assert C.first_reset(log) == 312
    assert C.side_of(311, 312) == "before" and C.side_of(312, 312) == "after" and C.side_of(5, None) == "before"


def test_stats_and_distance():
    s = C.stats([1, 2, 3, None, 4])
    assert s["n"] == 4 and s["median"] == 2.5 and s["p25"] == 1.75
    assert C.stats([]) == {"n": 0}
    assert C.distance_to([3, 4, 0], [[0, 0, 0], [0, 0, 0], [10, 10, 10]]) == 5.0
    assert C.distance_to([1, 1, 1], []) is None


def test_only_the_named_game_and_reference_fits():
    assert C.GAME == "gsw_cha_2016" and "gsw_ind_2016" in C.REFERENCE_ONLY and C.MIN_BOXES == 6
