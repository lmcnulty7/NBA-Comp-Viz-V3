"""sportvu.compare: which build is better on a row."""
from sportvu.compare import better


def test_lower_is_better():
    assert better({"v3": 7.3, "public": 24.7}, True) == "v3"


def test_bias_is_judged_by_distance_from_zero():
    assert better({"v3": -1.0, "public": 0.4}, "abs") == "public"


def test_higher_is_better_and_ties_and_missing():
    assert better({"v3": 98.6, "public": 99.2}, False) == "public"
    assert better({"v3": 25.1, "public": 25.1}, True) is None
    assert better({"v3": 25.1, "public": None}, True) is None
