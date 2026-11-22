"""tools.p8_baseline: the pooled ratio and its game bootstrap, on synthetic counts."""

import pytest

from tools import p8_baseline as PB


def test_pooled_ratio_sums_counts_and_bootstraps_games():
    out = PB.pooled_ratio({"a": (10, 100), "b": (30, 100), "c": (20, 100)})
    assert out["value"] == pytest.approx(0.2) and out["num"] == 60 and out["den"] == 300
    lo, hi = out["game_bootstrap"]
    assert 0.1 <= lo <= 0.2 <= hi <= 0.3


def test_one_game_has_no_game_bootstrap():
    assert PB.pooled_ratio({"a": (1, 4)})["game_bootstrap"] is None


def test_the_tool_refuses_cv_test_items_and_never_pools_the_d98_game():
    src = (PB.__file__ and open(PB.__file__).read())
    assert "refuse_cv_test(" in src
    with pytest.raises(SystemExit):
        PB.main(["--tags", "gsw_bkn_2015,gsw_ind_2016"])
