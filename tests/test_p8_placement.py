"""tools.p8_placement: the pure rules of placement step 1 on synthetic data."""

import numpy as np
import pytest

from tools import p8_placement as M


def test_constants_agree_with_the_scorer():
    from sportvu import eval as E
    from sportvu.bench import MATCH_FT
    assert M.MISS_FT == E.MISS_FT and M.MARGIN == E.FRAME_MARGIN_PX and M.NEAR_FT == MATCH_FT


def test_depth_basis_and_classes():
    u, v = M.depth_basis([0.0, 100.0], [0.0, 50.0])
    assert np.allclose(u, [0.0, -1.0]) and np.allclose(v, [1.0, 0.0])
    assert [M.klass(x) for x in (2.5, 2.6, 5.0, 5.1, np.inf)] == ["hit", "placement", "placement", "far", "far"]
    assert [M.hbin(x) for x in (100, 150, 249, 300, None)] == ["<150", "150-200", "200-250", ">=250", "no pose"]


def test_nearest_foot():
    idx, d = M.nearest_foot([[0.0, 0.0], [10.0, 0.0]], [[1.0, 0.0], [9.0, 0.0]])
    assert idx.tolist() == [0, 1] and np.allclose(d, [1.0, 1.0])
    idx, d = M.nearest_foot([[0.0, 0.0]], np.zeros((0, 2)))
    assert idx.tolist() == [-1] and np.isinf(d).all()


def test_shift_feet_removes_a_depth_offset():
    ground = [0.0, 100.0]
    truth = np.array([[0.0, 50.0], [20.0, 50.0]])
    feet = truth.copy()
    for i in range(2):                      # put each foot 3 ft beyond its player, away from the camera
        u, _ = M.depth_basis(ground, truth[i])
        feet[i] = truth[i] + 3.0 * u
    out = M.shift_feet(feet, ground, ["near", "near"], {"near": 3.0})
    assert np.allclose(out, truth, atol=0.15)
    lat = M.shift_feet(truth, ground, ["far"], {}, {"far": 1.0})
    u, v = M.depth_basis(ground, truth[0])
    assert np.allclose(lat[0], truth[0] - v)


def test_lag_slope_and_choose_lag():
    s = np.array([5.0, 10.0, 20.0])
    assert M.lag_slope(0.2 * s, s) == pytest.approx(0.2)
    assert M.lag_slope([], []) is None
    assert M.choose_lag({-2: 0.2, -1: 0.1, 0: 0.15, 1: 0.1, 2: 0.3}) == -1
    assert M.choose_lag({-1: 0.2, 0: 0.2, 1: 0.2}) == 0


def test_reading_rules():
    good = {"g1": {"net": -0.02, "gross_share": 0.2}, "g2": {"net": -0.011, "gross_share": 0.1},
            "g3": {"net": -0.03, "gross_share": 0.5}}
    weak = dict(good, g2={"net": -0.009, "gross_share": 0.3})
    assert M.read_systematic({"C1": good, "C2": weak}, True) == ["C1"]
    assert M.read_systematic({"C1": good}, False) == []
    assert M.read_timing({"a": {"lag": 2, "change": -0.02}, "b": {"lag": -1, "change": -0.01}})
    assert not M.read_timing({"a": {"lag": 0, "change": -0.02}, "b": {"lag": -1, "change": -0.01}})
    assert not M.read_timing({"a": {"lag": 2, "change": -0.005}})


def test_sign_rule():
    def off(m, iv):
        return {"placement": {"depth_median": m, "depth_median_interval": iv}}
    ok = {g: off(1.0, [0.5, 1.5]) for g in M.POOL}
    assert M.sign_rule(ok)
    mixed = dict(ok, gsw_cha_2016=off(-1.0, [-1.5, -0.5]))
    assert not M.sign_rule(mixed)
    crosses = dict(ok, gsw_sac_2015=off(0.3, [-0.1, 0.7]))
    assert not M.sign_rule(crosses)


def test_boot_median_and_guard():
    iv = M.boot_median({"a": [1.0, 2.0], "b": [3.0], "c": []}, n=50)
    assert iv[0] <= iv[1]
    with pytest.raises(SystemExit):
        M.check_games(["gsw_ind_2016"])
