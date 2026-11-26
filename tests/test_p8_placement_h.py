"""tools.p8_placement_h: the pure rules of placement step 1b on synthetic data."""

import numpy as np

from tools import p8_placement_h as M


def test_to_court_and_moved_in():
    H = np.diag([0.1, 0.1, 1.0])
    assert np.allclose(M.to_court(H, [[10.0, 20.0]]), [[1.0, 2.0]])
    assert M.to_court(H, np.zeros((0, 2))).shape == (0, 2)
    e = np.array([[3.0, 0.0], [3.0, 0.0]])
    d = np.array([[-3.0, 0.0], [1.0, 0.0]])
    assert M.moved_in(e, d).tolist() == [True, False]


def test_corr_slope():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    c, s = M.corr_slope(x, 2 * x + 1)
    assert abs(c - 1) < 1e-12 and abs(s - 2) < 1e-12
    assert M.corr_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == (None, None)


def test_permuted_share_breaks_the_pairing():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 3, (400, 2)) + np.array([3.5, 0.0])
    d = -e + rng.normal(0, 0.2, (400, 2))          # d_H undoes e_T exactly, up to noise
    assert M.moved_in(e, d).mean() > 0.99
    assert M.permuted_share(e, d, ["near"] * 400) < 0.3


def test_read_rule():
    g = {"a": {"share": 0.6, "excess": 0.3}, "b": {"share": 0.55, "excess": 0.26}}
    assert M.read_rule(g) == "GEOMETRY"
    assert M.read_rule({"a": {"share": 0.1, "excess": 0.05}, "b": {"share": 0.2, "excess": 0.1}}) == "FOOT PIXEL"
    assert M.read_rule({"a": {"share": 0.6, "excess": 0.3}, "b": {"share": 0.2, "excess": 0.05}}) == "MIXED"
    assert M.read_rule({"a": {"share": 0.6, "excess": 0.2}, "b": {"share": 0.6, "excess": 0.2}}) == "MIXED"
