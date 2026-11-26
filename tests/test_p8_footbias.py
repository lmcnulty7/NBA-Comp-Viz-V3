"""tools.p8_footbias: the pure rules of the foot-bias measurement on synthetic data."""

import numpy as np

from tools import p8_footbias as M


def test_constants_agree_with_the_scorer():
    from sportvu import eval as E
    assert M.GHOST_FT == E.GHOST_FT and M.MISS_FT == E.MISS_FT


def test_court_image_round_trip_and_scorer_path():
    H = np.array([[0.1, 0.01, -3.0], [0.0, 0.12, -5.0], [0.0, 0.0001, 1.0]])
    q = M.to_court(H, [[100.0, 200.0], [640.0, 400.0]])
    assert np.allclose(M.to_image(H, q), [[100.0, 200.0], [640.0, 400.0]], atol=1e-6)
    assert np.allclose(M.scorer_court(H, [[100.0, 200.0]]), q[:1], atol=1e-3)


def test_score_frame():
    sv = np.array([[0.0, 0.0], [10.0, 0.0]] + [[100.0, 100.0]] * 8)
    inside = np.array([True, True] + [False] * 8)
    ct = np.array([[1.0, 0.0], [50.0, 50.0], [50.0, 51.0]])
    out = M.score_frame(ct, sv, inside, [False, False, True])
    assert out == {"in_frame": 2, "missed": 1, "nr_ghosts": 1, "nr_boxes": 2}
    assert M.score_frame(np.zeros((0, 2)), sv, inside, [])["missed"] == 2


def test_cell_consistency():
    g = M.POOL
    med = {"a": dict(zip(g, (-1.0, -1.2, -0.9))), "b": dict(zip(g, (-1.0, 0.2, -0.9))),
           "c": dict(zip(g, (-1.0, -2.0, -0.5)))}
    cnt = {k: dict(zip(g, (300, 300, 300))) for k in med}
    pooled = {"a": -1.0, "b": -0.8, "c": -1.0}
    assert M.cell_consistency(med, cnt, pooled) == {"a": True, "b": False, "c": False}
    cnt["a"][g[0]] = 199
    assert M.cell_consistency(med, cnt, pooled)["a"] is False


def test_worth_arm_and_tables():
    good = {g: {"net": -0.012, "ghost": -0.001} for g in M.POOL}
    bad = dict(good, **{M.POOL[1]: {"net": -0.012, "ghost": 0.001}})
    assert M.worth_arm(True, {"K1": good, "K2": bad}) == ["K1"]
    assert M.worth_arm(False, {"K1": good}) == []
    rows = [{"k": "x", "v": float(i)} for i in range(60)] + [{"k": "y", "v": 5.0}]
    t = M.median_table(rows, lambda r: r["k"], "v", 50)
    assert t["x"] == 29.5 and t["y"] == 0.0


def test_corrected_px_k2_moves_feet_down_by_ratio():
    fr = {"px": np.array([[100.0, 300.0]]), "bh": np.array([180.0]), "Hb": None, "Gb": None, "nhb": None}
    out = M.corrected_px(fr, "K2", {"150-200": 0.05})
    assert np.allclose(out, [[100.0, 309.0]])
    assert np.allclose(M.corrected_px(fr, "K1", {}), fr["px"])
