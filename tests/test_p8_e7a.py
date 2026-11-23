"""tools.p8_e7a: sampling by time, the scorecard's ball counting, F1 and the bootstrap, on synthetic data. No model,
no frames."""

from tools import p8_e7a as E


def test_sampling_keeps_frames_at_least_the_spacing_apart():
    rows = [{"frame": f} for f in (5, 60, 124, 125, 300, 419, 420)]
    assert [r["frame"] for r in E.sample_by_time(rows, 120)] == [5, 125, 300, 420]


def test_counting_follows_the_scorecard_rule():
    fr = [{"in_image": True, "uv": [100, 100], "pred": [110, 100]},      # TP at 32 px
          {"in_image": True, "uv": [100, 100], "pred": [200, 100]},      # far: FP and FN
          {"in_image": True, "uv": [100, 100], "pred": None},            # FN
          {"in_image": False, "uv": [-50, 100], "pred": [10, 10]},       # FP (ball out of the image)
          {"in_image": False, "uv": [-50, 100], "pred": None}]           # nothing
    assert E.count(fr, 32.07) == {"tp": 1, "fp": 2, "fn": 2}
    m = E.prf(E.count(fr, 32.07))
    assert abs(m["precision"] - 1 / 3) < 1e-9 and abs(m["recall"] - 1 / 3) < 1e-9 and abs(m["f1"] - 1 / 3) < 1e-9
    assert E.count(fr, 5.0) == {"tp": 0, "fp": 3, "fn": 3}


def test_bands_and_bootstrap():
    assert [E.z_band(z) for z in (0.5, 4.0, 9.9, 10.0)] == ["low", "mid", "mid", "high"]
    groups = [[1], [2], [3]]
    a = E.boot(groups, lambda gs: sum(x for g in gs for x in g) / len(gs), n=200, seed=1)
    assert a == E.boot(groups, lambda gs: sum(x for g in gs for x in g) / len(gs), n=200, seed=1)
    assert 1 <= a[0] <= a[1] <= 3
    assert E.boot([[1]], lambda gs: None, n=10) is None


def test_registered_constants():
    assert E.GAMES == ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015") and E.APART == ("hou_cle_2016",)
    assert E.SPACING_FRAMES == 120 and E.SEGMENT_TRIPLETS == 4 and E.TOL_PX == 32.07
