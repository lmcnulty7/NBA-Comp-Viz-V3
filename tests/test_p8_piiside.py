"""tools.p8_piiside (docs/research/P8_piiside.md 1.1, 1.2 and 11.6): velocity and along offsets, the window tau, the
null and its power, the readings, the camera ratio, the refusal and the writes. Synthetic data only."""

import numpy as np
import pytest

from tools import p8_piiside as M


def test_velocity_and_along():
    v = M.velocity([0.0, 0.0], [4.0, 3.0], 100.0, 99.6)          # 5 ft in 0.4 s
    assert np.allclose(v, [10.0, 7.5]) and np.hypot(*v) == pytest.approx(12.5)
    assert M.velocity([0, 0], [1, 1], 99.0, 99.0) is None
    assert M.along([2.5, 0.0], [10.0, 0.0]) == pytest.approx(0.25)   # 2.5 ft ahead at 10 ft/s is 0.25 s
    assert M.along([0.0, 3.0], [10.0, 0.0]) == pytest.approx(0.0)
    assert M.along([1, 1], [0, 0]) is None


def mk_rows(n, tau, speed=10.0, noise=0.0, rng=None, third="middle"):
    rng = rng or np.random.default_rng(0)
    rows = []
    for _ in range(n):
        e = rng.normal(0, noise, 2) if noise else np.zeros(2)
        o = np.array([tau * speed, 0.0]) + e
        rows.append({"speed": speed, "v": [speed, 0.0], "o_raw": list(o), "third": third,
                     "a_raw": M.along(o, [speed, 0.0])})
    for _ in range(12):                                         # slow rows carry the window's noise
        e = rng.normal(0, noise, 2) if noise else np.zeros(2)
        rows.append({"speed": 2.0, "v": [2.0, 0.0], "o_raw": list(e), "third": third, "a_raw": M.along(e, [2.0, 0.0])})
    return rows


def test_window_tau_needs_thirty_moving_rows():
    assert M.window_tau(mk_rows(30, 0.05)) == pytest.approx(0.05)
    assert M.window_tau(mk_rows(29, 0.05)) is None


def test_null_without_noise_has_no_spread_and_injection_is_caught():
    wins = {f"w{i}": {"rows": mk_rows(40, 0.05)} for i in range(12)}
    null = M.simulate_null(wins, 0.05, {}, n=50)
    assert np.allclose(null, 0.0)
    inj = M.simulate_null(wins, 0.05, {}, n=50, inject=(0.5, 0.1, 1))
    assert (inj > 0.0).mean() > 0.9
    # the IQR is robust to a minority of windows: a quarter injected often leaves it at 0 (flagged in the report)
    quarter = M.simulate_null(wins, 0.05, {}, n=200, inject=(0.25, 0.1, 1))
    assert (quarter > 0.0).mean() < 0.6


def test_null_falls_back_to_the_games_third_pool():
    rows = mk_rows(40, 0.0, noise=0.0)
    rows = [r for r in rows if r["speed"] >= M.MOVING]          # no slow rows in the window
    null = M.simulate_null({"w0": {"rows": rows}, "w1": {"rows": rows}}, 0.0, {"middle": [[1.0, 0.0]] * 5}, n=10)
    assert np.allclose(null, 0.0)                               # the same fallback offset in both windows


def test_timing_reading_branches():
    null = np.zeros(100)
    good = {"value": 2.0, "interval": [1.2, 3.0]}
    assert M.timing_reading(0.0, null, 0.9, True, 6, good)["reading"].startswith("answered")
    assert "B is not independent" in M.timing_reading(0.0, null, 0.9, True, 6, good)["reading"]
    assert M.timing_reading(0.0, null, 0.5, True, 6, good)["reading"].startswith("not resolvable at this noise")
    assert M.timing_reading(0.0, null, 0.9, False, 6, good)["reading"].startswith("not resolvable")
    assert M.timing_reading(0.2, null, 0.9, True, 4, good)["reading"].startswith("not resolvable (fewer than 5")
    assert M.timing_reading(0.2, null, 0.9, True, 6, good)["reading"].startswith("timing found")
    weak = {"value": 1.6, "interval": [0.9, 2.5]}
    assert M.timing_reading(0.2, null, 0.9, True, 6, weak)["reading"].startswith("timing not shown")


def test_weighted_median_and_camera_ratio():
    assert M.weighted_median([1.0, 2.0, 10.0], [1, 1, 5]) == 10.0
    assert M.weighted_median([], []) is None
    own = [{"third": "near", "d_TB": 2.0}] * 3 + [{"third": "far", "d_TB": 6.0}]
    hits = [{"third": "near", "d_TB": 1.0}] * 3 + [{"third": "far", "d_TB": 2.0}]
    assert M.camera_ratio(own, hits) == pytest.approx(2.0)       # near carries three of the four own rows


def test_boot_ratio_and_spearman():
    r = M.boot_ratio([(10, 100), (12, 100)], [(5, 100), (5, 100), (6, 100)], n=200)
    assert r["value"] == pytest.approx((22 / 200) / (16 / 300)) and r["interval"][0] > 1
    assert M.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert M.spearman([1, 2], [1, 2]) is None


def test_readings_are_asymmetric():
    t_found = {"timing": {"reading": "timing found (read conservatively)"}, "camera": {"holds": False}}
    t_ans = {"timing": {"reading": "answered: no window timing error beyond B's noise; " + M.LIMIT},
             "camera": {"holds": False}}
    r = M.readings({M.CHA: t_found, M.BKN: t_found})
    assert r["timing"].startswith("timing found") and r["overall"] == "a truth-side pattern found"
    r = M.readings({M.CHA: t_ans, M.BKN: t_ans})
    assert "never read as evidence that the row is build-side" in r["overall"]
    assert "B is not independent" in r["camera"]


def test_hou_is_refused_before_any_read(monkeypatch):
    import tools.p8_rebaseline as RB
    monkeypatch.setattr(RB, "read_json", lambda p: (_ for _ in ()).throw(AssertionError("a file was read")))
    with pytest.raises(SystemExit):
        M.collect("hou_cle_2016")
    called = []
    monkeypatch.setattr(M, "collect", lambda g: called.append(g))
    with pytest.raises(SystemExit):
        M.main(["collect", "hou_cle_2016"])
    assert called == []


def test_the_only_writes_are_the_records():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(M))
    names = [getattr(c.func, "id", getattr(c.func, "attr", None)) for c in ast.walk(tree) if isinstance(c, ast.Call)]
    assert names.count("atomic_write") == 2
    assert not any(n in ("write_text", "write_bytes", "unlink", "rmtree", "remove", "rename", "open") for n in names)


def test_constants_match_the_design():
    assert (M.MOVING, M.SLOW_LO, M.MIN_MOVING_ROWS, M.MIN_SLOW_WINDOW) == (5.0, 1.0, 30, 10)
    assert (M.OFF_S, M.MIN_OFF_WINDOWS, M.ENRICH, M.CAM_RATIO) == (0.1, 5, 1.5, 1.5)
    assert (M.SHIFT_MOMENTS, M.SHIFT_TOL_S, M.POWER_BAR, M.POWER_SHIFT_S, M.POWER_FRAC) == (3, 0.02, 0.8, 0.1, 0.25)
    assert (M.N_SIM, M.SEED, M.SEED_POWER, M.LATERAL_MIN) == (2000, 0, 1, 5.0)
