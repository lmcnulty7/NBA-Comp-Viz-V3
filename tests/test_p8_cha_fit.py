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


def synthetic_frames(n=30, pairs=8, seed=1, C=(47.0, 125.0, -30.0)):
    """Frames seen by one pinhole camera: V3-like H from the true pose, feet projected without noise."""
    import numpy as np
    from sportvu.camera import h_px2ft, project
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        rv = np.array([1.9, 0.0, 0.0]) + rng.normal(0, 0.03, 3)
        f = float(rng.uniform(1500, 2500))
        ft = np.column_stack([rng.uniform(5, 89, pairs), rng.uniform(5, 45, pairs)])
        px = project(C, rv, f, 1280, 720, ft)
        out.append({"window": f"gsw_cha_2016_s01_w{k // 10:02d}", "frame": k, "w": 1280, "h": 720,
                    "H_v3": h_px2ft(C, rv, f, 1280, 720).ravel().tolist(), "H_b4": None, "px": px, "ft": ft,
                    "boxes": [], "row": {}})
    return out


def test_the_tools_side_copy_reproduces_fit_game():
    import numpy as np
    from sportvu.camera import fit_game
    fr = synthetic_frames()
    a, b = fit_game(fr, seed=0), C.pooled_ls(fr, seed=0)
    assert a["status"] == "ok" and np.allclose(a["C"], b["C"], atol=1e-9) and a["inits"] == b["inits"]
    assert b["nfev"] >= 1 and b["pooled_frames"] == len(fr)
    moved = C.pooled_ls(fr, seed=0, start_C=(47.0, 120.0, -30.0))
    assert moved["start_C"] == [47.0, 120.0, -30.0]
    assert C.pooled_ls(fr[:10])["status"] == "too_few_inits"


def test_leave_and_good_and_pair_key():
    import numpy as np
    assert C.leaves_start({"C": [0, 5, 0], "C0": [0, 0, 0]}) and not C.leaves_start({"C": [0, 4.9, 0], "C0": [0, 0, 0]})
    assert not C.leaves_start({"status": "too_few_inits"})
    assert C.good({"reproj_px_median_after": 10.0}) and not C.good({"reproj_px_median_after": 10.01})
    a = {"px": np.array([[1.0, 2.0]]), "ft": np.array([[3.0, 4.0]])}
    assert C.pair_key(a) == C.pair_key({"px": a["px"].copy(), "ft": a["ft"].copy()})
    assert C.pair_key(a) != C.pair_key({"px": a["px"] + 1, "ft": a["ft"]})


def run(leave, good_):
    return {"leaves_start": leave, "good": good_}


def test_readings_follow_the_registered_rules():
    stuck, ok = run(False, False), run(True, True)
    f2_h2 = {"from_C_old": {"C": list(C.C_OLD), "cost": 1.0}, "from_C0": {"C": [47, 104, -34], "cost": 2.0}}
    f2_no = {"from_C_old": {"C": [47, 104, -34], "cost": 2.0}, "from_C0": {"C": [47, 104, -34], "cost": 2.0}}
    f3_ok, f3_stuck = {"old_inits_only": ok, "no_post_reset": ok}, {"old_inits_only": stuck, "no_post_reset": stuck}
    r = C.readings({"fix": [stuck] * 9 + [ok], "nofix": [ok] * 10}, f2_h2, f3_stuck, {"share_changed": 0.0})
    assert r["F1"]["reading"].startswith("H1") and r["F2"]["reading"].startswith("H2")
    assert r["F3"]["reading"].startswith("the added frames are not") and r["F4"]["reading"] == "not H5"
    r = C.readings({"fix": [stuck] * 10, "nofix": [ok] * 10}, f2_no, f3_ok, {"share_changed": 0.2})
    assert r["F1"]["reading"].startswith("the fixed init set holds") and r["F3"]["reading"].startswith("H3")
    assert r["F2"]["reading"].startswith("the pool prefers") and r["F4"]["reading"].startswith("H5")
    r = C.readings({"fix": [stuck] * 10, "nofix": [stuck] + [ok] * 9}, f2_no, f3_ok, {"share_changed": None})
    assert r["F1"]["reading"].startswith("the fit is fragile on both")


def test_stage1_guards_every_game_it_reads_before_reading(monkeypatch):
    import pytest
    seen = {}

    def fake_guard(items, who):
        seen["guard"] = list(items)
        raise SystemExit("stop after the guard")
    monkeypatch.setattr(C, "refuse_cv_test", lambda items, who: seen.setdefault("refuse", list(items)))
    monkeypatch.setattr(C, "guard", fake_guard)
    monkeypatch.setattr(C, "roots", lambda: (_ for _ in ()).throw(AssertionError("read before the guard")))
    with pytest.raises(SystemExit, match="after the guard"):
        C.stage1()
    want = {"gsw_cha_2016", "gsw_bkn_2015", "gsw_sac_2015", "hou_cle_2016", "gsw_ind_2016"}
    assert set(seen["refuse"]) == want and set(seen["guard"]) == want
