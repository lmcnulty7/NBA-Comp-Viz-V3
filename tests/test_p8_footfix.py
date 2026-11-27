"""tools.p8_footfix (docs/research/P8_footfix.md version 1.1): fits, corrections, the copy check, fit rows, the rule
and the readings, on synthetic data. No video, no model, no workspace."""

import json

import numpy as np
import pytest

from tools import p8_footfix as M

W, H = 1280, 720
CAM_C = np.array([47.2, 137.1, -30.7])
CAM_RV = np.array([-1.345, -0.165, -0.267])
CAM_F = 4300.0


def camera_h():
    from sportvu.camera import h_px2ft
    return np.asarray(h_px2ft(CAM_C, CAM_RV, CAM_F, W, H), np.float64).reshape(3, 3)


def geom():
    from sportvu.camera import pose_from_h
    from sportvu.eval import near_side
    Hb = camera_h()
    pose = pose_from_h(Hb, W, H)
    assert pose is not None
    return Hb, np.asarray(pose[2][:2], float), near_side(Hb)


def box(foot, h=180.0, tid=1):
    x, y = foot
    return {"bbox": [x - 20, y - h, x + 20, y], "foot_raw": [x, y], "foot_corrected": [x, y],
            "foot_stab": [float(x), float(y)], "tid": tid}


# ── fits ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_median_table_needs_fifty_rows():
    rows = [{"k": "a", "v": float(i)} for i in range(50)] + [{"k": "b", "v": 1.0}] * 49 + [{"k": "c", "v": None}] * 60
    t, n = M.median_table(rows, lambda r: r["k"], "v")
    assert t == {"a": 24.5} and n == {"a": 50, "b": 49}


def test_hbin_edges():
    assert [M.hbin(v) for v in (149.9, 150.0, 199.9, 200.0, 250.0, 400.0)] == \
        ["<150", "150-200", "150-200", "200-250", ">=250", ">=250"]


def test_planted_ratio_is_recovered_and_removed_by_f2():
    rng = np.random.default_rng(0)
    rows = [{"hb": "150-200", "third": "middle", "d1": 0.0, "r2": 0.03 + rng.normal(0, 0.001), "stage": "A",
             "inlier": True} for _ in range(80)]
    t = M.fit_tables(rows)
    assert t["F2"]["150-200"] == pytest.approx(0.03, abs=5e-4)
    b = box((600.0, 400.0), h=180.0)
    out = M.corrected_feet([b], "F2", {"150-200": 0.03}, (None, None, None))
    assert out[0] == pytest.approx([600.0, 400.0 + 0.03 * 180.0])
    assert M.corrected_feet([b], "F2", {"<150": 0.03}, (None, None, None)) == [None]


def test_planted_depth_is_recovered_and_removed_by_f1():
    """A truth equal to B, feet planted 1.2 ft beyond each player along the camera ray: the fit rows' d1 recovers the
    shift and F1 brings the feet back onto the players."""
    from tools.p8_placement import depth_basis
    from tools.p8_scorecard import third_of
    Hb, Gb, nhb = geom()
    s = 1.2
    sv = M.to_court(Hb, [[150.0 + 100 * i, 300.0 + 35 * i] for i in range(10)])     # players spread over the frame
    q = np.array([p + s * depth_basis(Gb, p)[0] for p in sv])
    feet = M.to_image(Hb, q)
    t_px = M.to_image(Hb, sv)
    assert ((feet[:, 0] > 0) & (feet[:, 0] < W) & (feet[:, 1] > 0) & (feet[:, 1] < H)).all()
    ft = M.to_court(Hb, feet)
    rows = M.fit_rows_frame(feet, ft, sv, np.ones(10, bool), np.zeros(10, bool), [180.0] * 10, t_px,
                            (Hb, Gb, nhb), {101, 102}, np.arange(100, 110))
    assert len(rows) == 10
    assert all(r["d1"] == pytest.approx(s, abs=0.05) for r in rows)
    assert all(r["dB"] == pytest.approx(s, abs=0.05) for r in rows)         # T = B here, so the two agree
    assert sum(r["inlier"] for r in rows) == 2
    table = {M.cell("150-200", t): s for t in M.THIRDS}
    new = M.corrected_feet([box(p) for p in feet], "F1", table, (Hb, Gb, nhb))
    back = M.to_court(Hb, np.array(new))
    assert np.allclose(back, sv, atol=0.05)
    assert {third_of(p[1], nhb) for p in q} <= set(M.THIRDS)


def test_f1_without_a_pose_or_with_an_empty_cell_keeps_the_stored_feet():
    Hb, Gb, nhb = geom()
    b = box((600.0, 400.0))
    assert M.corrected_feet([b], "F1", {M.cell("150-200", "middle"): 1.0}, (None, None, None)) == [None]
    assert M.corrected_feet([b], "F1", {}, (Hb, Gb, nhb)) == [None]


# ── the corrected sidecar ──────────────────────────────────────────────────────────────────────────────────────────
def sidecar():
    Hb = camera_h()
    fr = [{"frame": f, "state": "TRACK", "H": Hb.ravel().tolist(), "accepted": True,
           "boxes": [box((500.0 + 10 * f, 400.0), 180.0, 1), box((700.0, 420.0), 120.0, 2)]} for f in range(3)]
    return {"source": "x.mp4", "fps": 59.94, "stride": 6, "frames": fr}


def test_identity_is_byte_equal_and_only_feet_move():
    side = sidecar()
    raw = json.dumps(side)
    new, moved = M.correct_sidecar(json.loads(raw), "F1", {}, W, H)
    assert moved == 0 and json.dumps(new) == raw
    new2, moved2 = M.correct_sidecar(json.loads(raw), "F2", {"150-200": 0.02}, W, H)
    assert moved2 == 3 and M.copy_problems(side, new2) == []
    assert new2["frames"][0]["boxes"][1]["foot_stab"] == side["frames"][0]["boxes"][1]["foot_stab"]


def test_copy_check_rejects_any_other_change():
    side = sidecar()
    for path in (("bbox",), ("tid",)):
        bad = json.loads(json.dumps(side))
        bad["frames"][1]["boxes"][0][path[0]] = [0, 0, 1, 1] if path[0] == "bbox" else 9
        assert M.copy_problems(side, bad)
    bad = json.loads(json.dumps(side)); bad["frames"][0]["H"][0] += 1e-9
    assert M.copy_problems(side, bad)
    bad = json.loads(json.dumps(side)); bad["frames"][0]["boxes"].pop()
    assert M.copy_problems(side, bad)
    bad = json.loads(json.dumps(side)); bad["stride"] = 3
    assert M.copy_problems(side, bad)


# ── fit rows ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_fit_rows_follow_d205s_main_split_and_keep_in_frame_players_only():
    sv = np.array([[10.0, 10.0], [30.0, 10.0], [50.0, 10.0], [70.0, 10.0]] + [[90.0, 45.0]] * 6)
    ft = np.array([[10.5, 10.0],       # player 0, stage A
                   [33.5, 10.0],       # player 1, stage B (3.5 ft)
                   [50.2, 10.0],       # a referee box on player 2: excluded
                   [70.1, 10.0]])      # player 3, out of frame: claims the box, gives no row
    inside = np.array([True, True, True, False] + [False] * 6)
    is_ref = np.array([False, False, True, False])
    px = np.array([[100.0, 300.0]] * 4)
    rows = M.fit_rows_frame(px, ft, sv, inside, is_ref, [160.0] * 4, np.array([[100.0, 305.0]] * 10),
                            (None, None, None), {7}, np.arange(10))
    assert [(r["stage"], r["inlier"]) for r in rows] == [("A", False), ("B", False)]
    assert all(r["third"] is None and r["d1"] is None and r["r2"] == pytest.approx(5.0 / 160.0) for r in rows)


def test_fold_rows_never_read_the_held_out_game():
    rows = {M.CHA: [{"w": "gsw_cha_2016_s00_w00"}], M.BKN: [{"w": "gsw_bkn_2015_s00_w00"}]}
    got, other = M.fold_rows(rows, M.CHA)
    assert other == M.BKN and got == rows[M.BKN]
    rows[M.BKN].append({"w": "gsw_cha_2016_s01_w00"})
    with pytest.raises(SystemExit):
        M.fold_rows(rows, M.CHA)
    with pytest.raises(SystemExit):
        M.fold_rows(rows, M.HOU)


def test_stages_s0_and_s1_refuse_hou_cle_2016():
    for st in ("s0", "s1"):
        with pytest.raises(SystemExit):
            M.check_stage_games(st, [M.HOU])
    M.check_stage_games("s0", list(M.POOL))
    with pytest.raises(SystemExit):
        M.check_stage_games("hou", [M.CHA])


# ── the rule and the readings ──────────────────────────────────────────────────────────────────────────────────────
def changes(**kw):
    base = {"missed": (-0.012, [-0.02, -0.004]), "ghosts_per_box": (0.002, [-0.003, 0.007]),
            "nearfield_bias_ft": (0.02, [-0.03, 0.08]), "position_p90_ft": (0.01, [-0.05, 0.07]),
            "near_p90_ft": (0.0, [-0.04, 0.04]), "team": (0.0, [-0.006, 0.006]),
            "missed_1to1": (-0.010, [-0.02, 0.0]), "position_p50_ft": (-0.02, [-0.05, 0.01])}
    base.update(kw)
    return {k: {"change": v[0], "interval": v[1]} for k, v in base.items()}


def test_guard_rows_read_harm_by_the_point_and_inconclusive_by_the_end():
    assert M.guard_row(0.004, [-0.002, 0.009], 0.010) == "pass"
    assert M.guard_row(0.004, [-0.002, 0.012], 0.010) == "inconclusive"
    assert M.guard_row(0.011, [0.005, 0.02], 0.010) == "harm"
    assert M.guard_row(-0.005, [-0.012, 0.0], -0.010, upper=False) == "inconclusive"
    assert M.guard_row(-0.011, [-0.02, 0.0], -0.010, upper=False) == "harm"


def test_game_rule():
    assert M.game_rule(changes())["passes"]
    r = M.game_rule(changes(missed=(-0.012, [-0.02, 0.001])))
    assert not r["passes"] and not r["rule1_missed"]
    r = M.game_rule(changes(position_p90_ft=(0.03, [-0.02, 0.11])))
    assert not r["passes"] and r["inconclusive"] == ["position_p90_ft"] and r["harm"] == []
    r = M.game_rule(changes(near_p90_ft=(0.06, [0.01, 0.1])))
    assert r["harm"] == ["near_p90_ft"]
    assert not M.game_rule(changes(missed_1to1=(-0.005, [-0.01, 0.0])))["passes"]


def test_rule7():
    assert M.rule7(-0.0075, -0.03)                 # the absolute bar
    assert M.rule7(-0.009, -0.012)                 # three quarters of the missed fall
    assert not M.rule7(-0.007, -0.012)
    assert not M.rule7(None, -0.02)


def test_g0_reads_points_only():
    assert M.g0_game(changes(missed=(-0.010, [-0.03, 0.01])))["passes"]
    assert not M.g0_game(changes(missed=(-0.009, [-0.03, -0.001])))["passes"]
    assert not M.g0_game(changes(team=(-0.011, [-0.02, 0.0])))["passes"]


def split(inl_m, non_m, inl_p=-0.01, non_p=-0.01):
    return {"inlier": {"missed": {"change": inl_m}, "position_p50_ft": {"change": inl_p}},
            "non_inlier": {"missed": {"change": non_m}, "position_p50_ft": {"change": non_p}}}


def test_readings():
    ok = {"passes": True, "harm": []}
    per = {g: {"rule": ok, "p50": -0.01, "split": split(-0.01, -0.02)} for g in M.POOL}
    assert M.reading(per) == {"verdict": "PASS", "reading": "feet nearer", "p50_under_B_not_rising": True,
                              "copying_T": False, "stage_2": True}
    per[M.CHA]["p50"] = 0.001
    assert M.reading(per)["reading"] == "agreement with T only" and not M.reading(per)["stage_2"]
    per[M.CHA]["p50"] = 0.0
    per[M.BKN]["split"] = split(-0.01, 0.0)
    assert M.reading(per)["copying_T"] and not M.reading(per)["stage_2"]
    per[M.BKN]["split"] = split(0.002, -0.02, inl_p=-0.03, non_p=0.01)
    assert M.reading(per)["copying_T"]
    per[M.BKN]["rule"] = {"passes": False, "harm": ["near_p90_ft"]}
    assert M.reading(per) == {"verdict": "FAIL", "reading": "harm", "harm_rows": ["near_p90_ft"], "stage_2": False}


def test_f2_null():
    assert M.f2_null({"<150": 0.01, ">=250": -0.004})["holds"]
    assert not M.f2_null({"<150": 0.011})["holds"]


def test_inlier_split():
    base = {"w0": {"1:7": [False, 1.0], "1:8": [True, None], "2:7": [False, 2.0]},
            "w1": {"1:9": [True, None], "1:7": [False, 1.5]}}
    new = {"w0": {"1:7": [False, 0.5], "1:8": [False, 1.0], "2:7": [False, 1.0]},
           "w1": {"1:9": [True, None], "1:7": [True, None]}}
    inl = {"w0": {"1:7", "2:7"}, "w1": {"1:7"}}
    sm = M.split_matrix(base, new, inl)
    assert sm["inlier"]["counts"] == {"w0": [2, 0, 0], "w1": [1, 0, 1]}
    assert sm["non_inlier"]["counts"] == {"w0": [1, 1, 0], "w1": [1, 1, 1]}
    rows = M.split_rows(sm, n=50)
    assert rows["inlier"]["missed"]["change"] == pytest.approx(1 / 3)
    assert rows["non_inlier"]["missed"]["change"] == pytest.approx(-0.5)
    assert rows["inlier"]["position_p50_ft"]["change"] == pytest.approx(0.75 - 1.5)
    with pytest.raises(SystemExit):
        M.split_matrix(base, {"w0": new["w0"], "w1": {"1:9": [True, None]}}, inl)


def test_same_row_table_and_audit():
    rows = [{"hb": "150-200", "third": "near", "d1": 1.0, "dB": -0.5, "stage": "A", "inlier": True, "r2": 0.0,
             "dist_T": 1.0}] * 3 + \
           [{"hb": "150-200", "third": "near", "d1": 1.0, "dB": 0.4, "stage": "B", "inlier": False, "r2": 0.0,
             "dist_T": 3.2}]
    t = M.same_row_table(rows)["150-200|near"]
    assert t["n"] == 4 and t["p_ii_rows"] == 1 and t["opposite_sign_share"] == pytest.approx(0.75)
    a = M.audit({M.CHA: rows})[M.CHA]
    assert a["p_ii_distance_ft"]["3.0-3.5"] == 1 and a["inlier_share"] == pytest.approx(0.75)


def test_capture_restores_the_scorer():
    from tools import p8_scorecard as S
    orig = S.score_window
    store = {}
    with M.capture_obs(store):
        assert S.score_window is not orig
    assert S.score_window is orig
    with pytest.raises(RuntimeError):
        with M.capture_obs(store):
            raise RuntimeError("boom")
    assert S.score_window is orig


def test_moves_together():
    base = {"a": {"missed": 10, "missed_1to1": 12}}
    assert M.moves_together(base, {"a": {"missed": 10, "missed_1to1": 12}})
    assert M.moves_together(base, {"a": {"missed": 8, "missed_1to1": 11}})
    assert not M.moves_together(base, {"a": {"missed": 8, "missed_1to1": 12}})


def test_constants_match_the_design():
    from sportvu import eval as E
    assert M.MISS_FT == E.MISS_FT and M.FIT_MIN == 50 and M.F2_NULL == 0.01
    assert M.MARGINS == {"ghosts_per_box": 0.010, "nearfield_bias_ft": 0.10, "position_p90_ft": 0.10,
                         "near_p90_ft": 0.05}
    assert M.TEAM_MARGIN == -0.010 and M.MISSED_BAR == -0.010 and (M.ONE2ONE_BAR, M.ONE2ONE_FRAC) == (-0.0075, 0.75)
    assert M.STAGE_GAMES["s0"] == M.STAGE_GAMES["s1"] == (M.CHA, M.BKN)
