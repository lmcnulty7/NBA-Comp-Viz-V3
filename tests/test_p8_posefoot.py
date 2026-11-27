"""tools.p8_posefoot (docs/research/P8_posefoot.md version 1.1), stage 0a: the refusal, the null, the rules, the
readings and the report rows, on synthetic data. No video, no model, no workspace."""

import numpy as np
import pytest

from tools import p8_posefoot as M


def test_hou_is_refused_before_any_file_is_read(monkeypatch):
    original = M.collect0a
    called = []
    monkeypatch.setattr(M, "collect0a", lambda g: called.append(g))
    with pytest.raises(SystemExit):
        M.main(["stage0a", "hou_cle_2016"])
    assert called == []
    with pytest.raises(SystemExit):
        M.refuse_game("gsw_sac_2015")
    import tools.p8_rebaseline as RB
    monkeypatch.setattr(RB, "read_json", lambda p: (_ for _ in ()).throw(AssertionError("a file was read")))
    with pytest.raises(SystemExit):
        original("hou_cle_2016")


def test_scorer_mapping_matches_footbias():
    from tools.p8_footbias import scorer_court
    H = np.array([[0.1, 0.01, -3.0], [0.0, 0.12, -5.0], [0.0, 0.0001, 1.0]])
    px = [[100.3, 200.7], [640.1, 400.9]]
    assert np.allclose(M.map_px(H, px), scorer_court(H, px))
    assert M.map_px(H, []).shape == (0, 2)


def test_null_partner_and_foot():
    hts = [180.0, 150.0, 175.0, 200.0]
    assert M.null_partner(hts, 0, [True] * 4) == 2
    assert M.null_partner(hts, 0, [True, True, False, True]) == 3        # 200 is nearer 180 than 150 (tie broken low)
    assert M.null_partner([180.0], 0, [True]) is None
    assert M.null_partner([100.0, 90.0, 110.0], 0, [True] * 3) == 1      # equal distance: lower index
    assert M.null_foot([10.0, 20.0], [2.0, -4.0], 180.0, 90.0) == [14.0, 12.0]


def test_speed_bins():
    assert [M.speed_bin(v) for v in (None, 0.0, 4.99, 5.0, 9.99, 10.0, 30.0)] == \
        [None, "under 5", "under 5", "5 to 10", "5 to 10", "10 and over", "10 and over"]


def own(R=0.30, Rn=0.15, mt=(-0.8, [-1.0, -0.6]), mb=-0.1):
    return {"R": R, "R_null": Rn, "med_T": {"value": mt[0], "interval": mt[1]}, "med_B": {"value": mb}}


def hits(mt=0.05, lost=0.03):
    return {"med_T": {"value": mt}, "lost": lost}


def test_rules():
    assert M.rule_set(own(), hits(), {"value": -0.02}, "x")["all"]
    assert not M.rule_set(own(R=0.24, Rn=0.15), hits(), {"value": -0.02}, "x")["rule1_null_margin"]
    assert M.rule_set(own(R=0.25, Rn=0.15), hits(), {"value": -0.02}, "x")["rule1_null_margin"]
    assert not M.rule_set(own(mt=(-0.6, [-0.9, 0.01])), hits(), {"value": -0.02}, "x")["rule2_median_T"]
    assert not M.rule_set(own(mt=(-0.4, [-0.9, -0.1])), hits(), {"value": -0.02}, "x")["rule2_median_T"]
    assert not M.rule_set(own(mb=0.01), hits(), {"value": -0.02}, "x")["rule3_median_B"]
    assert not M.rule_set(own(), hits(lost=0.06), {"value": -0.02}, "x")["rule4_hits"]
    assert not M.rule_set(own(), hits(mt=0.11), {"value": -0.02}, "x")["rule4_hits"]
    assert not M.rule_set(own(), hits(), {"value": -0.009}, "x")["rule5_est"]


def test_readings_0a():
    ok = M.rule_set(own(), hits(), {"value": -0.02}, "x")
    bad4 = M.rule_set(own(), hits(lost=0.2), {"value": -0.02}, "x")
    r = M.readings_0a({M.CHA: ok, M.BKN: bad4})
    assert r == {"c_stabiliser_finding": True, "d_pose_reads_hybrid_only": True}
    r = M.readings_0a({M.CHA: ok, M.BKN: M.rule_set(own(R=0.1), hits(), {"value": -0.02}, "x")})
    assert r == {"c_stabiliser_finding": False, "d_pose_reads_hybrid_only": False}


def row(wi, kind, cls, stab, raw, null, g_stab=False, g_raw=False, speed=7.0):
    return {"wi": wi, "kind": kind, "class": cls, "h": 180.0, "speed": speed,
            "dT": {"stab": stab, "raw": raw, "null": null}, "dB": {"stab": stab, "raw": raw, "null": null},
            "gT": {"stab": g_stab, "raw": g_raw, "null": False}}


def test_unit_rows_counts_and_est():
    rows = []
    for wi in range(4):
        rows += [row(wi, "P-ii", "own", 3.0, 2.0, 3.5, g_stab=True), row(wi, "P-ii", "own", 3.2, 3.0, 2.0),
                 row(wi, "P-ii", "switch", 3.0, 1.0, 3.0),
                 row(wi, "hit", "own", 1.0, 1.2, 1.0), row(wi, "hit", "own", 2.0, 2.6, 2.0, g_raw=True)]
    per = {M.CHA: {"windows": 4, "rows": rows, "in_frame_by_window": [100] * 4, "nr_boxes_by_window": [90] * 4}}
    out = M.unit_rows(per, [M.CHA], "raw", {M.CHA: 16})
    assert out["P-ii own"]["n"] == 8 and out["P-ii own"]["R"] == pytest.approx(0.5)
    assert out["P-ii own"]["R_null"] == pytest.approx(0.5)
    assert out["P-ii own"]["med_T"]["value"] == pytest.approx(np.median([-1.0, -0.2]))
    assert out["hits"]["lost"] == pytest.approx(0.5) and out["s_hit"] == pytest.approx(2.0)
    # est: per window -(2 rescued P-ii) + 2 x (1 lost hit) = 0, over 100 in-frame players
    assert out["est"]["value"] == pytest.approx(0.0)
    # ghosts: P-ii own row 1 stops being a ghost (-1); one hit becomes a ghost (+1, scaled by 2)
    assert out["ghost_change_per_box"]["value"] == pytest.approx((-1 + 2) / 90)
    assert out["R_by_speed"]["5 to 10"]["value"] == pytest.approx(0.5)
    assert out["R_by_speed"]["under 5"]["value"] is None


def test_constants_match_the_design():
    assert (M.NULL_MARGIN, M.MED_BAR_FT, M.HIT_MED_FT, M.HIT_LOST, M.EST_BAR) == (0.10, -0.5, 0.10, 0.05, -0.010)
    assert M.OWN == ("own", "own, with crossings", "own, misplaced, with crossings", "own, misplaced")
    assert M.POOL == (M.CHA, M.BKN) and (M.BOOT_N, M.SEED) == (2000, 0)


def test_cross_frame_partner_frame():
    fr = [0, 6, 60, 66, 120]
    assert M.cross_frame(fr, 6, 59.94) == 66
    assert M.cross_frame(fr, 120, 59.94) == 60
    assert M.cross_frame([0, 6, 12], 6, 59.94) is None


def test_rule1_reads_r_and_r_null_on_the_same_boxes():
    rows = []
    for wi in range(2):
        rows += [row(wi, "P-ii", "own", 3.0, 2.0, 3.5), row(wi, "P-ii", "own", 3.0, 2.0, None),
                 row(wi, "hit", "own", 1.0, 1.0, 1.0)]
    per = {M.CHA: {"windows": 2, "rows": rows, "in_frame_by_window": [50, 50], "nr_boxes_by_window": [40, 40]}}
    out = M.unit_rows(per, [M.CHA], "raw", {M.CHA: 2})
    assert out["P-ii own"]["R"] == pytest.approx(1.0) and out["P-ii own"]["R_null"] == pytest.approx(0.0)
    assert out["P-ii own"]["R_all_boxes"]["value"] == pytest.approx(1.0)
    assert out["P-ii own"]["cross_frame_null"]["R"]["value"] is None


def test_the_only_writes_are_the_records():
    """Writer check: the module writes through atomic_write only, at its two record paths."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(M))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = [getattr(c.func, "id", getattr(c.func, "attr", None)) for c in calls]
    assert names.count("atomic_write") == 2
    assert not any(n in ("write_text", "write_bytes", "unlink", "rmtree", "remove", "rename") for n in names)
    assert "open" not in names
