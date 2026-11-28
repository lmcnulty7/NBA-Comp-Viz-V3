"""tools.p8_c1reach: the sample, the court filter, the one-to-one reach, the cost rows, the per-frame statistics, the
scaling, the ceilings and the writes."""

import ast
import inspect

import numpy as np
import pytest

from tools import p8_c1reach as C


def test_sample_frames():
    assert C.sample_frames([5, 3, 3, 9]) == [3, 5, 9]
    fs = list(range(0, 300, 6))
    s = C.sample_frames(fs)
    assert len(s) == 8 and s[0] == 0 and s[-1] == 294 and s == sorted(s)
    assert C.sample_frames(fs) == s                                   # fixed, no seed


def test_court_keep():
    f = np.array([[0.0, 0.0], [-3.0, 25.0], [-3.1, 25.0], [97.0, 53.0], [97.1, 10.0], [47.0, -2.0]])
    assert C.court_keep(f).tolist() == [True, True, False, True, False, True]


def _sv():
    return np.array([[10.0, 10.0], [12.0, 10.0]] + [[60.0 + 3 * k, 40.0] for k in range(8)])


def test_one_to_one_reach_counts_a_shared_box_once():
    sv = _sv()
    one = np.array([[11.0, 10.0]])                                     # one foot between players 0 and 1
    assert len(C.matched(one, sv)) == 1
    two = np.array([[10.2, 10.0], [12.1, 10.0]])
    assert C.matched(two, sv) == {0, 1}
    assert C.matched(np.zeros((0, 2)), sv) == set()


def test_cost_rows():
    sv = _sv()
    inside = np.array([True] * 10)
    feet = np.array([[10.2, 10.0], [10.4, 10.1], [30.0, 30.0], [12.0, 12.8]])
    c = C.cost_rows(feet, sv, inside)
    assert c == {"boxes": 4, "ghosts": 1, "near_no_player": 2, "duplicates": 2}     # players 0 and 1 each have two
    assert C.cost_rows(np.zeros((0, 2)), sv, inside)["boxes"] == 0


def _box(foot_x, foot_y, s=0.9):                                       # Ht = identity: bottom centre is the foot
    return [foot_x - 1.0, foot_y - 6.0, foot_x + 1.0, foot_y, s, 0]


def test_frame_stats():
    sv = _sv()
    fr = {"sv": sv.tolist(), "Ht": np.eye(3).tolist(), "inside": [True] * 10,
          "r23_post": [_box(10.1, 10.0)], "r23_low": [_box(60.1, 40.0, 0.3)],
          "c1": [{"player": 1, "pid": 7, "unit": "D-i", "subclass": "C1a_other"},
                 {"player": 2, "pid": 8, "unit": "P-i", "subclass": "C1c"},
                 {"player": 3, "pid": 9, "unit": "P-i", "subclass": "C1c"}]}
    teacher = [_box(10.1, 10.0), _box(12.1, 10.0), _box(60.2, 40.0, 0.5), _box(63.0, 40.0, 0.1), _box(-20.0, 10.0)]
    st = C.frame_stats(fr, teacher)
    m = {x["player"]: x for x in st["misses"]}
    assert m[1]["teacher"] and not m[1]["floor"]                       # reached beyond the floor
    assert m[2]["teacher"] and m[2]["floor"]                           # r23 at 0.25 reaches it too
    assert not m[3]["teacher"] and m[3]["teacher_with_band"]           # only the 0.05 to 0.1471 band
    assert (st["hits"], st["hits_reached"]) == (1, 1)                   # the positive control's r23 hit
    assert st["cost"]["teacher"]["boxes"] == 3                         # the off-court box is filtered out


def test_scaled_and_ceilings():
    ws = {"misses": [{"unit": "D-i", "subclass": "C1a_other", "teacher": True, "floor": False},
                     {"unit": "P-i", "subclass": "C1c", "teacher": False, "floor": False}]}
    sc = C.scaled(ws, {"n": 6})
    assert sc[("D-i", "C1a_other")]["beyond"] == 3.0 and sc[("P-i", "C1c")]["unreached_oc"] == 3.0
    agg = C.aggregate({"gsw_cha_2016_s00_w00": ws, "gsw_bkn_2015_s00_w00": ws},
                      {"gsw_cha_2016_s00_w00": {"game": "gsw_cha_2016", "n": 6},
                       "gsw_bkn_2015_s00_w00": {"game": "gsw_bkn_2015", "n": 2}})
    f = {"gsw_cha_2016": {"f_pres": 0.5, "f_plc": 0.25}, "gsw_bkn_2015": {"f_pres": 0.6, "f_plc": 0.4}}
    c = C.ceilings(agg, f, {"gsw_cha_2016": 1000, "gsw_bkn_2015": 1000})
    assert c["gsw_cha_2016"]["C"] == pytest.approx(-3 * 0.5 / 1000)
    assert c["gsw_cha_2016"]["H"] == pytest.approx(-3 * 0.25 / 1000)
    assert c["gsw_cha_2016"]["C_all_plc"] == pytest.approx(-3 * 0.25 / 1000)
    assert c["gsw_cha_2016"]["C_with_residual"] == pytest.approx(c["gsw_cha_2016"]["C"] + C.CHA_RESIDUAL)
    assert c["pooled"]["C"] == pytest.approx((-1.5 - 0.6) / 2000)


def test_e1_fractions():
    from tools.p8_e1b_bound import TKEYS
    w = {k: {t: 0 for t in TKEYS} for k in ("g/a", "g/b")}
    w["g/a"].update(e1_gain_pres=4, e1_gain_pres_kept=2, e1_gain_plc=10, e1_gain_plc_kept=3)
    w["g/b"].update(e1_gain_pres=6, e1_gain_pres_kept=2)
    f = C.e1_fractions(w, ["g/a", "g/b"])
    assert f["f_pres"] == 0.4 and f["f_plc"] == 0.3


def test_allow_list_and_scope():
    for bad in ("hou_cle_2016_s00_w00", "gsw_sac_2015_s00_w00"):
        with pytest.raises(SystemExit):
            C.allowed([bad])
    src = inspect.getsource(C)
    assert "gsw_sac" not in src and "hou_cle" not in src and "cmd_run" not in src   # never T.cmd_run (10.3)


def test_writes():
    src = inspect.getsource(C)
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "atomic_write"]
    assert sorted(ast.unparse(c.args[0]) for c in calls) == sorted([
        "out_d / f\"inputs{('__smoke' if args.smoke else '')}.json\"", "out_d / f'teacher{sfx}.json'",
        "repo / 'reports' / 'research' / f\"p8_c1reach_stage1{('_smoke' if smoke else '')}.json\""])
