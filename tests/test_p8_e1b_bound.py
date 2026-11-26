"""tools.p8_e1b_bound: the pure rules of the E1b upper bound on synthetic data, and that they reuse the scorer's
and the corpus's constants."""

import numpy as np

from tools import p8_e1b_bound as B


def test_constants_agree_with_the_frozen_rules():
    import config
    from sportvu import eval as E
    from sportvu.bench import MATCH_FT
    from tools import p8_teachers as T
    assert B.MISS_FT == E.MISS_FT
    assert B.NEAR_FT == MATCH_FT
    assert B.TEACHER_CONF == T.MISS_TEACHER_CONF
    assert B.DET_CONF == config.PLAYER_CONF


def test_nearest():
    sv = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert np.allclose(B.nearest(sv, np.array([[1.0, 0.0]])), [1.0, 9.0])
    assert np.isinf(B.nearest(sv, np.zeros((0, 2)))).all()


def test_reach_class_priority():
    inf = np.inf
    assert B.reach_class(2.0, 1.0, 1.0, 1.0) == "teacher"
    assert B.reach_class(3.0, 2.5, 1.0, 1.0) == "teacher_low"
    assert B.reach_class(3.0, 3.0, 2.0, 2.0) == "r23_low"
    assert B.reach_class(3.0, 3.0, 3.0, 4.9) == "placement"
    assert B.reach_class(inf, inf, inf, 5.1) == "none"


def test_l1_on_find_and_bins():
    assert [B.l1_on_find(v) for v in (2.5, 4.0, 5.1, np.inf)] == ["kept", "displaced", "dropped", "dropped"]
    assert [B.height_bin(v) for v in (100, 150, 199.9, 240, 250, None)] == \
        ["<150", "150-200", "150-200", "200-250", ">=250", "no pose"]
    assert [B.overlap_bin(v) for v in (0.0, 0.1, 0.39, 0.4, None)] == ["none", "partial", "partial", "heavy", "no pose"]


def test_expected_boxes_and_overlap():
    b = B.expected_boxes(np.array([[100.0, 500.0], [100.0, 500.0], [400.0, 500.0]]), np.array([200.0, 200.0, 200.0]))
    assert np.allclose(b[0], [60.0, 300.0, 140.0, 500.0])
    assert np.allclose(B.max_other_iou(b), [1.0, 1.0, 0.0])
    assert B.max_other_iou(b[:1]).tolist() == [0.0]


def test_ceilings_and_summary():
    rows = [
        {"missed": True, "teacher_reaches": True, "teacher_low_reaches": False, "r23_low_reaches": False,
         "reach": "teacher", "r23_box_within_5ft": False, "third": "near", "height": "150-200", "edge": "False",
         "overlap": "none"},
        {"missed": True, "teacher_reaches": False, "teacher_low_reaches": True, "r23_low_reaches": False,
         "reach": "teacher_low", "r23_box_within_5ft": True, "third": "far", "height": "<150", "edge": "True",
         "overlap": "heavy"},
        {"missed": False, "teacher_reaches": False, "teacher_low_reaches": False, "r23_low_reaches": False,
         "l1": "dropped", "third": "near", "height": "150-200", "edge": "False", "overlap": "none"},
        {"missed": False, "teacher_reaches": True, "teacher_low_reaches": False, "r23_low_reaches": False,
         "l1": "kept", "third": "middle", "height": "150-200", "edge": "False", "overlap": "partial"},
    ]
    c = B.ceilings(rows)
    assert c["r23"] == 0.5 and c["r23_or_teacher"] == 0.25 and c["plus_teacher_low"] == 0.0
    s = B.summary(rows)
    assert s["misses_by_reach"]["teacher"] == 1 and s["misses_by_reach"]["teacher_low"] == 1
    assert s["r23_finds_under_teacher_labels"] == {"kept": 1, "displaced": 0, "dropped": 1}
    assert s["by_third"]["near"]["r23_finds_dropped_by_teacher_labels"] == 1
    bt = B.boot({"w0": rows[:2], "w1": rows[2:]}, n=50)
    assert bt["r23"]["point"] == 0.5 and bt["teacher_gain"]["point"] == -0.25 and bt["windows"] == 2


def test_guard_refuses_other_games():
    import pytest
    with pytest.raises(SystemExit):
        B.check_games(["hou_cle_2016"])


def test_transfer_counts_and_ceiling():
    # players: 0 presence gain kept, 1 placement gain lost in build, 2 loss carried, 3 both find
    c = B.transfer_counts(rr=[False, False, True, True], er=[True, True, False, True], tr=[True, True, False, True],
                          r5=[False, True, True, True], rb=[False, False, True, True], eb=[True, False, False, True])
    assert (c["e1_gain_pres"], c["e1_gain_pres_kept"], c["e1_gain_plc"], c["e1_gain_plc_kept"]) == (1, 1, 1, 0)
    assert (c["e1_loss"], c["e1_loss_kept"]) == (1, 1)
    assert (c["l1_pres"], c["l1_plc"], c["l1_loss"]) == (1, 1, 1)
    f = B.fractions(c)
    assert f == {"f_pres": 1.0, "f_plc": 0.0, "f_loss": 1.0}
    t = B.ceiling_terms(c)
    assert t["C"] == 0.0 and t["C_opt"] == -1.0 and t["E1_build"] == 0
    r = B.pooled_rates({"g": c})
    assert r["pooled"]["C"] == 0.0
    assert B.gate(r, ["g"])["passes"] is False
    good = dict(c, l1_pres=10)
    assert B.gate(B.pooled_rates({"g": good}), ["g"])["passes"] is True
