"""tools.p8_detrow: the pure rules of the detection-row cause breakdown on synthetic data."""

import numpy as np

from tools import p8_detrow as M


def test_constants_agree():
    from sportvu.bench import MATCH_FT
    from tools import p8_teachers as T
    assert M.NEAR_FT == MATCH_FT and M.TEACHER_CONF == T.MISS_TEACHER_CONF


def test_expected_box_inside_and_iou():
    b = M.expected_boxes([[100.0, 500.0]], [200.0])[0]
    assert np.allclose(b, [60.0, 300.0, 140.0, 500.0])
    assert M.inside_share(b, 1280, 720) == 1.0
    assert abs(M.inside_share([-40.0, 300.0, 40.0, 500.0], 1280, 720) - 0.5) < 1e-12
    assert M.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0 and M.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_cause_order():
    assert M.cause(True, True, True, True, True) == "edge"
    assert M.cause(False, True, True, True, True) == "occluded"
    assert M.cause(False, False, True, True, True) == "small"
    assert M.cause(False, False, False, None, None) == "not_measured"
    assert M.cause(False, False, False, True, True) == "raw_had_box"
    assert M.cause(False, False, False, False, True) == "teacher_only"
    assert M.cause(False, False, False, False, False) == "none"
    assert M.cause(None, None, False, False, True) == "teacher_only"


def test_near_any_and_matters():
    assert M.near_any([0.0, 0.0], [[3.0, 4.0]]) and not M.near_any([0.0, 0.0], [[3.0, 4.1]])
    assert not M.near_any([0.0, 0.0], np.zeros((0, 2)))
    assert M.matters({g: 0.012 for g in M.POOL})
    assert not M.matters(dict({g: 0.012 for g in M.POOL}, **{M.POOL[0]: 0.009}))


def test_summarise_and_boot():
    rows = [{"w": "g_s00_w00", "cls": "edge", "edge": True, "occ": False, "small": False, "raw": None,
             "teacher": None, "low": None},
            {"w": "g_s00_w01", "cls": "none", "edge": False, "occ": False, "small": False, "raw": False,
             "teacher": False, "low": True}]
    s = M.summarise(rows, {"g_s00_w00": 10, "g_s00_w01": 10})
    assert s["far_misses"] == 2 and s["classes"]["edge"]["rate_in_frame"] == 0.05
    assert s["report_only_threshold_reachable"]["none"] == 1
    b = M.boot(rows, {"g_s00_w00": 10, "g_s00_w01": 10}, n=50)
    assert b["edge"]["share"][0] <= b["edge"]["share"][1]
