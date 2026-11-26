"""tools.p8_buildloss: the pure rules of the build-loss classes on synthetic data."""

from tools import p8_buildloss as M


def box(tid, bbox, corrected=False):
    return {"tid": tid, "bbox": bbox, "foot_corrected": corrected}


def test_best_match():
    raw = [100, 100, 150, 250]
    boxes = [box(1, [100, 100, 150, 250]), box(2, [400, 100, 450, 250])]
    assert M.best_match(raw, boxes, 0.5)["tid"] == 1
    assert M.best_match(raw, [box(3, [130, 100, 180, 250])], 0.5) is None      # IoU 0.25
    assert M.best_match(raw, [], 0.3) is None


def test_classify_partition():
    nb = box(9, [0, 0, 1, 1])
    assert M.classify(None, None, nb, {1, 2}, {1}, None, None) == "a1_activation_lag"
    assert M.classify(None, None, box(2, [0, 0, 1, 1]), {1, 2}, {1}, None, None) == "a2_track_gap"
    assert M.classify(None, nb, None, {9}, set(), None, None) == "a2_track_gap"
    assert M.classify(None, None, None, set(), set(), None, None) == "a3_never_tracked_nearby"
    assert M.classify(nb, None, None, set(), {9}, False, False) == "b_tracked_box_differs"
    assert M.classify(box(9, [0, 0, 1, 1], True), None, None, set(), {9}, True, False) == "c_occlusion"
    assert M.classify(nb, None, None, set(), {9}, True, False) == "c_ema"
    assert M.classify(nb, None, None, set(), {9}, True, True) == "e_other"


def test_reset_and_matters():
    assert M.looks_reset({1, 2, 3, 4, 5}, {9, 10, 11, 12, 5})
    assert not M.looks_reset({1, 2, 3}, set())
    assert M.matters({g: 0.01 for g in M.POOL}) and not M.matters({g: 0.009 for g in M.POOL})


def test_classify_v2_decode_first_and_need():
    import pytest
    nb = box(9, [0, 0, 1, 1])
    assert M.classify_v2(None, False, nb, None, {9}, set(), None, None) == "d_decode"
    assert M.classify_v2(None, True, nb, None, {9}, set(), None, None) == "a2_track_gap"
    assert M.classify_v2(nb, False, None, None, set(), {9}, True, False) == "c_ema"      # a matched box: not decode
    with pytest.raises(SystemExit):
        M.need("/nonexistent/file.json")


def test_summarise_v2_reachable():
    rows = [{"w": "w0", "cls": "a3_never_tracked_nearby", "d_raw": 1.0},
            {"w": "w0", "cls": "a3_never_tracked_nearby", "d_raw": 3.0},
            {"w": "w1", "cls": "d_decode", "d_raw": 2.0}]
    s = M.summarise_v2(rows, {"w0": 50, "w1": 50})
    assert s["reachable"] == 2
    assert s["classes"]["a3_never_tracked_nearby"]["reachable"] == 1
    assert s["classes"]["a3_never_tracked_nearby"]["reachable_rate_in_frame"] == 0.01
    b = M.boot_v2(rows, {"w0": 50, "w1": 50}, n=50)
    assert b["d_decode"][0] <= b["d_decode"][1]
