"""tools.p8_a3drop: the pure rules of the dropped-box classes on synthetic data."""

from tools import p8_a3drop as M


def test_classify_order():
    assert M.classify(True, 0.3, 9.0, False) == "1_duplicate"
    assert M.classify(False, 0.9, 1.0, True) == "2_displaced_track_box"
    assert M.classify(None, 0.45, 1.0, True) == "3_low_confidence"
    assert M.classify(None, 0.8, 2.5, True) == "4a_real_player_gap"
    assert M.classify(None, 0.8, 2.0, False) == "4b_real_player_never_tracked"
    assert M.classify(None, 0.8, 2.8, False) == "5_near_player"
    assert M.classify(None, 0.8, 3.1, False) == "6_false_positive"


def test_frame_drops():
    pred = [[0, 0, 10, 10, 0.9], [4, 0, 14, 10, 0.8], [50, 50, 60, 60, 0.7], [100, 100, 110, 110, 0.6]]
    kept = [[0, 0, 10, 10], [53, 50, 63, 60]]
    d = {i: (best, dup) for i, best, dup in M.frame_drops(pred, kept)}
    assert 0 not in d                                   # kept at IoU 1
    assert d[1][1] is True                              # near kept box 0, which matches predict box 0
    assert 2 not in d                                   # IoU 0.54 with kept box 1: kept
    assert d[3] == (0.0, None)


def test_frame_drops_displaced():
    pred = [[0, 0, 10, 10, 0.9]]
    kept = [[4, 0, 14, 10]]                             # IoU 0.43: near, and no other predict box matches it
    assert M.frame_drops(pred, kept) == [(0, M.iou(pred[0], kept[0]), False)]


def test_arms():
    sh = {c: 0.0 for c in M.CLASSES}
    sh["2_displaced_track_box"] = 0.3
    sh["4b_real_player_never_tracked"] = 0.25
    sh["1_duplicate"] = 0.4
    assert M.arms(sh) == ["tracker-state arm", "activation arm"]
