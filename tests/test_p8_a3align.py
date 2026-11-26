"""tools.p8_a3align: the pure rules of the alignment check on synthetic data."""

from tools import p8_a3align as M


def test_best_ious():
    raw = [[0, 0, 10, 10, 0.9], [100, 100, 110, 110, 0.8]]
    side = [{"bbox": [0, 0, 10, 10]}]
    assert M.best_ious(raw, side) == [1.0, 0.0]
    assert M.best_ious(raw, []) == [0.0, 0.0]


def test_window_alignment_offset_and_sub_stride():
    aligned = {0: [(0.9, 0.9)] * 10, 1: [(0.9, 0.6)] * 6 + [(0.9, 0.4)] * 4, -1: [(0.9, 0.3)] * 10,
               2: [], -2: []}
    a = M.window_alignment(aligned)
    assert not a["misaligned"] and abs(a["median_same_frame_iou"] - 0.9) < 1e-12
    shifted = {0: [(0.4, 0.4)] * 10, 1: [(0.4, 0.8)] * 10, -1: [], 2: [], -2: []}
    b = M.window_alignment(shifted)
    assert b["offset_misaligned"] and b["best_offsets"] == [1] and b["misaligned"]
    blurry = {0: [(0.6, 0.6)] * 10, 1: [(0.6, 0.5)] * 10, -1: [], 2: [], -2: []}
    c = M.window_alignment(blurry)
    assert c["sub_stride_suspect"] and not c["offset_misaligned"] and c["misaligned"]


def test_reading():
    assert M.reading(0.5) == "artefact" and M.reading(0.49) == "model run justified" and M.reading(None) == "no a3"
