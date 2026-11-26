"""tools.p8_a3run: the pure rules of the build-runtime model run on synthetic data (the run itself needs the lock)."""

from tools import p8_a3run as M


def test_windows_fixed():
    assert len(M.HEAVY) == 6 and len(M.CONTROL) == 3
    assert {M.game_of_window(w) for w in M.HEAVY + M.CONTROL} == set(M.POOL)


def test_same_boxes_and_matched_count():
    a = [[0, 0, 10, 10], [20, 20, 30, 30]]
    assert M.same_boxes(a, [[20.5, 20, 31, 30], [0, 1, 10, 10]])
    assert not M.same_boxes(a, [[0, 0, 10, 10]])
    assert not M.same_boxes(a, [[0, 0, 13, 10], [20, 20, 30, 30]])
    assert M.same_boxes([], [])
    assert M.matched_count(a, [[0, 0, 10, 10], [100, 100, 110, 110]], 0.7) == 1
    assert M.matched_count([], a, 0.7) == 0


def test_dropped():
    pred = [[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]]
    track = [[0, 0, 10, 10]]
    unconf = [[20, 20, 30, 30]]
    assert M.dropped(pred, track, unconf) == (2, 1)


def test_reading():
    assert M.read_v({"tracker_explains": True, "runtime_differs": False}) == "tracker"
    assert M.read_v({"tracker_explains": False, "runtime_differs": True}) == "runtime"
    assert M.read_v({"tracker_explains": True, "runtime_differs": True}) == "both"
    assert M.read_v({"tracker_explains": False, "runtime_differs": False}) == "neither"


def test_window_report_counts():
    rec = {"per_frame": [{"frame": 0, "pred": [[0, 0, 10, 10, 0.9], [20, 20, 30, 30, 0.8]],
                          "track": [[0, 0, 10, 10, 1]], "unconfirmed": [[20, 20, 30, 30]], "lost": 0}]}
    side = {"frames": [{"frame": 0, "boxes": [{"bbox": [0, 0, 10, 10]}]}]}
    r = M.window_report(rec, side, {0: [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]})
    assert r["reproduced_share"] == 1.0 and r["dropped"] == 1 and r["dropped_under_unconfirmed"] == 1
    assert r["ours_boxes"] == 3 and r["ours_matched"] == 2
