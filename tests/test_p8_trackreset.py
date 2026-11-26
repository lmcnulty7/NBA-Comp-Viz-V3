"""tools.p8_trackreset: the pure rules of the tracker-reset check on synthetic data."""

from tools import p8_trackreset as M


def test_heavy_windows():
    assert M.heavy_windows({"a": 80, "b": 15, "c": 5}) == ["a"]
    assert M.heavy_windows({"a": 50, "b": 30, "c": 20}) == ["a", "b"]
    assert M.heavy_windows({}) == []
    assert M.heavy_windows({"b": 10, "a": 10}) == ["a", "b"]


def test_runs_and_reset():
    assert M.runs([True, True, False, True, False, False]) == [2, 1]
    assert M.runs([]) == []
    assert M.reset_like({1, 2, 3, 4, 5}, {5, 9, 10})
    assert not M.reset_like({1, 2, 3}, set())
    assert not M.reset_like({1, 2, 3, 4}, {1, 2, 3})


def test_window_metrics():
    frames = [{"frame": 0, "boxes": [{"tid": 1}, {"tid": 2}, {"tid": 3}, {"tid": 4}]},
              {"frame": 6, "boxes": []},
              {"frame": 12, "boxes": [{"tid": 9}]},
              {"frame": 18, "boxes": [{"tid": 9}, {"tid": 10}]}]
    m, st = M.window_metrics(frames, {0: 4, 6: 3, 18: 4}, {"camera_cut_resets": 1, "median_players_per_frame": 2})
    assert m["empty_share"] == 0.25 and m["longest_empty_run"] == 1 and m["reset_like_frames"] == 1
    assert m["empty_share_where_detector_fired"] == 1 / 3
    assert st[6] == (True, True) and st[12][1] is True and st[18] == (False, True)
    assert abs(m["median_box_yield_where_detector_fired"] - 0.5) < 1e-12
