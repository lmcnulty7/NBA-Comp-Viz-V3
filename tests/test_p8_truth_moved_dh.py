"""tools.p8_truth_moved_dh: the post hoc |d_H| report on synthetic homographies. No video, no model, no workspace."""

import numpy as np
import pytest

from tools import p8_truth_moved_dh as D

I3 = np.eye(3).tolist()


def shift(dx, dy=0.0):
    """A homography that maps pixels to court ft as identity plus a translation."""
    return [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]


def row(H=I3, feet=((10.0, 20.0), (30.0, 40.0)), ok=True):
    return {"H": H, "ok": ok, "boxes": [{"foot_stab": list(p)} for p in feet]}


def acc(r):
    return r["ok"]


def test_to_court_applies_the_homography():
    out = D.to_court(shift(3.0, -4.0), [[1.0, 2.0], [0.0, 0.0]])
    np.testing.assert_allclose(out, [[4.0, -2.0], [3.0, -4.0]])
    assert D.to_court(I3, []).shape == (0, 2)


def test_d_h_is_the_distance_between_build_and_truth_at_each_foot():
    w = {"w0": ({0: row()}, {0: shift(3.0, 4.0)}, {0: shift(1.0, 0.0)})}
    r = D.measure(w, acc)
    assert r["median_abs_d_H_ft_mps_all"] == pytest.approx(5.0)
    assert r["median_abs_d_H_ft_cpu_all"] == pytest.approx(1.0)
    assert r["median_abs_truth_shift_ft_common"] == pytest.approx(np.hypot(2.0, 4.0), abs=1e-3)
    assert r["frames"] == {"mps": 1, "cpu": 1, "common": 1}
    assert r["boxes"]["mps_all"] == 2 and r["boxes"]["truth_shift_common"] == 2


def test_all_and_common_frames_are_kept_apart():
    side = {0: row(), 1: row(), 2: row()}
    w = {"w0": (side, {0: shift(2.0), 1: shift(2.0)}, {1: shift(0.5), 2: shift(0.5)})}
    r = D.measure(w, acc)
    assert r["frames"] == {"mps": 2, "cpu": 2, "common": 1}
    assert r["boxes"]["mps_common"] == r["boxes"]["cpu_common"] == 2
    assert r["boxes"]["mps_all"] == r["boxes"]["cpu_all"] == 4


def test_frames_without_an_accepted_build_h_or_boxes_are_skipped():
    side = {0: row(ok=False), 1: row(H=None), 2: row(feet=()), 3: row()}
    truth = {f: shift(1.0) for f in side}
    r = D.measure({"w0": (side, truth, truth)}, acc)
    assert r["frames"] == {"mps": 1, "cpu": 1, "common": 1}
    assert r["median_abs_truth_shift_ft_common"] == pytest.approx(0.0)


def test_truth_frames_missing_from_the_sidecar_are_skipped_and_empty_is_none():
    r = D.measure({"w0": ({}, {5: I3}, {5: I3})}, acc)
    assert r["frames"] == {"mps": 0, "cpu": 0, "common": 0}
    assert r["median_abs_d_H_ft_mps_all"] is None


def test_the_apart_game_and_a_cv_test_game_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="apart"):
        D.run("gsw_ind_2016", tmp_path / "a", tmp_path / "b")
    with pytest.raises(SystemExit, match="CV test"):
        D.run("tor_gsw_2016", tmp_path / "a", tmp_path / "b")


def test_the_two_roots_must_differ(tmp_path):
    with pytest.raises(SystemExit, match="same folder"):
        D.run("gsw_cha_2016", tmp_path, tmp_path)
