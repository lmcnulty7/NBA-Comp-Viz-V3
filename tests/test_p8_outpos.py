"""tools.p8_outpos: the pure rules of the output-position arm on synthetic data."""

import numpy as np

from tools import p8_outpos as M


def test_shift_point_moves_along_the_camera_ray():
    ground = [0.0, 100.0]
    q = np.array([0.0, 50.0])                      # 50 ft in front of the camera
    out = M.shift_point(q, ground, ("150-200", "far"), {("150-200", "far"): -1.0})
    assert np.allclose(out, [0.0, 49.0])            # a short foot (-1 ft) is pushed 1 ft beyond, away from the camera
    assert np.allclose(M.shift_point(q, None, None, {}), q)
    assert np.allclose(M.shift_point(q, ground, ("x", "y"), {}), q)


def test_table_round_trip():
    t = {("150-200", "far"): -1.2, ("<150", "near"): -0.3}
    assert M.table_from_json(M.table_to_json(t)) == t


def test_decide():
    diff = {"position_p50_ft": -0.2, "nearfield_bias_ft": -0.9, "position_p90_ft": 0.02}
    iv = {"position_p50_ft": [-0.3, -0.1], "nearfield_bias_ft": [-1.0, -0.8], "position_p90_ft": [-0.05, 0.09]}
    assert M.decide(diff, iv)["verdict"] == "PASS"
    assert M.decide(diff, dict(iv, position_p90_ft=[-0.05, 0.11]))["verdict"] == "FAIL"
    assert M.decide(diff, dict(iv, position_p50_ft=[-0.3, 0.01]))["verdict"] == "FAIL"
    assert M.decide(dict(diff, nearfield_bias_ft=0.1), iv)["verdict"] == "FAIL"


def test_score_hou_refuses_without_a_committed_table(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(M, "TABLE", tmp_path / "missing.json")
    monkeypatch.setattr(M, "REPO", tmp_path)
    with pytest.raises(SystemExit):
        M.table_provenance()
