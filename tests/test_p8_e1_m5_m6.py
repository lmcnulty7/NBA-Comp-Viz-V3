"""tools.p8_e1_m5 and tools.p8_e1_m6: bottom-centre sidecars, the raw-to-sidecar box match, the reproduction check, and
M6's guards against touching the registered result, on synthetic data."""

import fnmatch

import pytest

from tools import p8_e1_m5 as M5
from tools import p8_e1_m6 as M6


def test_bottom_sidecar_replaces_only_foot_stab():
    side = {"source": "x", "fps": 30, "frames": [{"frame": 0, "H": None, "boxes": [
        {"tid": 1, "bbox": [10, 20, 30, 80], "foot_raw": [20.0, 80.0], "foot_stab": [22.0, 83.0], "foot_corrected": True}]}]}
    out = M5.bottom_sidecar(side)
    b = out["frames"][0]["boxes"][0]
    assert b["foot_stab"] == [20.0, 80.0] and b["foot_raw"] == [20.0, 80.0] and b["bbox"] == [10, 20, 30, 80]
    assert side["frames"][0]["boxes"][0]["foot_stab"] == [22.0, 83.0]          # the input is not changed
    assert out["fps"] == 30 and out["source"] == "x"


def test_match_boxes_is_one_to_one_at_iou_half():
    raw = [[0, 0, 10, 10, 0.9], [100, 0, 110, 10, 0.8], [0, 0, 10, 9, 0.5]]
    side = [[0, 0, 10, 10], [200, 0, 210, 10]]
    c = M5.match_boxes(raw, side)
    assert c == {"matched": 1, "raw_only": 2, "sidecar_only": 1, "raw": 3, "sidecar": 2}
    assert M5.match_boxes([], side)["sidecar_only"] == 2
    assert M5.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_same_stats_detects_any_difference():
    a = {"missed": 1, "in_frame": 5, "nr_ghosts": 0, "nr_boxes": 4, "cov_acc": 1.0, "cov_live": 2.0,
         "team": {"n": 1, "majority": 1}, "matched": {"3:1": [1.5, 0.2, "near"]}}
    assert M5.same_stats(a, dict(a))
    assert not M5.same_stats(a, {**a, "missed": 2})
    assert not M5.same_stats(a, {**a, "matched": {"3:1": [1.6, 0.2, "near"]}})
    assert not M5.same_stats(a, {**a, "matched": {}})


def test_m5_guard(monkeypatch):
    monkeypatch.setattr(M5, "GAMES", M5.GAMES + ("gsw_phx_2016",))
    with pytest.raises(SystemExit, match="held-out"):
        M5.check_games(["gsw_phx_2016"])
    with pytest.raises(SystemExit):
        M5.check_games(["hou_cle_2016"])


def test_m6_never_names_a_registered_build_or_score_file():
    assert M6.build_dir_name("r23") == "e1m6_r23_detcpu"
    assert M6.build_dir_name("e1") == "e1m6_e1_gsw_bkn_2015_detcpu"
    for bad in ("r23", "e1_gsw_bkn_2015_seed0", "e1_stub_seed0"):
        with pytest.raises(SystemExit):
            M6.check_new_name(bad)
    name = M6.score_file_name("e1")
    from tools.p8_e1_score import FOLDS
    assert not any(fnmatch.fnmatch(name, f"{m}__seed*__{g}.json") for m in FOLDS + ("stub",) for g in FOLDS)


def test_m6_proxy_rewrites_only_the_build_and_wraps_the_device_proxy():
    env = {M6.ENV: "/w.pt"}
    cmd = ["python", "build_trajectories.py", "--source", "x.mp4"]
    assert M6.rewrite(cmd, env) == ["python", "-m", "tools.p8_e1_m6", "child", "build_trajectories.py", "--source", "x.mp4"]
    assert M6.rewrite(cmd, {}) == cmd
    assert M6.rewrite(["python", "other.py"], env) == ["python", "other.py"]
    from tools import p8_device
    assert p8_device.rewrite(M6.rewrite(cmd, env)) == M6.rewrite(cmd, env)   # the inner proxy leaves it alone
    assert M6.DEVICE_ENV == {"P8_DEVICE_DETECTOR": "cpu"}


def test_m9_allowlist_is_r23_only_and_new_folders_only():
    assert M6.build_dir_name("r23", "gsw_cha_2016") == "m9_r23_detcpu__gsw_cha_2016"
    assert M6.build_dir_name("r23", "hou_cle_2016") == "m9_r23_detcpu__hou_cle_2016"
    assert M6.build_dir_name("r23") == "e1m6_r23_detcpu"
    for model, game in (("e1", "gsw_cha_2016"), ("e1", "hou_cle_2016"), ("r23", "gsw_ind_2016")):
        with pytest.raises(SystemExit):
            M6.build_dir_name(model, game)
    for bad in ("r23", "m9_r23_detcpu__gsw_ind_2016", "m9_e1_detcpu__gsw_cha_2016"):
        with pytest.raises(SystemExit):
            M6.check_new_name(bad)
    from tools.p8_e1_score import FOLDS
    for g in M6.M9_GAMES:
        name = M6.m9_score_name(g)
        assert not any(fnmatch.fnmatch(name, f"{m}__seed*__{x}.json") for m in FOLDS + ("stub",)
                       for x in FOLDS + ("hou_cle_2016",))
    assert set(M6.M9_REGISTERED) == set(M6.M9_GAMES)
