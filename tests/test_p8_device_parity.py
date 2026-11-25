"""tools.p8_device_parity: bands, the CPU-to-MPS box match, per-frame parity counts, the parity rule, the M4 reuse
check and the guards, on synthetic data."""

import pytest

from tools import p8_device_parity as M


def test_settings_match_the_pipeline_and_the_rule():
    import config
    assert (M.CUT, M.NMS_IOU) == (config.PLAYER_CONF, config.PLAYER_IOU)
    assert (M.FLOOR, M.IMGSZ) == (0.10, 640)
    assert (M.LOSS_MAX, M.MISSED_MAX) == (0.01, 0.005)


def test_bands():
    assert [M.conf_band(c) for c in (0.40, 0.55, 0.85, 0.95)] == ["0.40-0.50", "0.50-0.70", "0.70-0.90", ">=0.90"]
    assert [M.band(h, M.HEIGHT_BANDS) for h in (100, 150, 220, 300)] == ["<150", "150-200", "200-250", ">=250"]


def test_frame_parity_counts_losses_at_the_cut_and_conf_diffs_at_the_floor():
    cpu = [[0, 0, 10, 160, 0.95], [100, 0, 110, 100, 0.45], [200, 0, 210, 300, 0.20]]
    mps = [[0, 0, 10, 160, 0.93], [100, 0, 110, 100, 0.38], [200, 0, 210, 300, 0.25], [400, 0, 410, 50, 0.60]]
    f = M.frame_parity(cpu, mps)
    assert (f["cpu"], f["mps"], f["matched"], f["cpu_only"], f["mps_only"]) == (2, 2, 1, 1, 1)
    assert f["lost_by_conf"] == {"0.40-0.50": 1} and f["lost_by_height"] == {"<150": 1}
    assert sorted(round(d, 2) for d in f["conf_diffs"]) == [0.02, 0.05, 0.07]
    s = {}
    M.merge(s, f)
    M.merge(s, f)
    out = M.summarise(s, 2)
    assert out["mps_box_loss"] == 0.5 and out["cpu_boxes_per_frame"] == 2.0
    assert out["conf_abs_diff_max"] == pytest.approx(0.07)


def test_match_is_one_to_one():
    assert M.match([[0, 0, 10, 10, 1]], [[0, 0, 10, 10, 1], [0, 0, 10, 10, 1]]) == [(0, 0)]
    assert M.match([], [[0, 0, 1, 1, 1]]) == []


def test_parity_rule():
    assert M.passes(0.009, 0.004) and M.passes(0.0, -0.0049)
    assert not M.passes(0.01, 0.0) and not M.passes(0.0, 0.005) and not M.passes(None, 0.0)


def test_m4_reuse_needs_an_exact_match():
    a = {"f1": {"e1": [[0, 0, 1, 1, 0.5]], "r23": []}}
    assert M.same_preds(a, {**a, "f2": {}})
    assert not M.same_preds(a, {"f1": {"e1": [[0, 0, 1, 1, 0.50001]], "r23": []}})
    assert not M.same_preds(a, {"f9": {}})


def test_guard(monkeypatch):
    monkeypatch.setattr(M, "GAMES", M.GAMES + ("gsw_phx_2016",))
    with pytest.raises(SystemExit, match="held-out"):
        M.check_games(["gsw_phx_2016"])
    with pytest.raises(SystemExit):
        M.check_games(["hou_cle_2016"])
    src = open(M.__file__).read()
    assert "builds" not in src and "__seed" not in src                 # never a build folder or a score file


def test_mps_repeat_folders_and_counts(monkeypatch):
    from tools import p8_mps_repeat as R
    assert R.folder("r23") == "m7_r23_mps_repeat" and R.folder("e1") == "m7_e1_gsw_bkn_2015_mps_repeat"
    monkeypatch.setitem(R.FOLDERS, "r23", "r23")
    with pytest.raises(SystemExit):
        R.folder("r23")
    assert R.WINDOWS["r23"][0] == R.WINDOWS["e1"][0] == "gsw_bkn_2015_s04_w00"
    c = R.compare_counts({0: 3, 1: 4, 2: 5}, {1: 4, 2: 6, 3: 1})
    assert c == {"frames_a": 3, "frames_b": 3, "shared_frames": 2, "boxes_per_frame_a": 4.0,
                 "boxes_per_frame_b": 3.6667, "same_count_frames": 1}
    cmd = ["python", "build_trajectories.py", "--source", "x.mp4"]
    assert R.rewrite(cmd, {R.ENV: "/w.pt"})[:4] == ["python", "-m", "tools.p8_mps_repeat", "child"]
    assert R.rewrite(cmd, {}) == cmd
