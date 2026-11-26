"""tools.p8_e1r: arms and folders, the fold rule, per-window record and frame checks, the four readings, the blind and
Stage 2 verdicts through the registered d5_verdict, and the guards, on synthetic data. No build runs here."""

import fnmatch

import numpy as np
import pytest

from tools import p8_e1r as E


def test_r23_arm_never_resolves_to_the_registered_build():
    assert E.r23_dir("gsw_bkn_2015") == "e1m6_r23_detcpu"
    for g in ("gsw_cha_2016", "gsw_sac_2015", "hou_cle_2016"):
        assert E.r23_dir(g) == f"m9_r23_detcpu__{g}"
    assert all(E.r23_dir(g) != "r23" for g in E.POOL + (E.APART,))
    with pytest.raises(SystemExit):
        E.r23_dir("gsw_ind_2016")


def test_each_pool_game_is_scored_by_the_fold_that_left_it_out():
    for g in E.POOL:
        assert E.fold_for(g) == g and E.fold_for(g, g) == g
        for other in set(E.POOL) - {g}:
            with pytest.raises(SystemExit):
                E.fold_for(g, other)                  # a fold trained on g
    with pytest.raises(SystemExit):
        E.fold_for(E.APART)                           # hou_cle_2016 needs the fold named
    assert E.fold_for(E.APART, "gsw_cha_2016") == "gsw_cha_2016"
    assert E.e1_dir("gsw_bkn_2015") == "e1m6_e1_gsw_bkn_2015_detcpu"
    assert E.e1_dir("gsw_sac_2015") == "e1r_e1_gsw_sac_2015_detcpu"
    assert E.e1_dir(E.APART, "gsw_sac_2015") == "e1r_e1_gsw_sac_2015_detcpu__hou_cle_2016"


def test_only_e1r_folders_are_built():
    assert E.build_folder("e1", "gsw_cha_2016", None, False) == "e1r_e1_gsw_cha_2016_detcpu"
    assert E.build_folder("r23", "gsw_cha_2016", None, True) == E.SMOKE_DIR
    with pytest.raises(SystemExit):
        E.build_folder("e1", "gsw_bkn_2015", None, False)          # M6's folder is never rebuilt
    with pytest.raises(SystemExit):
        E.build_folder("r23", "gsw_cha_2016", None, False)         # r23 only for smoke gate 1
    with pytest.raises(SystemExit):
        E.build_folder("r23", "gsw_sac_2015", None, True)
    assert all(n.startswith("e1r_") for n in E.BUILDABLE)


def test_score_files_never_collide_with_the_registered_ones():
    assert E.score_name("gsw_cha_2016") == "gsw_cha_2016__seed0__gsw_cha_2016.json"
    assert E.score_name("gsw_cha_2016", smoke=True).endswith("__smoke.json")
    src = open(E.__file__).read()
    assert 'p8_root() / "e1r"' in src and 'p8_root() / "e1" /' not in src      # written under e1r/ only


def test_record_checks():
    sha_r, sha_e = E.R23_SHA, "0579a034" + "0" * 56
    dev_cpu = {"requested": dict(E.REQUESTED), "resolved": {"detector": "cpu"}}
    assert E.check_records("r23", {"sha256": sha_r}, dev_cpu, sha_r, None, False) == []
    assert E.check_records("new", {"sha256": sha_e, "fold": "gsw_cha_2016"}, dev_cpu, sha_e, "gsw_cha_2016", True) == []
    assert E.check_records("new", {"sha256": sha_e}, dev_cpu, sha_e, "gsw_cha_2016", True)          # fold missing
    assert E.check_records("new", {"sha256": sha_e, "fold": "gsw_bkn_2015"}, dev_cpu, sha_e, "gsw_cha_2016", False)
    assert E.check_records("r23", {"sha256": sha_r}, {"resolved": {"detector": "mps"}}, sha_r, None, False)
    assert E.check_records("r23", {"sha256": sha_r}, {**dev_cpu, "clip_fix": True}, sha_r, None, False)
    assert E.check_records("r23", {"sha256": "x"}, dev_cpu, sha_r, None, False)
    assert E.check_records("r23", None, None, sha_r, None, False)
    assert E.check_records("r23", {"sha256": sha_r}, {**dev_cpu, "requested": {"detector": "cpu", "court": "cpu",
                                                                                "gate": "auto"}}, sha_r, None, False)


def test_frame_set_and_court_h_checks():
    a = {"frames": [{"frame": 0, "H": [1, 2]}, {"frame": 6, "H": [3, 4]}]}
    assert E.check_frames(a, {"frames": [{"frame": 6, "H": [3, 4]}, {"frame": 0, "H": [1, 2]}]}) == []
    assert E.check_frames(a, {"frames": [{"frame": 0, "H": [1, 2]}]})
    assert E.check_frames(a, {"frames": [{"frame": 0, "H": [1, 2]}, {"frame": 6, "H": [3, 4.0001]}]})


def test_duplicates():
    sv = np.array([[10.0, 5.0], [30.0, 5.0], [60.0, 5.0]])
    feet = np.array([[10.2, 5.0], [11.0, 5.0], [30.0, 5.0], [61.0, 5.0], [61.5, 5.0]])
    assert E.duplicates(feet, sv, [True, True, False]) == 1
    assert E.duplicates(np.zeros((0, 2)), sv, [True, True, True]) == 0


def test_the_four_readings_and_stage2():
    assert E.reading(True, True) == "seed 0 passes on the reference platform"
    for p, b in ((True, False), (False, True), (False, False)):
        assert E.reading(p, b).startswith("fail")
    assert E.stage2_passes({f: True for f in E.POOL})
    assert not E.stage2_passes({"gsw_bkn_2015": True, "gsw_cha_2016": True})
    assert not E.stage2_passes({**{f: True for f in E.POOL}, "gsw_sac_2015": False})


def _win(missed_r, missed_e, ghosts=(1, 1), team=(8, 8), err=(1.0, 0.9)):
    def side(m, gh, t, e):
        return {"missed": m, "in_frame": 20, "nr_ghosts": gh, "nr_boxes": 20, "team": {"n": 10, "majority": t},
                "cov_acc": 10.0, "cov_live": 10.0,
                "matched": {"0:1": [e, 0.5 * e, "near"], "0:2": [e, -0.5 * e, "far"]}}
    return {"r23": side(missed_r, ghosts[0], team[0], err[0]), "new": side(missed_e, ghosts[1], team[1], err[1])}


def test_window_pool_verdict_uses_the_registered_rule():
    good = [_win(5, 4, ghosts=(2, 1), team=(8, 9)) for _ in range(6)]
    assert E.window_pool_verdict(good)["adopt"]
    worse = [_win(4, 6, ghosts=(2, 1), team=(8, 9)) for _ in range(6)]
    v = E.window_pool_verdict(worse)
    assert not v["adopt"] and v["rows"]["missed_rate"]["worse"]


def test_rows_equal():
    a = {"missed_rate": {"r23": 0.25253, "e1": 0.23782}}
    assert E.rows_equal(a, {"missed_rate": {"r23": 0.25253, "e1": 0.23782}}) == []
    assert E.rows_equal(a, {"missed_rate": {"r23": 0.25253, "e1": 0.2378201}})


def test_guards(monkeypatch):
    with pytest.raises(SystemExit):
        E.check_game("gsw_ind_2016")
    monkeypatch.setattr(E, "POOL", E.POOL + ("gsw_phx_2016",))
    with pytest.raises(SystemExit, match="held-out"):
        E.check_game("gsw_phx_2016")


def test_proxy_rewrites_only_the_build_and_wraps_the_device_proxy():
    cmd = ["python", "build_trajectories.py", "--source", "x.mp4"]
    env = {E.ENV: "/w.pt"}
    assert E.rewrite(cmd, env)[:4] == ["python", "-m", "tools.p8_e1r", "child"]
    assert E.rewrite(cmd, {}) == cmd
    from tools import p8_device
    assert p8_device.rewrite(E.rewrite(cmd, env)) == E.rewrite(cmd, env)
    src = open(E.__file__).read()
    assert ".train(" not in src and ".fit(" not in src


def test_stage_order():
    none = {"smoke_gate1": None, "smoke_gate4": None, "smoke_score": None, "report": None}
    assert E.stage_allowed("smoke", none) == []
    assert len(E.stage_allowed("stage1", none)) == 3
    ok = {"smoke_gate1": {"passes": True}, "smoke_gate4": {"passes": True}, "smoke_score": {}, "report": None}
    assert E.stage_allowed("stage1", ok) == []
    assert E.stage_allowed("stage1", {**ok, "smoke_gate4": {"passes": False}})
    assert E.stage_allowed("stage2", ok)                                         # no report yet
    assert E.stage_allowed("stage2", {**ok, "report": {"primary_pass": True, "blind": {"pass": False}}})
    assert E.stage_allowed("stage2", {**ok, "report": {"primary_pass": False, "blind": {"pass": True}}})
    assert E.stage_allowed("stage2", {**ok, "report": {"primary_pass": True, "blind": {"pass": True}}}) == []
    assert E.stage_of("gsw_cha_2016", True) == "smoke" and E.stage_of("gsw_cha_2016", False) == "stage1"
    assert E.stage_of(E.APART, False) == "stage2"


def test_stage2_block_wording():
    assert E.stage2_block(False, {f: True for f in E.POOL})["runs"] is False
    assert E.stage2_block(True, {"gsw_bkn_2015": True})["all_three_pass"] is None
    assert E.stage2_block(True, {f: True for f in E.POOL})["reading"] == E.STAGE2_PASS
    assert E.stage2_block(True, {**{f: True for f in E.POOL}, "gsw_cha_2016": False})["reading"] == E.STAGE2_FAIL


def _file(game, fold, wins, extra_expected=()):
    names = [f"{game}_s00_w{i:02d}" for i in range(len(wins))]
    return {"model": fold, "seed": 0, "game": game, "weights_sha256": "sha_" + fold,
            "expected_windows": names + list(extra_expected), "scored_windows": names, "gate_empty_windows": [],
            "frame_sets_differ": {}, "windows": dict(zip(names, wins)), "counts": {}}


def test_cmd_report_wiring(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(E, "p8_root", lambda: tmp_path)
    d = tmp_path / "e1r"
    d.mkdir()
    good = [_win(5, 4, ghosts=(2, 1), team=(8, 9)) for _ in range(6)]
    bad = [_win(4, 6, ghosts=(1, 2), team=(9, 8)) for _ in range(6)]

    def write(files):
        for p in d.glob("*.json"):
            p.unlink()
        for (g, f), body in files.items():
            (d / E.score_name(g, f if g == E.APART else None)).write_text(json.dumps(body))

    # the blind verdict uses only gsw_cha_2016 and gsw_sac_2015; gsw_bkn_2015 is bad
    write({("gsw_bkn_2015", None): _file("gsw_bkn_2015", "gsw_bkn_2015", bad),
           ("gsw_cha_2016", None): _file("gsw_cha_2016", "gsw_cha_2016", good),
           ("gsw_sac_2015", None): _file("gsw_sac_2015", "gsw_sac_2015", good)})
    out = E.cmd_report(None)
    assert out["blind"]["windows"] == 12 and out["blind"]["pass"]
    assert not out["primary_pass"] and out["stage2"]["runs"] is False      # pooled missed does not improve
    assert out["claim"] == E.reading(False, True)
    # a gate-empty mismatch (an expected window left unscored) makes the primary pool partial
    write({("gsw_bkn_2015", None): _file("gsw_bkn_2015", "gsw_bkn_2015", good, extra_expected=["gsw_bkn_2015_s09_w09"]),
           ("gsw_cha_2016", None): _file("gsw_cha_2016", "gsw_cha_2016", good),
           ("gsw_sac_2015", None): _file("gsw_sac_2015", "gsw_sac_2015", good)})
    out = E.cmd_report(None)
    assert out["partial"] and not out["primary_pass"] and out["claim"].startswith("fail")
    assert out["stage2"]["runs"] is False
    # all good: the claim holds; Stage 2 needs all 3 folds
    files = {(g, None): _file(g, g, good) for g in E.POOL}
    files.update({(E.APART, f): _file(E.APART, f, good) for f in ("gsw_bkn_2015", "gsw_cha_2016")})
    write(files)
    out = E.cmd_report(None)
    assert out["claim"] == "seed 0 passes on the reference platform"
    assert out["stage2"]["runs"] and out["stage2"]["all_three_pass"] is None
    files[(E.APART, "gsw_sac_2015")] = _file(E.APART, "gsw_sac_2015", good)
    write(files)
    assert E.cmd_report(None)["stage2"]["reading"] == E.STAGE2_PASS
