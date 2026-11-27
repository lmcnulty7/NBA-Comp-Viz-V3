"""tools.p8_nmsarm_stage1: the split, the IoU patch reaching model.track (after reset() too, with or without the device
patch), the records, the arm-only counts, the pilot gate and the writes."""

import ast
import inspect
import sys
import types

import numpy as np
import pytest

from tools import p8_nmsarm_stage1 as N1


def test_scorable_and_split():
    d = {f"gsw_cha_2016_s00_w{i:02d}": {"in_frame": (0 if i in (1, 5) else 10)} for i in range(12)}
    sc = N1.scorable(d)
    assert len(sc) == 10 and "gsw_cha_2016_s00_w01" not in sc
    pil, con = N1.split(sc)
    assert pil == [sc[0], sc[4], sc[8]] and len(con) == 7 and not set(pil) & set(con)


def test_iou_from_env():
    assert N1.iou_from_env({N1.IOU_ENV: "0.55"}) == 0.55 and N1.iou_from_env({N1.IOU_ENV: "0.45"}) == 0.45
    for bad in ({}, {N1.IOU_ENV: "0.7"}, {N1.IOU_ENV: "x"}):
        with pytest.raises(SystemExit):
            N1.iou_from_env(bad)


def test_rewrite_only_the_build_command():
    cmd = ["python", "build_trajectories.py", "--source", "a.mp4"]
    assert N1.rewrite(cmd, {}) == cmd
    assert N1.rewrite(cmd, {N1.ENV: "w.pt"})[:4] == ["python", "-m", "tools.p8_nmsarm_stage1", "child"]
    assert N1.rewrite(["python", "other.py"], {N1.ENV: "w.pt"}) == ["python", "other.py"]


class _FakeModel:
    def __init__(self, w):
        self.calls, self.predictor, self.callbacks = [], None, {"on_predict_start": [], "on_predict_postprocess_end": []}

    def track(self, bgr, **k):
        self.calls.append(k)
        self.predictor = object()
        return []


@pytest.fixture
def tracker_cls(monkeypatch):
    import detect.tracker as DT
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeModel))
    for name in ("__init__", "reset", "update"):
        monkeypatch.setattr(DT.PlayerTracker, name, getattr(DT.PlayerTracker, name))   # restored after the test
    return DT.PlayerTracker


@pytest.mark.parametrize("with_device_patch", [False, True])
@pytest.mark.parametrize("with_reset_fix", [False, True])
def test_iou_reaches_model_track_after_reset(tracker_cls, monkeypatch, with_device_patch, with_reset_fix):
    if with_device_patch:
        from tools import p8_device as D
        monkeypatch.setattr("config.get_device", __import__("config").get_device)
        D.apply({"detector": "cpu", "court": "auto", "gate": "auto"})
    if with_reset_fix:
        from tools.p8_resetfix import apply_reset_fix
        apply_reset_fix()
    N1.patch_iou(0.55, tracker_cls)
    t = tracker_cls(device="cpu")
    assert t.iou == 0.55
    t.update(np.zeros((8, 8, 3), np.uint8), 0)
    t.reset()
    t.update(np.zeros((8, 8, 3), np.uint8), 1)
    calls = t.model.calls
    assert len(calls) == 2 and all(c["iou"] == 0.55 for c in calls)
    if with_device_patch:
        assert all(c["device"] == "cpu" for c in calls)


def test_patch_iou_overrides_a_caller_value(tracker_cls):
    N1.patch_iou(0.45, tracker_cls)
    assert tracker_cls(device="cpu", iou=0.9).iou == 0.45


def test_record_problems():
    from tools.p8_resetfix import R23_SHA, REFERENCE
    dev = {"requested": REFERENCE, "resolved": {"detector": "cpu"}, "reset_fix": True, "detector_iou": 0.55}
    det = {"sha256": R23_SHA, "model": "r23"}
    assert N1.record_problems(dev, det, 0.55) == []
    assert N1.record_problems({**dev, "detector_iou": 0.45}, det, 0.55)
    assert N1.record_problems({**dev, "reset_fix": False}, det, 0.55)
    assert N1.record_problems(dev, {**det, "sha256": "x"}, 0.55)
    assert N1.record_problems(None, det, 0.55)


def test_same_fields_and_arm_only():
    w = {"missed": 3, "in_frame": 10, "nr_ghosts": 1, "nr_boxes": 9, "matched": {"1:2": [1.0, 0.5, "near"]}, "far": 1,
         "missed_1to1": 3, "unmatched_1to1": 4, "team": {"n": 2, "majority": 1}, "cov_acc": 1.0, "cov_live": 2.0,
         "duplicates": 0}
    assert N1.same_fields(w, dict(w)) == [] and N1.same_fields({**w, "missed": 4}, w) == ["missed"]
    assert N1.same_fields({**w, "matched": {"1:2": (1.0, 0.5, "near")}}, w) == []     # a tuple equals its JSON list
    b = [{"f": 1, "pid": 7, "missed": True}, {"f": 1, "pid": 8, "missed": False}, {"f": 2, "pid": 7, "missed": True}]
    a = [{"f": 1, "pid": 7, "missed": False}, {"f": 1, "pid": 8, "missed": True}, {"f": 2, "pid": 7, "missed": True}]
    assert N1.arm_only(b, a) == ([(1, 7)], [(1, 8)])


def _win(missed, u1, dup, n=1000, far=0):
    return {"missed": missed, "in_frame": n, "nr_ghosts": 50, "nr_boxes": 900, "far": far, "unmatched_1to1": u1,
            "duplicates": dup, "cov_acc": 1.0, "cov_live": 1.0, "matched": {}, "team": {"n": 0, "majority": 0}}


def test_pilot_gate_outcomes():
    rng = np.random.default_rng(0)
    prev = [_win(200, 300, 50) for _ in range(31)]
    good = [_win(195 + int(rng.integers(0, 2)), 290 + int(rng.integers(0, 2)), 52) for _ in range(31)]
    mech = {"arm_only_hits_by_class": {"P-i": 30, "D-i": 20, "P-ii": 10}}
    g = N1.pilot_gate(prev, good, mech, n_boot=200)
    assert g["outcome"] == "PASS" and not g["gates"]["P2_kill"] and not g["mechanism"]["flag_ceiling"]
    worse = [_win(195, 301, 50) for _ in range(31)]                          # missed falls, one-to-one rises
    assert N1.pilot_gate(prev, worse, mech, n_boot=200)["outcome"] == "KILL"
    dup = [_win(195, 290, 60) for _ in range(31)]                            # +0.01 duplicates per in-frame player
    r = N1.pilot_gate(prev, dup, mech, n_boot=200)
    assert r["outcome"] == "STOP" and not r["gates"]["duplicate_guard"]
    pii = {"arm_only_hits_by_class": {"P-i": 5, "P-ii": 10}}
    r = N1.pilot_gate(prev, good, pii, n_boot=200)
    assert r["outcome"] == "STOP" and r["mechanism"]["flag_pii"]
    big = [_win(185, 290, 50) for _ in range(31)]                            # missed falls 0.015, above the ceiling
    r = N1.pilot_gate(prev, big, mech, n_boot=200)
    assert r["mechanism"]["flag_ceiling"] and r["outcome"] == "STOP"
    noisy = [_win(195, 300 + (40 if i % 2 else -48), 50) for i in range(31)]  # a small mean fall, wide spread
    r = N1.pilot_gate(prev, noisy, mech, n_boot=200)
    assert r["outcome"] == "STOP" and not r["gates"]["P3_powered"]


def test_stage_order(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(N1, "score_path", lambda st: tmp_path / f"{st}.json")
    (tmp_path / "screen.json").write_text(json.dumps({"verdict": {"arm": "0.55"}}))
    assert N1.stage_problems("identity", "build") == []
    assert N1.stage_problems("control", "build") == ["stage identity has not passed"]
    (tmp_path / "identity.json").write_text(json.dumps({"passed": True}))
    assert N1.stage_problems("control", "build") == [] and N1.stage_problems("pilot", "build")
    (tmp_path / "screen.json").write_text(json.dumps({"verdict": {"arm": None}}))
    assert "stage 0b did not choose 0.55" in N1.stage_problems("identity", "build")


def test_allow_list_and_windows():
    for bad in ("hou_cle_2016_s00_w00", "gsw_sac_2015_s00_w00"):
        with pytest.raises(SystemExit):
            N1.allowed([bad])
    assert all(s.startswith(("gsw_cha_2016_", "gsw_bkn_2015_")) for s in N1.IDENTITY + (N1.CONTROL,))
    d202 = {g: {"windows": {f"{g}_s00_w{i:02d}": {"in_frame": 1} for i in range(n)}}
            for g, n in (("gsw_cha_2016", 56), ("gsw_bkn_2015", 65))}
    assert len(N1.stage_windows("pilot", d202)) == N1.N_PILOT == 14 + 17


def test_writes_and_folders():
    src = inspect.getsource(N1)
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "atomic_write"]
    assert sorted(ast.unparse(c.args[0]) for c in calls) == ["d / f'build__{stage}.json'", "score_path(stage)"]
    reps = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_write_report"]
    assert sorted(ast.unparse(c.args[0]) for c in reps) == ["'pilot_gate'", "stage"]
    assert all("r23" not in v for v in N1.FOLDERS.values())
