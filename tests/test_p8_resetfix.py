"""tools.p8_resetfix and the P8_TRACKER_RESET_FIX flag of tools.p8_device: callbacks register once after a reset.

The fake below reproduces the three ultralytics 8.4.48 behaviours the bug needs (engine/model.py): Model.track registers
the tracker callbacks when the predictor has no trackers, Model.add_callback appends, and a new predictor takes the
model's callback lists. PlayerTracker.reset's frozen behaviour is predictor = None."""

import sys
import types
from functools import partial

import numpy as np
import pytest

from tools import p8_resetfix as R


def _tracker_module():
    mod = types.ModuleType(R.TRACKER_MODULE)

    def on_predict_start(predictor, persist=False):
        predictor.trackers = ["bot"]

    def on_predict_postprocess_end(predictor, persist=False):
        predictor.updates += 1
    for f in (on_predict_start, on_predict_postprocess_end):
        f.__module__ = R.TRACKER_MODULE
        setattr(mod, f.__name__, f)
    return mod


class FakePredictor:
    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.updates = 0

    def run(self):
        for c in self.callbacks["on_predict_start"]:
            c(self)
        for c in self.callbacks["on_predict_postprocess_end"]:
            c(self)


class FakeModel:
    def __init__(self, mod):
        self.mod = mod
        self.callbacks = {"on_predict_start": [], "on_predict_postprocess_end": [lambda p: None]}
        self.predictor = None

    def add_callback(self, ev, f):
        self.callbacks[ev].append(f)

    def track(self):
        if not hasattr(self.predictor, "trackers"):
            self.add_callback("on_predict_start", partial(self.mod.on_predict_start, persist=True))
            self.add_callback("on_predict_postprocess_end", partial(self.mod.on_predict_postprocess_end, persist=True))
        if self.predictor is None:
            self.predictor = FakePredictor(self.callbacks)
        before = self.predictor.updates
        self.predictor.run()
        return self.predictor.updates - before


class FakeTracker:
    def __init__(self, mod):
        self.model = FakeModel(mod)

    def update(self, bgr, frame_idx=0, timestamp_sec=0.0):
        return self.model.track()

    def reset(self):                     # the frozen detect.tracker.PlayerTracker.reset, reduced to what matters
        self.model.predictor = None


def test_bug_doubles_and_fix_restores_one_update_per_frame(monkeypatch):
    mod = _tracker_module()
    bug = FakeTracker(mod)
    assert [bug.model.track() for _ in range(3)] == [1, 1, 1]
    bug.reset()
    assert [bug.model.track() for _ in range(3)] == [2, 2, 2]          # the double update
    bug.reset()
    assert bug.model.track() == 3                                       # each reset adds one more
    fixed = FakeTracker(mod)
    orig = FakeTracker.reset

    def reset(self):
        orig(self)
        R.strip_tracker_callbacks(self.model.callbacks)
    monkeypatch.setattr(FakeTracker, "reset", reset)
    assert [fixed.model.track() for _ in range(2)] == [1, 1]
    for _ in range(3):
        fixed.reset()
        assert [fixed.model.track() for _ in range(2)] == [1, 1]        # exactly once after every reset
    # the non-tracker default callback is kept
    assert sum(1 for c in fixed.model.callbacks["on_predict_postprocess_end"] if not R.is_tracker_callback(c)) == 1


def test_is_tracker_callback_and_strip():
    mod = _tracker_module()
    cbs = {"on_predict_start": [partial(mod.on_predict_start, persist=True)],
           "on_predict_postprocess_end": [print, partial(mod.on_predict_postprocess_end, persist=True),
                                          partial(mod.on_predict_postprocess_end, persist=True)]}
    assert not R.is_tracker_callback(print) and not R.is_tracker_callback(partial(print))
    assert R.strip_tracker_callbacks(cbs) == 3
    assert cbs == {"on_predict_start": [], "on_predict_postprocess_end": [print]}


def test_device_flag():
    from tools import p8_device as D
    assert D.reset_fix_requested({}) is False and D.reset_fix_requested({"P8_TRACKER_RESET_FIX": "1"}) is True
    assert D.is_set({"P8_TRACKER_RESET_FIX": "1"})
    with pytest.raises(SystemExit):
        D.reset_fix_requested({"P8_TRACKER_RESET_FIX": "yes"})


def test_apply_reset_fix_patches_player_tracker_and_logs(monkeypatch, tmp_path):
    """apply_reset_fix on a stand-in detect.tracker module: the frozen reset still runs, the tracker callbacks go, the
    log records 2 stripped per reset, and the log file is written."""
    mod = _tracker_module()
    fake_dt = types.ModuleType("detect.tracker")
    fake_dt.PlayerTracker = type("PlayerTracker", (FakeTracker,), {})
    pkg = types.ModuleType("detect")
    pkg.tracker = fake_dt
    monkeypatch.setitem(sys.modules, "detect", pkg)
    monkeypatch.setitem(sys.modules, "detect.tracker", fake_dt)
    monkeypatch.setattr(R, "RESET_LOG", {"path": None, "resets": [], "last_frame": None})
    monkeypatch.setattr("atexit.register", lambda f: None)
    assert R.apply_reset_fix() == "detect.tracker.PlayerTracker.reset"
    t = fake_dt.PlayerTracker(mod)
    assert t.update(None, 0) == 1
    for i in range(3):
        t.reset()
        assert t.model.predictor is None                                 # the frozen reset still ran
        assert [t.update(None, 10 * (i + 1) + k) for k in range(2)] == [1, 1]
    assert R.RESET_LOG["resets"] == [{"frame": 0, "stripped": 2}, {"frame": 11, "stripped": 2},
                                     {"frame": 21, "stripped": 2}]
    monkeypatch.delenv("TRACKER_REID", raising=False)
    R.start_reset_log(tmp_path / "w_resetfix.json")
    R.write_reset_log()
    import json
    log = json.loads((tmp_path / "w_resetfix.json").read_text())
    assert log["reset_fix"] is True and log["resets"] == R.RESET_LOG["resets"] and log["tracker_reid_env"] is None
    assert set(log["versions"]) == {"opencv", "torch", "ultralytics"}


def test_real_ultralytics_callbacks_are_recognised():
    """With ultralytics importable (the frozen CV interpreter), its own tracker partials are the ones stripped."""
    TR = pytest.importorskip("ultralytics.trackers.track")
    assert R.is_tracker_callback(partial(TR.on_predict_start, persist=True))
    assert R.is_tracker_callback(partial(TR.on_predict_postprocess_end, persist=True))


# ── design v2: the rule and the helpers ─────────────────────────────────────────────────────────────────────────────
def _good():
    point = {"missed_rate": -0.01, "detection_rate": -0.005, "ghost_rate": 0.002, "position_p50_ft": 0.0,
             "position_p90_ft": 0.02, "nearfield_bias_ft": 0.0, "team_accuracy": 0.0, "missed_1to1_rate": -0.008}
    wiv = {"missed_rate": [-0.02, -0.001], "detection_rate": [-0.01, 0.001], "ghost_rate": [-0.001, 0.008],
           "position_p50_ft": [-0.03, 0.04], "position_p90_ft": [-0.05, 0.09], "nearfield_bias_ft": [-0.05, 0.05],
           "team_accuracy": [-0.02, 0.02]}
    return point, wiv


def test_v2_verdict_passes_and_each_condition_fails_alone():
    point, wiv = _good()
    assert R.v2_verdict(point, wiv, (0.8, 0.8))["pass"]
    for row, iv in (("ghost_rate", [0.0, 0.011]), ("position_p50_ft", [0.0, 0.051]), ("position_p90_ft", [0.0, 0.11]),
                    ("nearfield_bias_ft", [0.0, 0.101]), ("team_accuracy", [-0.031, 0.0])):
        w = dict(wiv); w[row] = iv
        v = R.v2_verdict(point, w, (0.8, 0.8))
        assert not v["pass"] and not v["rows"][row]["pass"], row
    w = dict(wiv); w["missed_rate"] = [0.001, 0.02]                          # worse by the window interval
    assert not R.v2_verdict(point, w, (0.8, 0.8))["pass"]
    assert not R.v2_verdict(point, wiv, (0.8, 0.8), {"detection_rate": [0.001, 0.01]})["pass"]   # worse by the game iv
    p = dict(point); p["missed_rate"] = 0.001                                # does not improve
    assert not R.v2_verdict(p, wiv, (0.8, 0.8))["pass"]
    assert R.v2_verdict(p, wiv, (0.8, 0.8), improve_rows=(), one_to_one=True)["pass"]
    p = dict(point); p["missed_1to1_rate"] = 0.0
    assert not R.v2_verdict(p, wiv, (0.8, 0.8))["pass"] and R.v2_verdict(p, wiv, (0.8, 0.8), one_to_one=False)["pass"]
    assert not R.v2_verdict(point, wiv, (0.8, 0.49))["pass"]                 # coverage floor
    w = dict(wiv); w["ghost_rate"] = None                                    # no interval fails a margin row
    assert not R.v2_verdict(point, w, (0.8, 0.8))["pass"]
    # a margin row is judged only by its margin: a ghost interval wholly above zero but under the margin passes
    w = dict(wiv); w["ghost_rate"] = [0.001, 0.009]
    assert R.v2_verdict(point, w, (0.8, 0.8))["pass"]


def test_pool_verdict_needs_every_game():
    point, wiv = _good()
    ok = R.v2_verdict(point, wiv, (0.8, 0.8))
    w = dict(wiv); w["ghost_rate"] = [0.0, 0.02]
    bad = R.v2_verdict(point, w, (0.8, 0.8), improve_rows=(), one_to_one=False)
    v = R.pool_verdict(ok, {"a": ok, "b": bad})
    assert not v["pass"] and v["pooled"] and "b:ghost_rate" in v["failing"]
    assert R.pool_verdict(ok, {"a": ok})["pass"]


def test_ratio_interval_matches_the_registered_boot():
    from tools.p8_e1_score import boot
    rng = np.random.default_rng(3)
    wins = []
    for _ in range(12):
        b = {"missed": int(rng.integers(5, 40)), "in_frame": 100, "nr_ghosts": int(rng.integers(0, 10)), "nr_boxes": 80,
             "cov_acc": 5.0, "cov_live": 8.0, "team": {"n": 10, "majority": 8},
             "matched": {f"{i}:0": [float(rng.random()), 0.1, "near"] for i in range(5)}}
        n = dict(b); n["missed"] = b["missed"] - int(rng.integers(0, 4))
        wins.append({"r23": b, "new": n})
    real = boot({i: [w] for i, w in enumerate(wins)}, n=300)
    pk = R.resample_picks(len(wins), 300, 0)
    for row, (num, den) in (("missed_rate", ("missed", "in_frame")), ("ghost_rate", ("nr_ghosts", "nr_boxes"))):
        got = R.ratio_interval(pk, [w["r23"][num] for w in wins], [w["r23"][den] for w in wins],
                               [w["new"][num] for w in wins], [w["new"][den] for w in wins])
        assert got == real[row]


def test_sim_arm_changes_only_reset_windows():
    base = {"missed": 50, "far": 20, "nr_boxes": 100, "nr_ghosts": 10, "in_frame": 200}
    after = {"reach": 10, "far_reach": 4, "ours_boxes": 40, "nr_boxes": 20}
    assert R.sim_arm(base, after, 0, 0.7, 0.2) == base
    new = R.sim_arm(base, after, 1, 0.7, 0.25)
    assert new == {"missed": 43, "far": 17, "nr_boxes": 114, "nr_ghosts": 14, "in_frame": 200}
    assert base["missed"] == 50


def test_truth_window_and_summary():
    fr = [{"frame": f, "boxes": [{"tid": t} for t in ((1, 2, 3, 4, 5) if f < 30 else (9,))]} for f in range(0, 60, 3)]
    w = R.truth_window(fr, {0, 3, 6, 9, 33, 100}, 1)
    assert w["first_reset_frame"] == 30 and w["proc_before"] == 10 and w["proc_after"] == 10
    assert w["acc_before"] == 4 and w["acc_after"] == 1 and w["acc_not_processed"] == 1
    s = R.truth_summary({"a": w, "b": dict(w)}, n_boot=50)
    assert s["rate_before_in_reset_windows"] == 0.4 and s["rate_after_in_reset_windows"] == 0.1
    assert abs(s["ratio_before_over_after"] - 4.0) < 1e-12 and abs(s["under_representation"] - 2.5) < 1e-12


def test_frame_counts():
    sv = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]] + [[100.0, 100.0]] * 7)
    inside = np.array([True, True, True] + [False] * 7)
    feet = np.array([[0.5, 0.0], [40.0, 40.0]])             # finds player 0; one ghost
    ours = np.array([[0.2, 0.0], [11.0, 0.0], [23.0, 0.0]])    # reaches player 1 within 2.5; player 2 within 5
    c = R.frame_counts(feet, sv, inside, [False, False], ours)
    assert c["in_frame"] == 3 and c["missed"] == 2 and c["nr_ghosts"] == 1 and c["nr_boxes"] == 2
    assert c["reach"] == 1 and c["far"] == 2 and c["far_reach"] == 2 and c["ours_ghosts"] == 0


# ── design v2 stage 0: pure checks ─────────────────────────────────────────────────────────────────────────────────
def test_plan_windows():
    plan = R.plan_windows({"g_s00_w00": 0, "g_s00_w01": 1, "g_s01_w00": 0, "g_s02_w00": 2, "g_s03_w00": 0}, {"g_s04_w00"})
    assert plan["reset_windows"] == ["g_s00_w01", "g_s02_w00"]
    assert plan["equivalence_windows"] == ["g_s00_w00", "g_s03_w00"]
    assert R.plan_windows({"a": 0}, set())["equivalence_windows"] == ["a"]


def _arm_records():
    det = {"sha256": R.R23_SHA, "model": "r23"}
    dev = {"requested": dict(R.REFERENCE), "resolved": {"detector": "cpu"}, "reset_fix": True}
    log = {"resets": [{"frame": 12, "stripped": 2}], "tracker_reid_env": None, "versions": dict(R.PINNED_VERSIONS)}
    return det, dev, log


def test_check_arm_records():
    det, dev, log = _arm_records()
    assert R.check_arm_records(det, dev, log, 1) == []
    assert R.check_arm_records(det, dev, log, 2)                         # count differs from the identity file
    for change in ({"reset_fix": None}, {"clip_fix": True}, {"requested": {"detector": "mps"}}):
        assert R.check_arm_records(det, {**dev, **change}, log, 1)
    assert R.check_arm_records({**det, "sha256": "x"}, dev, log, 1)
    assert R.check_arm_records(det, dev, {**log, "tracker_reid_env": "1"}, 1)
    assert R.check_arm_records(det, dev, {**log, "versions": {**R.PINNED_VERSIONS, "torch": "2.15.0"}}, 1)
    assert R.check_arm_records(det, dev, {**log, "resets": [{"frame": 12, "stripped": 4}]}, 1)
    assert R.check_arm_records(det, dev, {**log, "resets": [{"frame": None, "stripped": 2}]}, 1)
    assert R.check_arm_records(det, dev, None, 1)
    assert R.check_base_records(det, {"requested": dict(R.REFERENCE), "resolved": {"detector": "cpu"}}) == []
    assert R.check_base_records(det, dev)                                # a baseline must not carry the fix


def test_prefix_equal():
    base = [{"frame": f, "boxes": [f]} for f in range(0, 30, 3)]
    arm = [dict(r) for r in base]
    arm[6] = {"frame": 18, "boxes": ["new"]}
    assert R.prefix_equal(base, arm, 15)["equal"]
    v = R.prefix_equal(base, arm, 18)
    assert not v["equal"] and v["first_differing_frame"] == 18
    assert not R.prefix_equal(base, arm, None)["equal"]


def test_boxes_match():
    ref = [[486.21, 178.46, 541.55, 328.99], [894.55, 334.1, 999.21, 527.7]]
    side = [[894, 334, 999, 527], [486, 178, 541, 328]]                 # the sidecar truncates
    assert R.boxes_match(side, ref)
    assert not R.boxes_match(side[:1], ref)
    assert not R.boxes_match([[894, 334, 999, 527], [480, 178, 541, 328]], ref)


def test_one_to_one_missed_and_frame_extra():
    sv = np.array([[0.0, 0.0], [1.0, 0.0], [20.0, 0.0]] + [[100.0, 100.0]] * 7)
    inside = np.array([True, True, True] + [False] * 7)
    feet = np.array([[0.5, 0.0]])                    # one box near two players: nearest-foot finds both
    assert R.one_to_one_missed(feet, sv, inside) == 2
    assert R.one_to_one_missed(np.zeros((0, 2)), sv, inside) == 3
    e = R.frame_extra(np.array([[0.5, 0.0], [0.7, 0.0]]), sv, inside)
    assert e == {"far": 1, "missed_1to1": 1, "duplicates": 2}


def test_truth_pin_and_label_fix():
    files = {"a.jsonl": "1", "g_result.json": "2"}
    assert R.truth_pin_problems(None, files)
    assert R.truth_pin_problems({"status": "pending", "files": files}, files)
    assert R.truth_pin_problems({"status": "use", "files": files}, {**files, "a.jsonl": "3"})
    assert R.truth_pin_problems({"status": "use", "files": files}, files) == []
    ok = {"reset_fix": True, "resolved": {"detector": "cpu"}}
    assert R.label_fix_problems({"w1": ok, "w2": ok}) == []
    assert R.label_fix_problems({"w1": ok, "w2": {"resolved": {"detector": "cpu"}}})
    assert R.label_fix_problems({"w1": None})


def test_stage_allowed(tmp_path):
    import json
    assert R.stage_allowed("smoke", tmp_path)                              # version 2.2: stage 0 must pass first
    (tmp_path / "stage0.json").write_text(json.dumps({"passes": True}))
    assert R.stage_allowed("pool", tmp_path) and R.stage_allowed("hou", tmp_path) and R.stage_allowed("smoke", tmp_path) == []
    (tmp_path / "smoke_check.json").write_text(json.dumps({"passes": True}))
    # version 2.1: hou builds need only the smoke; hou scoring needs the pool pass
    assert R.stage_allowed("pool", tmp_path) == [] and R.stage_allowed("hou", tmp_path) == []
    assert R.stage_allowed("hou_score", tmp_path)
    (tmp_path / "report.json").write_text(json.dumps({"pool": {"verdict": {"pass": False}}}))
    assert R.stage_allowed("hou_score", tmp_path)
    (tmp_path / "report.json").write_text(json.dumps({"pool": {"verdict": {"pass": True}}}))
    assert R.stage_allowed("hou_score", tmp_path) == []


def test_rewrite_only_the_build_script(monkeypatch):
    env = {R.ENV: "/w.pt"}
    assert R.rewrite(["py", "build_trajectories.py", "--x"], env) == ["py", "-m", "tools.p8_resetfix", "child",
                                                                       "build_trajectories.py", "--x"]
    assert R.rewrite(["py", "other.py"], env) == ["py", "other.py"]
    assert R.rewrite(["py", "build_trajectories.py"], {}) == ["py", "build_trajectories.py"]


def _win(missed, far, m1, nr_g, nr_b, matched, inf=100):
    return {"missed": missed, "in_frame": inf, "far": far, "missed_1to1": m1, "duplicates": 1, "nr_ghosts": nr_g,
            "nr_boxes": nr_b, "cov_acc": 6.0, "cov_live": 8.0, "team": {"n": 10, "majority": 8}, "matched": matched}


def test_unit_eval_and_report_only():
    rng = np.random.default_rng(5)
    wins = []
    for i in range(10):
        m = {f"{f}:0": [float(rng.random()), 0.1, "near"] for f in range(6)}
        mn = dict(m)
        mn[f"99{i}:1"] = [2.0, 0.0, "far"]                               # a player matched only in the arm
        wins.append({"r23": _win(30, 10, 32, 5, 90, m), "new": _win(28, 9, 30, 5, 92, mn)})
    ev = R.unit_eval(wins, R.resample_picks(10, 200, 0))
    assert abs(ev["point"]["missed_rate"] + 0.02) < 1e-12 and abs(ev["point"]["detection_rate"] + 0.01) < 1e-12
    assert abs(ev["point"][R.ONE_TO_ONE_ROW] + 0.02) < 1e-12 and ev["window_interval"]["detection_rate"] is not None
    ro = R.report_only(wins)
    assert ro["arm_only_matches"]["n"] == 10 and ro["matched_only_in_baseline"] == 0
    assert abs(ro["new"]["boxes_per_in_frame"] - 0.92) < 1e-12
    v = R.v2_verdict(ev["point"], ev["window_interval"], ev["rows"]["coverage"], None, improve_rows=(), one_to_one=False)
    assert v["rows"]["position_p50_ft"]["pass"] and v["rows"]["missed_rate"]["pass"]


def test_cmd_report_on_synthetic_scores(tmp_path, monkeypatch):
    """The report reads pins, plans and score files and applies the v2 rule; the repository record write is caught."""
    import json
    from tools import p8_common
    root = tmp_path / "ref"
    (root / "builds").mkdir(parents=True)
    (root / "label_build").mkdir()
    d = root / "resetfix"
    d.mkdir()
    rng = np.random.default_rng(1)
    for g in R.GAMES:
        wins, common = {}, {}
        for i in range(8):
            m = {f"{f}:0": [float(rng.random()), 0.1, "near"] for f in range(6)}
            pair = {"r23": _win(30, 10, 32, 5, 90, m), "new": _win(27 + (i % 2), 8, 28, 5, 91, dict(m))}
            wins[f"{g}_s00_w{i:02d}"] = pair
            common[f"{g}_s00_w{i:02d}"] = pair
        (d / f"score__{g}.json").write_text(json.dumps({"windows": wins, "windows_common": common}))
        (d / f"plan__{g}.json").write_text(json.dumps({"reset_windows": [f"{g}_s00_w00"],
                                                      "baseline_resets": {f"{g}_s00_w00": 1}}))
        (d / f"truth_pin__{g}.json").write_text(json.dumps({"status": "use", "files": {}}))
    monkeypatch.setenv("V3_P8_ROOT", str(root))
    written = {}
    monkeypatch.setattr(p8_common, "atomic_write", lambda path, text: written.__setitem__(str(path), text))
    res = R.cmd_report(None)
    assert res["pool"]["verdict"]["pass"] is True and res["hou"]["verdict"]["pass"] is True
    assert res["pool"]["label"] == "the fix lowers net error and missed players on the pool; ghosts are reported as a cost"
    assert res["adoption"]["adopt"] is True and "ghost_cost_pooled" in res["pool"]
    assert res["bracket_common_frames"]["pool"]["verdict"]["pass"] is True
    assert any(k.endswith("p8_resetfix.json") for k in written)
    # a pool game whose truth is in use but not scored stops the report; one whose truth is not in use makes it partial
    (d / "score__gsw_sac_2015.json").unlink()
    with pytest.raises(SystemExit):
        R.cmd_report(None)
    (d / "truth_pin__gsw_sac_2015.json").write_text(json.dumps({"status": "apart", "files": {}}))
    res = R.cmd_report(None)
    assert res["partial"] and res["pool"]["verdict"]["pass"] is False
    assert res["hou"] is None                                            # no hou row without a pool pass


# ── version 2.1: the unfixed baseline in builds/r23_nofix and check C's builds/r23 ─────────────────────────────────
def _roots(tmp_path, monkeypatch):
    import json
    ws, ref = tmp_path / "work", tmp_path / "work_ref"
    for r in (ws, ref):
        (r / "label_build").mkdir(parents=True)
        (r / "builds").mkdir()
    monkeypatch.setenv("V3_P8_ROOT", str(ref))
    return ws, ref, json


def _ws_build(ws, ref, folder, stem, json, clip=b"clip", ref_clip=None, **dev):
    (ws / "label_build" / f"{stem}.mp4").write_bytes(clip)
    (ref / "label_build" / f"{stem}.mp4").write_bytes(clip if ref_clip is None else ref_clip)
    (ref / "label_build" / f"{stem}_timemap.json").write_text("{}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}_frames.json").write_text(json.dumps({"source": str(ws / "label_build" / f"{stem}.mp4"),
                                                            "frames": []}))
    (folder / f"{stem}_detector.json").write_text(json.dumps({"sha256": R.R23_SHA, "model": "r23"}))
    (folder / f"{stem}_device.json").write_text(json.dumps({"requested": dict(R.REFERENCE), **dev}))
    (folder / f"{stem}_identity.json").write_text(json.dumps({"identity": {"camera_cut_resets": 0}}))


def test_ref_root_refuses_the_workspace(tmp_path, monkeypatch):
    ws, ref, _ = _roots(tmp_path, monkeypatch)
    assert R.ref_root() == ref
    monkeypatch.setenv("V3_P8_ROOT", str(ws))
    with pytest.raises(SystemExit):
        R.ref_root()
    monkeypatch.delenv("V3_P8_ROOT")
    with pytest.raises(SystemExit):
        R.ref_root()


def test_baseline_problems(tmp_path, monkeypatch):
    ws, ref, json = _roots(tmp_path, monkeypatch)
    src = ws / "builds" / "m9"
    _ws_build(ws, ref, src, "g_s00_w00", json)
    assert R.baseline_problems("g_s00_w00", src, ws, ref) is None
    _ws_build(ws, ref, src, "g_s00_w01", json, ref_clip=b"other")
    assert R.baseline_problems("g_s00_w01", src, ws, ref) == "clips differ"
    _ws_build(ws, ref, src, "g_s00_w02", json, reset_fix=True)
    assert R.baseline_problems("g_s00_w02", src, ws, ref) == "built with a fix"
    _ws_build(ws, ref, src, "g_s00_w03", json, clip_fix=True)
    assert R.baseline_problems("g_s00_w03", src, ws, ref) == "built with a fix"
    (src / "g_s00_w00_detector.json").write_text(json.dumps({"sha256": "x", "model": "r23"}))
    assert R.baseline_problems("g_s00_w00", src, ws, ref) == "detector record is not r23"
    assert R.baseline_problems("g_s09_w09", src, ws, ref) == "no build"


def test_baseline_copy(tmp_path, monkeypatch):
    ws, ref, json = _roots(tmp_path, monkeypatch)
    from tools.p8_placement import BUILDS
    g = "gsw_cha_2016"
    src = ws / "builds" / BUILDS[g]
    for i in range(3):
        _ws_build(ws, ref, src, f"{g}_s00_w{i:02d}", json)
    rec = R.cmd_baseline_copy(type("A", (), {"game": g, "workspace": str(ws)})())
    assert rec["complete"] and len(rec["copied"]) == 3
    assert (ref / "builds" / "r23_nofix" / f"{g}_s00_w01_frames.json").exists()
    assert not (ref / "builds" / "r23").exists()                           # check C's folder is not written here
    assert R.cmd_baseline_copy(type("A", (), {"game": g, "workspace": str(ws)})())["present"] == rec["copied"]
    _ws_build(ws, ref, src, f"{g}_s00_w03", json, ref_clip=b"other")       # a refused window: incomplete
    rec = R.cmd_baseline_copy(type("A", (), {"game": g, "workspace": str(ws)})())
    assert not rec["complete"] and f"{g}_s00_w03" in rec["refused"]


# ── version 2.2 ───────────────────────────────────────────────────────────────────────────────────────────────────
def test_compare_window_ignores_only_source(tmp_path):
    """A1: two sidecars that differ only in "source" compare equal; any other difference does not."""
    import json
    base, arm = tmp_path / "base", tmp_path / "arm"
    for d, src in ((base, "/root_a/label_build/w.mp4"), (arm, "/root_b/label_build/w.mp4")):
        d.mkdir()
        (d / "w_frames.json").write_text(json.dumps({"source": src, "fps": 59.94, "frames": [{"frame": 0, "boxes": []}]}))
        (d / "w_trajectories.json").write_text("{}")
        (d / "w_identity.json").write_text(json.dumps({"identity": {"camera_cut_resets": 0}}))
    c = R.compare_window(base, arm, "w")
    assert all(c["byte_identical"].values())
    (arm / "w_frames.json").write_text(json.dumps({"source": "/x.mp4", "fps": 59.94, "frames": [{"frame": 0, "boxes": [1]}]}))
    assert not R.compare_window(base, arm, "w")["byte_identical"]["frames"]


def test_arm_source_problem(tmp_path):
    ws, ref = tmp_path / "work", tmp_path / "work_ref"
    for r in (ws, ref):
        (r / "label_build").mkdir(parents=True)
        (r / "label_build" / "w.mp4").write_bytes(b"clip")
    assert R.arm_source_problem({"source": str(ref / "label_build" / "w.mp4")}, "w", ref, ws) is None
    assert R.arm_source_problem({"source": str(ws / "label_build" / "w.mp4")}, "w", ref, ws)
    assert R.arm_source_problem({}, "w", ref, ws)
    (ws / "label_build" / "w.mp4").write_bytes(b"other")
    assert R.arm_source_problem({"source": str(ref / "label_build" / "w.mp4")}, "w", ref, ws)


def test_boxes_consistency():
    a = [{"frame": 0, "boxes": [{"bbox": [1, 2, 3, 4], "tid": 1}]}, {"frame": 6, "boxes": []}, {"frame": 12, "boxes": []}]
    b = [{"frame": 0, "boxes": [{"bbox": [1, 2, 3, 4], "tid": 1}]}, {"frame": 6, "boxes": [{"bbox": [0, 0, 1, 1], "tid": 2}]}]
    assert R.boxes_consistency(a, b) == {"frames_shared": 2, "frames_equal": 1, "frames_only_arm": 1, "frames_only_label": 0}


def test_pin_order_problems():
    assert R.pin_order_problems([10.0, 20.0], {"a.jsonl": 30.0, "acc": 31.0}, 2, 2) == []
    assert R.pin_order_problems([10.0, 20.0], {"a.jsonl": 15.0}, 2, 2)      # a label file older than a build
    assert R.pin_order_problems([10.0], {"a.jsonl": None}, 1, 1)
    assert R.pin_order_problems([10.0], {"a.jsonl": 30.0}, 73, 72)          # tally differs
    assert R.pin_order_problems([], {"a.jsonl": 30.0}, 0, 0)


def test_clip_changed(tmp_path):
    (tmp_path / "label_build").mkdir()
    (tmp_path / "label_build" / "w.mp4").write_bytes(b"clip")
    plan = {"clip_sha256": {"w": R.sha256_file(tmp_path / "label_build" / "w.mp4")}}
    assert not R.clip_changed(plan, tmp_path, "w")
    (tmp_path / "label_build" / "w.mp4").write_bytes(b"new")
    assert R.clip_changed(plan, tmp_path, "w") and R.clip_changed({}, tmp_path, "w")


def test_adoption():
    pooled_ok = {"pass": True}
    assert R.adoption({"pass": True, "failing": []}, pooled_ok, True) == {"adopt": True, "kind": "pass"}
    a = R.adoption({"pass": False, "failing": ["gsw_cha_2016:ghost_rate"]}, pooled_ok, True)
    assert a["adopt"] and a["ghost_failing_games"] == ["gsw_cha_2016"]
    assert not R.adoption({"pass": False, "failing": ["gsw_cha_2016:ghost_rate"]}, pooled_ok, False)["adopt"]
    assert not R.adoption({"pass": False, "failing": ["pool:ghost_rate"]}, {"pass": False}, True)["adopt"]
    assert not R.adoption({"pass": False, "failing": ["gsw_cha_2016:ghost_rate", "gsw_bkn_2015:position_p90_ft"]},
                          pooled_ok, True)["adopt"]
    assert not R.adoption({"pass": False, "partial": True, "failing": []}, pooled_ok, True)["adopt"]


def test_scale_after_and_excess_counts():
    st = {"in_frame": 100, "missed": 30, "far": 10, "nr_ghosts": 5, "nr_boxes": 80, "cov_acc": 1.0}
    aft = {"in_frame": 10, "missed": 6, "far": 2, "nr_ghosts": 1, "nr_boxes": 4, "reach": 3, "far_reach": 1,
           "ours_boxes": 9, "ours_ghosts": 3}
    st2, aft2 = R.scale_after(st, aft, 3.0)
    assert st2["in_frame"] == 120 and st2["missed"] == 42 and st2["cov_acc"] == 1.0
    assert aft2["reach"] == 9 and aft2["ours_boxes"] == 27
    assert R.scale_after(st, aft, 1.0) == (st, aft)
    assert R.excess_ghost_counts(aft) == (2, 5)


def test_stage0_checks():
    import types
    mod = _tracker_module()
    det = types.SimpleNamespace(reset=1, update=1)
    ok = R.stage0_checks(mod, dict(R.PINNED_VERSIONS), det, R.R23_SHA)
    assert all(ok.values())
    assert not R.stage0_checks(mod, {**R.PINNED_VERSIONS, "torch": "2.15.0"}, det, R.R23_SHA)["versions_pinned"]
    assert not R.stage0_checks(mod, dict(R.PINNED_VERSIONS), det, "x")["detector_is_r23"]


def test_report_bracket_guard(tmp_path, monkeypatch):
    """A2: when the bracket's pooled missed does not improve, the pool does not pass and nothing is adopted."""
    import json
    from tools import p8_common
    root = tmp_path / "ref"
    (root / "builds").mkdir(parents=True)
    (root / "label_build").mkdir()
    d = root / "resetfix"
    d.mkdir()
    rng = np.random.default_rng(2)
    for g in R.POOL:
        wins, common = {}, {}
        for i in range(8):
            m = {f"{f}:0": [float(rng.random()), 0.1, "near"] for f in range(6)}
            wins[f"{g}_s00_w{i:02d}"] = {"r23": _win(30, 10, 32, 5, 90, m), "new": _win(27, 8, 28, 5, 91, dict(m))}
            common[f"{g}_s00_w{i:02d}"] = {"r23": _win(30, 10, 32, 5, 90, m), "new": _win(31, 10, 32, 5, 91, dict(m))}
        (d / f"score__{g}.json").write_text(json.dumps({"windows": wins, "windows_common": common}))
        (d / f"plan__{g}.json").write_text(json.dumps({"reset_windows": [], "baseline_resets": {}}))
        (d / f"truth_pin__{g}.json").write_text(json.dumps({"status": "use", "files": {}}))
    monkeypatch.setenv("V3_P8_ROOT", str(root))
    monkeypatch.setattr(p8_common, "atomic_write", lambda path, text: None)
    res = R.cmd_report(None)
    assert res["pool"]["verdict"]["pass"] is False
    assert res["bracket_guard"]["pooled_missed_point_below_zero"] is False
    assert res["adoption"]["adopt"] is False and res["pool"]["label"] is None and res["hou"] is None


# ── version 2.3: the net-error rule ───────────────────────────────────────────────────────────────────────────────
def _v23_good():
    point = {"net_error_rate": -0.02, "missed_rate": -0.03, "missed_1to1_rate": -0.02, "ghost_rate": 0.02,
             "detection_rate": -0.01, "position_p50_ft": 0.0, "position_p90_ft": 0.02, "nearfield_bias_ft": 0.0,
             "team_accuracy": 0.0}
    wiv = {"net_error_rate": [-0.04, -0.001], "missed_rate": [-0.05, -0.01], "ghost_rate": [0.01, 0.05],
           "detection_rate": [-0.02, 0.0], "position_p50_ft": [-0.03, 0.04], "position_p90_ft": [-0.05, 0.09],
           "nearfield_bias_ft": [-0.05, 0.05], "team_accuracy": [-0.02, 0.02]}
    return point, wiv


def test_v23_verdict_ghosts_reported_not_gated():
    point, wiv = _v23_good()
    v = R.v23_verdict(point, wiv, (0.8, 0.8), "pool")
    assert v["pass"] and v["rows"]["ghost_rate"]["gated"] is False         # a ghost rise far over +0.010 passes
    for row, iv in (("net_error_rate", [-0.04, 0.001]), ("missed_rate", [-0.05, 0.0])):
        w = dict(wiv); w[row] = iv
        assert not R.v23_verdict(point, w, (0.8, 0.8), "pool")["pass"]       # pooled: interval wholly below zero
        assert R.v23_verdict(point, w, (0.8, 0.8), "game")["pass"]           # a game: only not worse
    w = dict(wiv); w["net_error_rate"] = [0.001, 0.02]
    assert not R.v23_verdict(point, w, (0.8, 0.8), "game")["pass"]
    p = dict(point); p["missed_1to1_rate"] = 0.0
    assert not R.v23_verdict(p, wiv, (0.8, 0.8), "pool")["pass"] and R.v23_verdict(p, wiv, (0.8, 0.8), "game")["pass"]
    for row, iv in (("position_p90_ft", [0.0, 0.11]), ("team_accuracy", [-0.031, 0.0])):
        w = dict(wiv); w[row] = iv
        assert not R.v23_verdict(point, w, (0.8, 0.8), "game")["pass"]
    p = dict(point); p["missed_rate"] = 0.001
    w = dict(wiv); w["missed_rate"] = [-0.01, 0.01]
    assert not R.v23_verdict(p, w, (0.8, 0.8), "hou")["pass"]                # hou: missed must improve


def test_v23_pool_and_bracket_guard():
    point, wiv = _v23_good()
    ok = R.v23_verdict(point, wiv, (0.8, 0.8), "pool")
    game = R.v23_verdict(point, wiv, (0.8, 0.8), "game")
    good = {"missed_rate": -0.01, "net_error_rate": -0.005}
    assert R.v23_pool(ok, {"a": game}, good)["pass"]
    assert not R.v23_pool(ok, {"a": game}, {"missed_rate": -0.01, "net_error_rate": 0.001})["pass"]
    assert not R.v23_pool(ok, {"a": game}, None)["pass"]
    assert not R.v23_pool(ok, {"a": game}, good, partial=True)["pass"]
    w = dict(wiv); w["position_p50_ft"] = [0.0, 0.2]
    v = R.v23_pool(ok, {"a": R.v23_verdict(point, w, (0.8, 0.8), "game")}, good)
    assert not v["pass"] and "a:position_p50_ft" in v["failing"]


def test_net_error_rows_in_unit_eval():
    rng = np.random.default_rng(7)
    wins = []
    for _ in range(6):
        m = {"0:0": [float(rng.random()), 0.1, "near"]}
        wins.append({"r23": _win(30, 10, 32, 5, 90, m), "new": _win(26, 8, 28, 7, 95, dict(m))})
    ev = R.unit_eval(wins, R.resample_picks(6, 100, 0), R.resample_picks(2, 100, 0), [wins[:3], wins[3:]])
    assert abs(ev["point"]["net_error_rate"] - ((26 + 7) - (30 + 5)) / 100) < 1e-12
    assert ev["game_interval"]["net_error_rate"] is not None


# ── re-review B1: a source problem and a record problem are both kept ────────────────────────────────────────────
def test_arm_window_problems_keeps_every_problem(tmp_path):
    import json
    ws, ref = tmp_path / "work", tmp_path / "work_ref"
    for r in (ws, ref):
        (r / "label_build").mkdir(parents=True)
        (r / "label_build" / "w.mp4").write_bytes(b"clip")
    out = ref / "builds" / "arm"
    out.mkdir(parents=True)
    det, dev, log = _arm_records()
    (out / "w_frames.json").write_text(json.dumps({"source": str(ref / "label_build" / "w.mp4"), "frames": []}))
    (out / "w_detector.json").write_text(json.dumps(det))
    (out / "w_device.json").write_text(json.dumps(dev))
    (out / "w_resetfix.json").write_text(json.dumps(log))
    (out / "w_identity.json").write_text(json.dumps({"identity": {"camera_cut_resets": 1}}))
    assert R.arm_window_problems(out, "w", ref, ws) == []
    (out / "w_frames.json").write_text(json.dumps({"source": str(ws / "label_build" / "w.mp4"), "frames": []}))
    (out / "w_detector.json").write_text(json.dumps({**det, "sha256": "x"}))
    bad = R.arm_window_problems(out, "w", ref, ws)
    assert any("reference root's clip" in b for b in bad) and any("r23" in b for b in bad)


# ── version 2.4 ───────────────────────────────────────────────────────────────────────────────────────────────────
def test_reset_log_is_shared_with_a_second_module_copy(monkeypatch, tmp_path):
    """The build child runs the file as __main__ while tools.p8_device imports tools.p8_resetfix: the log started in
    the __main__ copy must hold the resets the canonical patch records (the first smoke wrote an empty log)."""
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("p8_resetfix_main_copy", R.__file__)
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    assert copy is not R
    mod = _tracker_module()
    fake_dt = types.ModuleType("detect.tracker")
    fake_dt.PlayerTracker = type("PlayerTracker", (FakeTracker,), {})
    pkg = types.ModuleType("detect")
    pkg.tracker = fake_dt
    monkeypatch.setitem(sys.modules, "detect", pkg)
    monkeypatch.setitem(sys.modules, "detect.tracker", fake_dt)
    monkeypatch.setattr(R, "RESET_LOG", {"path": None, "resets": [], "last_frame": None})
    registered = []
    monkeypatch.setattr("atexit.register", registered.append)
    copy.start_reset_log(tmp_path / "w_resetfix.json")          # as child() does in __main__
    R.apply_reset_fix()                                         # as tools.p8_device.device_record does
    t = fake_dt.PlayerTracker(mod)
    t.update(None, 0)
    t.reset()
    t.update(None, 6)
    t.reset()
    assert registered and registered[0] is R.write_reset_log
    registered[0]()
    log = json.loads((tmp_path / "w_resetfix.json").read_text())
    assert log["resets"] == [{"frame": 0, "stripped": 2}, {"frame": 6, "stripped": 2}]
    assert copy.RESET_LOG["resets"] == []                       # the copy's own dict stays unused


def test_label_fix_problems_checks_reset_logs():
    ok_dev = {"reset_fix": True, "resolved": {"detector": "cpu"}}
    log = {"resets": [{"frame": 30, "stripped": 2}], "tracker_reid_env": None, "versions": dict(R.PINNED_VERSIONS)}
    assert R.label_fix_problems({"w": ok_dev}, {"w": log}, {"w": 1}) == []
    assert R.label_fix_problems({"w": ok_dev}, {"w": None}, {"w": 1})                      # no log
    assert R.label_fix_problems({"w": ok_dev}, {"w": log}, {"w": 2})                       # count differs
    assert R.label_fix_problems({"w": ok_dev}, {"w": {**log, "resets": []}}, {"w": 1})     # the empty-log case
    assert R.label_fix_problems({"w": ok_dev}, {"w": {**log, "tracker_reid_env": "1"}}, {"w": 1})
    assert R.label_fix_problems({"w": ok_dev}, {"w": {**log, "versions": {}}}, {"w": 1})
    assert R.label_fix_problems({"w": ok_dev}) == []                                      # device check alone


def test_unmatched_1to1():
    sv = np.array([[0.0, 0.0], [1.0, 0.0], [20.0, 0.0], [-30.0, 0.0]] + [[100.0, 100.0]] * 6)
    inside = np.array([True, True, True, False] + [False] * 6)
    feet = np.array([[0.5, 0.0], [0.6, 0.0], [40.0, 0.0], [-30.5, 0.0], [1.0, 1.0]])
    is_ref = [False, False, False, False, True]
    # p0 and p1 each take one box; p2 unmatched; the box at 40 unmatched; the box on the out-of-frame player is
    # matched; the referee box is left out
    assert R.unmatched_1to1(feet, sv, inside, is_ref) == 2
    e = R.frame_extra(feet[:3], sv, inside, is_ref[:3])
    assert e["unmatched_1to1"] == 2 and "unmatched_1to1" not in R.frame_extra(feet[:3], sv, inside)
    assert R.unmatched_1to1(np.zeros((0, 2)), sv, inside, []) == 3
